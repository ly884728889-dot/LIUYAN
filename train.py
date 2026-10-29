"""Independent Train/Val reproduction; never use Benchmark for model selection."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as TF
from soclr.models import build_model
from soclr.inference import MEAN,STD,flip,probability
from soclr.losses import improved_loss
from soclr.orientation import orientation_gt

class Images(Dataset):
    def __init__(self,rows,root,training):self.rows,self.root,self.training=rows,root,training
    def __len__(self):return len(self.rows)*(4 if self.training else 1)
    def __getitem__(self,index):
        row=self.rows[index//4 if self.training else index]
        image=Image.open(self.root/row['image_path']).convert('RGB')
        mask=Image.open(self.root/row['mask_path']).convert('L')
        assert image.size==mask.size==(672,378)
        if not self.training:return str(self.root/row['image_path']),torch.from_numpy((np.asarray(mask)>0).copy())[None],row['image_id']
        variant=index%4
        rgb=Image.new('RGB',(672,384));rgb.paste(flip(image,variant));rgb=rgb.resize((840,480),Image.Resampling.BILINEAR)
        target=Image.new('L',(672,384));target.paste(flip(mask,variant));target=target.resize((840,480),Image.Resampling.NEAREST)
        valid=Image.new('L',(672,384));valid.paste(Image.new('L',(672,378),255));valid=valid.resize((840,480),Image.Resampling.NEAREST)
        outputs=[]
        for item,mode in [(rgb,'RGB'),(target,'L'),(valid,'L')]:
            canvas=Image.new(mode,(864,480));canvas.paste(item);outputs.append(canvas)
        x=TF.normalize(TF.to_tensor(outputs[0]),MEAN,STD)
        y=torch.from_numpy((np.asarray(outputs[1])>0).copy()).float()[None]
        v=torch.from_numpy((np.asarray(outputs[2])>0).copy()).float()[None]
        return x,y,v,row['image_id'],variant

@torch.no_grad()
def validate(model,loader):
    model.eval();records=[]
    for paths,targets,names in loader:
        pred=probability(model,Image.open(paths[0]).convert('RGB'))>=.5
        truth=targets[0,0].numpy()
        tp=int(np.count_nonzero(pred & truth));fp=int(np.count_nonzero(pred & ~truth));fn=int(np.count_nonzero(~pred & truth))
        records.append(dict(image=names[0],dice=(2*tp+1)/(2*tp+fp+fn+1),fp=fp,fn=fn))
    return float(np.mean([r['dice'] for r in records])),records

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--train-list',type=Path,required=True)
    parser.add_argument('--val-list',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['full','selective_refine','direction_gate'],required=True)
    parser.add_argument('--encoder-path',type=Path)
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--start-epoch',type=int,default=1)
    parser.add_argument('--last-epoch',type=int,default=50)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Independent new run directory required')
    args.output.mkdir(parents=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(16)
    with args.inventory.open(newline='',encoding='utf-8-sig') as stream:inventory={r['image_id']:r for r in csv.DictReader(stream)}
    train=[inventory[n] for n in args.train_list.read_text(encoding='utf-8').splitlines()]
    val=[inventory[n] for n in args.val_list.read_text(encoding='utf-8').splitlines()]
    assert not ({r['group_id'] for r in train} & {r['group_id'] for r in val})
    model=build_model(args.arm).cuda()
    if args.warm_start:model.load_state_dict(torch.load(args.warm_start,map_location='cpu',weights_only=True),strict=True)
    else:
        if args.encoder_path is None:raise ValueError('Use the original ImageNet MiT-B2 encoder checkpoint')
        model.core.encoder.load_state_dict(torch.load(args.encoder_path,map_location='cpu',weights_only=True))
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(Images(train,args.data_root,True),batch_size=2,shuffle=True,generator=generator,num_workers=0)
    val_loader=DataLoader(Images(val,args.data_root,False),batch_size=1,num_workers=0)
    if len(loader)%2:raise ValueError('Require an even microstep count for accumulation=2')
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=0)
    history=[];best=-1.;cache={};began=time.monotonic()
    def checkpoint(epoch,step,total,gen):
        temporary=args.output/'last.pt.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,microstep=step,total_loss=total,history=history,best=best,epoch_generator_state=gen,rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),temporary)
        temporary.replace(args.output/'last.pt')
    for epoch in range(args.start_epoch,args.last_epoch+1):
        lr=1e-5+(.01-1e-5)*(1+math.cos(math.pi*(epoch-1)/50))/2
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True);total=0.;gen=generator.get_state().clone()
        for step,(x,y,valid,names,variants) in enumerate(loader,1):
            distributions=[];supports=[]
            for target,name,variant in zip(y,names,variants):
                key=(name,int(variant))
                if key not in cache:
                    d,s,_,_=orientation_gt(target[0].numpy()>=.5,stride=8,bins=8,radius=5,min_coherence=.3)
                    cache[key]=(torch.from_numpy(d.copy()).float(),torch.from_numpy(s.copy()).float())
                d,s=cache[key];distributions.append(d);supports.append(s)
            direction=torch.stack(distributions).cuda();active=torch.stack(supports)[:,None].cuda()
            x,y,valid=x.cuda(),y.cuda(),valid.cuda()
            logits=model(x);loss=improved_loss(model,logits,y,valid,direction,active)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/2).backward()
            if step%2==0:optimizer.step();optimizer.zero_grad(set_to_none=True)
            total+=float(loss.detach())
            if step%200==0:checkpoint(epoch,step,total,gen)
        dice,records=validate(model,val_loader)
        if dice>best:
            best=dice;torch.save(model.state_dict(),args.output/'best.pt')
            (args.output/'best_per_image.json').write_text(json.dumps(records),encoding='utf-8')
        record=dict(epoch=epoch,train_loss=total/len(loader),dice=dice,best_dice=best,lr=lr,elapsed_seconds=time.monotonic()-began)
        history.append(record);checkpoint(epoch+1,0,0.,generator.get_state().clone())
        (args.output/'epochs.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(status='complete',arm=args.arm,start_epoch=args.start_epoch,last_epoch=args.last_epoch,seed=42,train_original=len(train),val_original=len(val),warm_start=str(args.warm_start) if args.warm_start else None,best_dice=best,scope='Independent reproduction; no bitwise historical training guarantee'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
