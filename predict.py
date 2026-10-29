"""Export one fixed competition submission; never tune on Benchmark."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
import torch
from soclr.models import build_model
from soclr.inference import ensemble

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def check(raw,name):
    with Image.open(raw) as image:
        assert image.format=="PNG" and image.mode=="L" and image.size==(672,378),name
        assert set(np.unique(np.asarray(image))).issubset({0,1}),name

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--images",type=Path,required=True)
    parser.add_argument("--weights-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--expected-count",type=int,default=300)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Use a new output directory")
    config=json.loads((Path(__file__).parent/"configs/final.json").read_text(encoding="utf-8"))
    torch.set_num_threads(16)
    models={}
    for arm,expected in config["checkpoint_sha256"].items():
        path=args.weights_dir/(arm+".pt")
        if digest(path)!=expected:raise RuntimeError("Checkpoint hash mismatch: "+arm)
        model=build_model(arm).to(args.device).eval()
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        models[arm]=model
    images=sorted(path for path in args.images.iterdir() if path.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"))
    if len(images)!=args.expected_count:raise ValueError("Unexpected input count")
    names=[path.stem+".png" for path in images]
    if len(set(names))!=len(names):raise ValueError("Duplicate output stems")
    output=args.output/"result";output.mkdir(parents=True)
    started=time.monotonic()
    for index,path in enumerate(images,1):
        mask=(ensemble(models,Image.open(path).convert("RGB"))>=config["threshold"]).astype(np.uint8)
        Image.fromarray(mask).save(output/(path.stem+".png"))
        if index%20==0:print(f"Predicted {index}/{len(images)}",flush=True)
    for name in names:
        with (output/name).open("rb") as stream:check(stream,name)
    archive_path=args.output/"result.zip"
    with zipfile.ZipFile(archive_path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(names):archive.write(output/name,arcname="result/"+name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())=={"result/"+name for name in names}
        for name in archive.namelist():check(io.BytesIO(archive.read(name)),name)
    report=dict(status="complete",count=len(images),seconds=time.monotonic()-started,threshold=config["threshold"],zip_sha256=digest(archive_path),pixel_values=[0,1],dimensions=[672,378])
    (args.output/"prediction.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
