"""Original-grid probability restoration and selected fixed ensemble."""
import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

MEAN=[.485,.456,.406]
STD=[.229,.224,.225]

def flip(image,variant):
    if variant & 1: image=TF.hflip(image)
    if variant & 2: image=TF.vflip(image)
    return image

def canvas(image,variant,scale):
    if image.size != (672,378): raise ValueError("Expected 672x378 RGB image")
    width,height=round(672*scale),round(384*scale)
    original=Image.new("RGB",(672,384),(0,0,0)); original.paste(flip(image,variant))
    resized=original.resize((width,height),Image.Resampling.BILINEAR)
    padded=Image.new("RGB",(math.ceil(width/32)*32,math.ceil(height/32)*32),(0,0,0));padded.paste(resized)
    return TF.normalize(TF.to_tensor(padded),MEAN,STD)[None],width,height

@torch.no_grad()
def probability(model,image,scale=1.25):
    device=next(model.parameters()).device
    restored=[]
    for variant in range(4):
        x,width,height=canvas(image,variant,scale)
        logits=model(x.to(device))
        if model.arm=="selective_refine":
            if scale != 1.25: raise ValueError("Native refinement requires its trained 1.25x input")
            native=model.last_native_probability
            if native.shape != (1,1,378,672): raise RuntimeError("Native refinement shape mismatch")
        else:
            native=F.interpolate(logits.sigmoid()[:,:,:height,:width],(384,672),mode="bilinear",align_corners=False)[:,:,:378]
        if variant & 1: native=native.flip(-1)
        if variant & 2: native=native.flip(-2)
        restored.append(native)
    array=torch.stack(restored).mean(0)[0,0].cpu().numpy()
    if array.shape != (378,672) or not np.isfinite(array).all(): raise RuntimeError("Invalid native probability")
    return array

def ensemble(models,image):
    multi=(probability(models["full"],image,1.)+probability(models["full"],image,1.25)+probability(models["full"],image,1.5))/3
    refine=probability(models["selective_refine"],image)
    gate=probability(models["direction_gate"],image)
    return .6*multi+.2*refine+.2*gate
