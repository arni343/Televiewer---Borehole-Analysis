"""Trainable U-Net segmentation and its sinusoidal-Hough interface.
Mask values: 0 background, 255 target trace, 128 unknown/ignored.
Training uses clean, full-circumference image tracks and borehole-disjoint splits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F

class PeriodicConv(nn.Module):
    """Circular azimuth padding; replicated depth padding."""
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv=nn.Conv2d(channels_in,channels_out,3)
    def forward(self,x):
        return self.conv(F.pad(F.pad(x,(1,1,0,0),mode="circular"),(0,0,1,1),mode="replicate"))

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.layers=nn.Sequential(PeriodicConv(cin,cout),nn.GroupNorm(4,cout),nn.ReLU(),
                                  PeriodicConv(cout,cout),nn.GroupNorm(4,cout),nn.ReLU())
    def forward(self,x):
        return self.layers(x)

class UNet(nn.Module):
    """Three-level encoder/decoder U-Net with skip connections and one logit/pixel.
    Four inputs: normalized RGB plus an explicit validity channel.
    """
    def __init__(self,base=16):
        super().__init__()
        if base<4 or base%4:
            raise ValueError("base must be a positive multiple of four, at least four.")
        self.base=base
        self.e1=Block(4,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4)
        self.bottom=Block(base*4,base*8)
        self.d3=Block(base*12,base*4); self.d2=Block(base*6,base*2); self.d1=Block(base*3,base)
        self.head=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(F.max_pool2d(e1,2,ceil_mode=True))
        e3=self.e3(F.max_pool2d(e2,2,ceil_mode=True))
        z=self.bottom(F.max_pool2d(e3,2,ceil_mode=True))
        for skip,decoder in ((e3,self.d3),(e2,self.d2),(e1,self.d1)):
            z=decoder(torch.cat([F.interpolate(z,size=skip.shape[-2:],mode="bilinear",align_corners=False),skip],1))
        return self.head(z)

def input_tensor(rgb,valid):
    a=rgb.astype(np.float32)/255.
    a=np.where(valid[:,:,None],a,0)
    return torch.from_numpy(np.concatenate((a,valid[:,:,None].astype(np.float32)),2).transpose(2,0,1).copy())

def masked_loss(logits,target,valid):
    if valid.sum().item()==0:
        raise ValueError("A training batch must contain labelled pixels.")
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*valid).sum()/valid.sum()
    prob=torch.sigmoid(logits)*valid
    truth=target*valid
    dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
    return bce+dice

def load_manifest(path, require_training=True):
    path=Path(path).resolve()
    obj=json.loads(path.read_text(encoding="utf-8-sig"))
    if obj.get("dataset_kind") not in ("rgl","synthetic"):
        raise ValueError("dataset_kind must explicitly be rgl or synthetic.")
    samples=obj.get("samples",[])
    if not samples:
        raise ValueError("Dataset has no labelled samples.")
    boreholes={}; images={}
    for s in samples:
        if s["split"] not in ("train","val","test"):
            raise ValueError("split must be train, val, or test.")
        if s.get("annotated_input",False):
            raise ValueError("Do not train on report images containing expert-pick overlays.")
        if s.get("channel") not in ("optical","acoustic"):
            raise ValueError("channel must be optical or acoustic.")
        b=str(s["borehole"])
        if b in boreholes and boreholes[b]!=s["split"]:
            raise ValueError("Borehole leakage: all related crops/modalities must share one split.")
        boreholes[b]=s["split"]
        for key in ("image","mask"):
            s[key]=str((path.parent/s[key]).resolve())
            if not Path(s[key]).is_file():
                raise ValueError(f"Missing {key}: {s[key]}")
        digest=hashlib.sha256(Path(s["image"]).read_bytes()).hexdigest()
        if digest in images and images[digest]!=s["split"]:
            raise ValueError("An identical image is present in different splits.")
        images[digest]=s["split"]
    if require_training and not {"train","val"}<=set(boreholes.values()):
        raise ValueError("Use separate training and validation boreholes.")
    obj["manifest_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    return obj

def read_pair(sample):
    rgb=np.asarray(Image.open(sample["image"]).convert("RGB"))
    mask=np.asarray(Image.open(sample["mask"]).convert("L"))
    if mask.shape!=rgb.shape[:2] or min(mask.shape)<8:
        raise ValueError("Image/mask shapes must match and have at least eight rows/columns.")
    if not np.isin(mask,[0,128,255]).all():
        raise ValueError("Masks must contain only 0, 128 (ignore), and 255.")
    valid=mask!=128
    return rgb,mask==255,valid

class TraceDataset(torch.utils.data.Dataset):
    """Depth tiles retain the full native azimuth width. Batch size is one."""
    def __init__(self,samples,tile_rows=256,augment=False):
        self.samples=samples; self.tile_rows=tile_rows; self.augment=augment; self.tiles=[]
        for i,s in enumerate(samples):
            _,_,v=read_pair(s)
            for lo in range(0,len(v),tile_rows):
                if v[lo:lo+tile_rows].any():
                    self.tiles.append((i,lo))
        if not self.tiles:
            raise ValueError("Split contains no labelled pixels.")
    def __len__(self):
        return len(self.tiles)
    def __getitem__(self,index):
        i,lo=self.tiles[index]
        rgb,target,valid=read_pair(self.samples[i])
        rgb=rgb[lo:lo+self.tile_rows].copy(); target=target[lo:lo+self.tile_rows].copy(); valid=valid[lo:lo+self.tile_rows].copy()
        pad=self.tile_rows-len(rgb)
        if pad:
            rgb=np.pad(rgb,((0,pad),(0,0),(0,0)),mode="edge")
            target=np.pad(target,((0,pad),(0,0))); valid=np.pad(valid,((0,pad),(0,0)))
        if self.augment:
            shift=random.randrange(rgb.shape[1])
            rgb=np.roll(rgb,shift,axis=1); target=np.roll(target,shift,axis=1); valid=np.roll(valid,shift,axis=1)
            gain=random.uniform(.85,1.15)
            rgb=np.clip(rgb.astype(float)*gain,0,255).astype(np.uint8)
            # Match missing/header boundary masks encountered during report inference.
            # Dropped rows are ignored in the loss, never converted to background labels.
            border=random.randrange(min(9,len(rgb)//4))
            if border:
                valid[:border]=False
                valid[-border:]=False
        return input_tensor(rgb,valid),torch.from_numpy(target[None].astype(np.float32)),torch.from_numpy(valid[None].astype(np.float32))

def metrics_from_counts(tp,fp,fn,tn):
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)

def validate(model,loader,device):
    model.eval(); total_loss=0.; counts=np.zeros(4,dtype=np.int64)
    with torch.inference_mode():
        for x,y,v in loader:
            x,y,v=x.to(device),y.to(device),v.to(device)
            logits=model(x); total_loss+=float(masked_loss(logits,y,v))
            pred=logits>=0; truth=y>.5; mask=v>.5
            counts+=np.array([int((pred&truth&mask).sum()),int((pred&~truth&mask).sum()),
                              int((~pred&truth&mask).sum()),int((~pred&~truth&mask).sum())])
    return dict(loss=total_loss/len(loader),**metrics_from_counts(*map(int,counts)))

def train(args):
    if args.epochs<1 or args.tile_rows<16 or args.patience<1:
        raise ValueError("epochs/patience must be positive; tile_rows must be at least 16.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dataset=load_manifest(args.manifest)
    training=[s for s in dataset["samples"] if s["split"]=="train"]
    validation=[s for s in dataset["samples"] if s["split"]=="val"]
    for split in (training,validation):
        if not any(read_pair(s)[1].any() for s in split):
            raise ValueError("Both training and validation need positive trace labels.")
    loaders=[torch.utils.data.DataLoader(TraceDataset(s,args.tile_rows,a),batch_size=1,
              shuffle=a,num_workers=0) for s,a in ((training,True),(validation,False))]
    model=UNet(args.base).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1.; stale=0
    provenance=dict(dataset_kind=dataset["dataset_kind"],manifest_sha256=dataset["manifest_sha256"],
                    training_boreholes=sorted({s["borehole"] for s in training}),
                    validation_boreholes=sorted({s["borehole"] for s in validation}),
                    channels=sorted({s["channel"] for s in training}),
                    labelled_files=[dict(image=s["image"],mask=s["mask"],split=s["split"],
                         image_sha256=hashlib.sha256(Path(s["image"]).read_bytes()).hexdigest(),
                         mask_sha256=hashlib.sha256(Path(s["mask"]).read_bytes()).hexdigest())
                         for s in training+validation],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    torch_version=str(torch.__version__),seed=args.seed,
                    label_definition=dataset.get("label_definition","Unclassified planar traces"))
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0.
        for x,y,v in loaders[0]:
            x,y,v=x.to(args.device),y.to(args.device),v.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(model(x),y,v)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); loss_sum+=float(loss.detach())
        val=validate(model,loaders[1],args.device)
        row=dict(epoch=epoch,train_loss=loss_sum/len(loaders[0]),**{"val_"+k:v for k,v in val.items()})
        history.append(row); print(json.dumps(row),flush=True)
        if val["dice"]>best:
            best=val["dice"]; stale=0
            torch.save(dict(format_version=1,architecture="periodic_unet_v1",base=args.base,
                       state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                       trained_epochs=epoch,tile_rows=args.tile_rows,threshold=.5,
                       validation=val,provenance=provenance),out/"best.pt")
        else:
            stale+=1
        (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if stale>=args.patience:
            break
    (out/"training_manifest.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(f"Saved {out/'best.pt'}; dataset_kind={dataset['dataset_kind']}. Validation scores are pixel segmentation metrics.")

def load_checkpoint(path,device="cpu",allow_synthetic=False):
    data=torch.load(path,map_location="cpu",weights_only=True)
    if data.get("architecture")!="periodic_unet_v1" or data.get("trained_epochs",0)<1:
        raise ValueError("Expected a trained periodic_unet_v1 checkpoint.")
    if data["provenance"]["dataset_kind"]=="synthetic" and not allow_synthetic:
        raise ValueError("Synthetic-only checkpoint: explicitly allow it for demos; it is not RGL-trained.")
    model=UNet(data["base"])
    model.load_state_dict(data["state_dict"],strict=True)
    return model.to(device).eval(),data

def predict(model,rgb,valid,tile_rows=256,overlap=64,device="cpu"):
    """Overlapping depth tiles blend probabilities without resizing azimuth/depth."""
    h,w=valid.shape
    if rgb.shape!=(h,w,3) or tile_rows<16 or not 0<=overlap<tile_rows:
        raise ValueError("Invalid image, tile height, or overlap.")
    if not valid.any():
        return np.zeros((h,w),np.float32)
    starts=list(range(0,max(1,h-tile_rows+1),tile_rows-overlap))
    if h>tile_rows and starts[-1]!=h-tile_rows:
        starts.append(h-tile_rows)
    total=np.zeros((h,w),np.float32); weight=np.zeros((h,1),np.float32)
    model.eval()
    with torch.inference_mode():
        for lo in starts:
            hi=min(h,lo+tile_rows)
            if not valid[lo:hi].any():
                continue
            image=rgb[lo:hi]; v=valid[lo:hi]
            pad=tile_rows-len(image)
            if pad:
                image=np.pad(image,((0,pad),(0,0),(0,0)),mode="edge")
                v=np.pad(v,((0,pad),(0,0)))
            p=torch.sigmoid(model(input_tensor(image,v)[None].to(device)))[0,0].cpu().numpy()[:hi-lo]
            blend=np.maximum(np.hanning(tile_rows),.05).astype(np.float32)[:hi-lo,None]
            total[lo:hi]+=p*blend; weight[lo:hi]+=blend
    return np.where(valid,total/np.maximum(weight,1e-8),0).astype(np.float32)

def probability_to_evidence(probability,valid,threshold=.5,min_component_pixels=5):
    """Threshold then thin a periodic mask to one-pixel traces for Hough voting."""
    from skimage.morphology import skeletonize
    p=np.asarray(probability)
    if p.shape!=valid.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not 0<threshold<1:
        raise ValueError("Probability must be finite HxW in [0,1]; threshold must be in (0,1).")
    binary=(p>=threshold)&valid
    # Triple the azimuth so skeleton/component handling sees the circular seam.
    tiled=np.tile(binary,(1,3)); w=p.shape[1]
    labels,n=ndi.label(tiled,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel())
    keep=sizes>=min_component_pixels; keep[0]=False
    skeleton=skeletonize(keep[labels])[:,w:2*w]&valid
    return np.where(skeleton,p,0).astype(np.float32),binary

def segment_checkpoint(rgb,valid,path,channel,device="cpu",allow_synthetic=False,threads=2):
    torch.set_num_threads(threads)
    model,checkpoint=load_checkpoint(path,device,allow_synthetic)
    if channel not in checkpoint["provenance"]["channels"]:
        raise ValueError("Checkpoint was not trained for this image modality.")
    tile=checkpoint["tile_rows"]
    p=predict(model,rgb,valid,tile_rows=tile,overlap=min(64,tile//4),device=device)
    evidence,mask=probability_to_evidence(p,valid,checkpoint["threshold"])
    metadata=dict(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  threshold=checkpoint["threshold"],trained_epochs=checkpoint["trained_epochs"],
                  provenance=checkpoint["provenance"])
    return p,evidence,mask,metadata

def evaluate_checkpoint(args):
    dataset=load_manifest(args.manifest,require_training=False)
    model,ck=load_checkpoint(args.checkpoint,args.device,args.allow_synthetic)
    samples=[s for s in dataset["samples"] if s["split"]==args.split]
    if not samples:
        raise ValueError("Requested split contains no samples.")
    if args.split=="test":
        used=set(ck["provenance"]["training_boreholes"]+ck["provenance"]["validation_boreholes"])
        seen_hashes={x["image_sha256"] for x in ck["provenance"]["labelled_files"]}
        if any(s["borehole"] in used or hashlib.sha256(Path(s["image"]).read_bytes()).hexdigest() in seen_hashes for s in samples):
            raise ValueError("Test samples overlap the checkpoint's training/validation data.")
    counts=np.zeros(4,dtype=np.int64)
    for s in samples:
        if s["channel"] not in ck["provenance"]["channels"]:
            raise ValueError("Evaluation modality was not used for training.")
        rgb,y,v=read_pair(s)
        p=predict(model,rgb,v,ck["tile_rows"],min(64,ck["tile_rows"]//4),args.device)>=ck["threshold"]
        counts+=np.array([(p&y&v).sum(),(p&~y&v).sum(),(~p&y&v).sum(),(~p&~y&v).sum()])
    result=dict(split=args.split,dataset_kind=dataset["dataset_kind"],**metrics_from_counts(*map(int,counts)))
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

def synthetic_data(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    samples=[]; rng=np.random.default_rng(710)
    for split,count in (("train",16),("val",4),("test",4)):
        for i in range(count):
            h,w=96,96; x=np.arange(w); y=np.arange(h)[:,None]
            center=rng.uniform(30,66); amp=rng.uniform(4,16); phase=rng.uniform(0,2*np.pi)
            trace=center+amp*np.sin(2*np.pi*x/w+phase)
            target=np.abs(y-trace)<1.6
            noise=rng.normal(0,.02,(h,w))
            image=.65+noise-.42*np.exp(-.5*((y-trace)/1.3)**2)
            rgb=np.repeat((np.clip(image,0,1)*255).astype(np.uint8)[:,:,None],3,2)
            name=f"{split}_{i:02}"
            Image.fromarray(rgb).save(out/(name+".png"))
            Image.fromarray(target.astype(np.uint8)*255).save(out/(name+"_mask.png"))
            samples.append(dict(borehole="SYNTH_"+split,split=split,channel="optical",
                                image=name+".png",mask=name+"_mask.png",annotated_input=False))
    manifest=dict(dataset_kind="synthetic",label_definition="Synthetic narrow planar traces",samples=samples)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(out/"manifest.json")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train")
    t.add_argument("--manifest",required=True); t.add_argument("--output",required=True)
    t.add_argument("--epochs",type=int,default=30); t.add_argument("--base",type=int,default=16)
    t.add_argument("--tile-rows",type=int,default=256); t.add_argument("--lr",type=float,default=.001)
    t.add_argument("--patience",type=int,default=8); t.add_argument("--seed",type=int,default=42)
    t.add_argument("--threads",type=int,default=2); t.add_argument("--device",default="cpu")
    e=sub.add_parser("evaluate")
    e.add_argument("--manifest",required=True); e.add_argument("--checkpoint",required=True)
    e.add_argument("--split",choices=["val","test"],default="test"); e.add_argument("--output",required=True)
    e.add_argument("--device",default="cpu"); e.add_argument("--allow-synthetic",action="store_true")
    e.add_argument("--threads",type=int,default=2)
    s=sub.add_parser("synthetic-data"); s.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="train":
        train(args)
    elif args.command=="evaluate":
        torch.set_num_threads(args.threads); evaluate_checkpoint(args)
    else:
        synthetic_data(args.output)

if __name__=="__main__":
    main()

