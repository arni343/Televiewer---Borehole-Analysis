"""Deterministic synthetic benchmark, separate from field accuracy."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from televiewer import prepare,ridge_orientation,detect,curve,params,evaluate,write_csv

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start",type=int,default=1000)
    parser.add_argument("--output",default=str(Path(__file__).with_name("synthetic_benchmark")))
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(exist_ok=True)
    report=[]
    for seed in range(20):
        rng=np.random.default_rng(args.seed_start+seed)
        h,w=600,160
        kind=("clean","noisy","partial","crossing","bright")[seed%5]
        truth_params=[[180.,rng.uniform(-45,45),rng.uniform(-30,30)],
                      [420.,rng.uniform(-45,45),rng.uniform(-30,30)]]
        if kind=="crossing":
            truth_params=[[290.,60.,20.],[290.,-55.,-15.]]
        y=np.arange(h)[:,None]
        background=.52 + .06*np.sin(np.arange(w)[None,:]*2*np.pi/w)
        signal=np.broadcast_to(background,(h,w)).copy()
        signal+=rng.normal(0,.025 if kind=="noisy" else .010,(h,w))
        if kind=="noisy":
            signal+=.025*ndi.gaussian_filter(rng.normal(size=(h,w)),(2,3))
        for p in truth_params:
            ridge=.30*np.exp(-.5*((y-curve(p,w))/1.3)**2)
            if kind=="partial":
                ridge[:,30:65]=0
            signal+=ridge if kind=="bright" else -ridge
        rgb=np.repeat((np.clip(signal,0,1)*255).astype(np.uint8)[:,:,None],3,axis=2)
        ev,valid,_,gray=prepare(rgb,threshold=2.)
        orientation=ridge_orientation(gray,valid)
        pred=detect(ev,valid,orientation=orientation)
        truth=[]
        for i,p in enumerate(truth_params,1):
            truth.append(dict(feature_id=i,center_row_px=p[0],sin_coefficient_px=p[1],
                              cos_coefficient_px=p[2],amplitude_px=float(np.hypot(p[1],p[2])),
                              phase_deg=float(np.degrees(np.arctan2(p[2],p[1]))%360)))
        metrics=evaluate(pred,truth,[[0,h]],w,tolerance=4)
        report.append(dict(seed=args.seed_start+seed,scene=kind,**metrics))
        Image.fromarray(rgb).save(out/f"scene_{seed:02}.png")
        write_csv(out/f"truth_{seed:02}.csv",truth)
        write_csv(out/f"predictions_{seed:02}.csv",
                  [{k:v for k,v in d.items() if not k.startswith("_")} for d in pred],
                  [k for k in pred[0] if not k.startswith("_")] if pred else ["center_row_px"])
    write_csv(out/"metrics.csv",report)
    tp=sum(r["tp"] for r in report); fp=sum(r["fp"] for r in report); fn=sum(r["fn"] for r in report)
    summary=dict(seed_start=args.seed_start,scenes=len(report),true_features=40,tp=tp,fp=fp,fn=fn,
                 precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn),
                 f1=2*tp/(2*tp+fp+fn),
                 interpretation="Synthetic engineering checks only; not RGL field accuracy.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

