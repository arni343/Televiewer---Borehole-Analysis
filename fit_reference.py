"""Fit expert-selected trace points into the ground-truth CSV schema.
Input CSV: feature_id,x_px,row_px,feature_type
x_px is local to the configured track; row_px is the ORIGINAL report row.
Example: python fit_reference.py --points expert_points.csv --width 151 --output truth.csv
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from televiewer import write_csv

def fit_points(x,y,width,endpoint=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if width<4 or len(x)<6 or len(x)!=len(y) or not np.isfinite(np.r_[x,y]).all():
        raise ValueError("Supply at least six finite trace points, distributed around the borehole.")
    if np.any((x<0)|(x>width-1)):
        raise ValueError("Point columns must be inside the local image track.")
    theta=x*2*np.pi/(width-1 if endpoint else width)
    mat=np.column_stack((np.ones(len(x)),np.sin(theta),np.cos(theta)))
    if np.linalg.cond(mat)>10 or len(np.unique(x))<6:
        raise ValueError("Trace points cover too little azimuth for a stable sinusoid.")
    initial=np.linalg.lstsq(mat,y,rcond=None)[0]
    fit=least_squares(lambda p:mat@p-y,initial,loss="soft_l1",f_scale=1.5)
    p=fit.x
    return dict(center_row_px=float(p[0]),sin_coefficient_px=float(p[1]),
                cos_coefficient_px=float(p[2]),amplitude_px=float(np.hypot(p[1],p[2])),
                phase_deg=float(np.degrees(np.arctan2(p[2],p[1]))%360) if np.hypot(p[1],p[2])>.5 else None,
                reference_fit_rmse_px=float(np.sqrt(np.mean((mat@p-y)**2))),point_count=len(x))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points",required=True)
    parser.add_argument("--width",type=int,required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--endpoint",action="store_true")
    args=parser.parse_args()
    groups={}
    with Path(args.points).open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            groups.setdefault(r["feature_id"],[]).append(r)
    records=[]
    for key,rows in groups.items():
        fit=fit_points([float(r["x_px"]) for r in rows],[float(r["row_px"]) for r in rows],
                       args.width,args.endpoint)
        records.append(dict(feature_id=key,feature_type=rows[0].get("feature_type","unclassified"),**fit))
    if not records:
        raise ValueError("No expert points were provided.")
    write_csv(args.output,records)
    print(f"Wrote {len(records)} reference traces. Inspect fitting residuals before evaluation.")

if __name__=="__main__":
    main()

