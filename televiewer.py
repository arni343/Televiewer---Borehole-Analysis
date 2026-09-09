"""Televiewer sinusoid proof of principle. See README.md before interpreting results.
Model: row = center + sin_coefficient*sin(theta) + cos_coefficient*cos(theta).
Coordinates are native report pixels; depth increases downwards.
"""
from __future__ import annotations
import argparse
import os
import sys
import csv
import hashlib
import json
import math
from pathlib import Path
import time
RESULTS = Path(__file__).resolve().parent.parent / "RESULTS"
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / "_cache" / "matplotlib"))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks

DEFAULTS = dict(max_amplitude_px=100., amplitude_step_px=4., phase_step_deg=12.,
                min_support=.50, min_valid_fraction=.65, max_gap_fraction=.38,
                min_sectors=6, max_rmse_px=2., ridge_threshold=2.0,
                fit_radius_px=5, duplicate_distance_px=5., tile_rows=640,
                max_seed_candidates=400, min_direction_cosine=.87,
                min_orientation_coherence=.20, min_support_excess=.12,
                min_chain_fraction=.15)

def design(width, endpoint=False):
    theta = np.linspace(0, 2*np.pi, width, endpoint=endpoint)
    return np.column_stack((np.ones(width), np.sin(theta), np.cos(theta)))

def curve(p, width, endpoint=False):
    return design(width, endpoint) @ np.asarray(p)

def circular_gap(supported):
    """Longest unsupported run on the circular azimuth axis."""
    a = np.asarray(supported, bool)
    if a.all():
        return 0.
    if not a.any():
        return 1.
    z = np.concatenate(([False], ~a, ~a, [False])).astype(int)
    starts = np.flatnonzero(np.diff(z) == 1)
    ends = np.flatnonzero(np.diff(z) == -1)
    return min(len(a), int(np.max(ends-starts))) / len(a)

def annotation_mask(rgb):
    """Exact report ink colours only; does not reconstruct overwritten geology."""
    a = rgb.astype(np.int16)
    palette = np.array([[255,0,255], [0,255,0], [255,255,0],
                        [255,0,0], [255,128,0], [128,128,128]])
    return np.any(np.max(np.abs(a[:,:,None,:]-palette), axis=3) <= 8, axis=2)

def prepare(rgb, annotated=False, threshold=2.5):
    gray = (rgb.astype(np.float32) @ np.array([.2126,.7152,.0722], np.float32))/255.
    # White report margins are missing data, not low-contrast rock.
    present = np.mean(np.min(rgb, axis=2) < 245, axis=1) > .60
    present = ndi.binary_erosion(present, iterations=4, border_value=0)
    ink = annotation_mask(rgb) if annotated else np.zeros(gray.shape, bool)
    invalid = ~np.broadcast_to(present[:,None], gray.shape).copy()
    invalid |= ndi.binary_dilation(ink, iterations=2) if annotated else ink
    valid = ~invalid
    if not valid.any():
        return np.zeros_like(gray), valid, ink, gray
    # Fill only for filtering; all filled pixels remain excluded from evidence.
    indices = ndi.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    filled = gray[tuple(indices)]
    sm = ndi.gaussian_filter(filled, (.7,.7), mode=("nearest","wrap"))
    residual = filled-sm
    noise = ndi.gaussian_filter(np.abs(residual), (12,5), mode=("nearest","wrap"))*1.48
    noise = np.maximum(noise, .012)
    responses = []
    for sigma in (2., 5., 10.):
        bg = ndi.gaussian_filter1d(sm, sigma, axis=0, mode="nearest")
        responses.append((bg-sm)/noise)
    dark = np.maximum.reduce(responses)
    bright = np.maximum.reduce([-r for r in responses])
    # Keep vertical extrema, not a fixed percentage of all pixels.
    def thin(s, is_dark):
        extrema=ndi.minimum_filter1d(sm,3,axis=0) if is_dark else ndi.maximum_filter1d(sm,3,axis=0)
        image_extremum=sm<=extrema if is_dark else sm>=extrema
        local = ndi.maximum_filter1d(s, 3, axis=0, mode="nearest")
        return np.where(valid & image_extremum & (s >= threshold) & (s >= local), s, 0).astype(np.float32)
    return np.stack((thin(dark,True), thin(bright,False))), valid, ink, gray


def ridge_orientation(gray, valid):
    """Local ridge normals from a smoothed structure tensor (periodic azimuth)."""
    if not valid.any():
        return np.zeros((3,)+gray.shape,np.float32)
    idx=ndi.distance_transform_edt(~valid,return_distances=False,return_indices=True)
    filled=gray[tuple(idx)]
    gx=ndi.gaussian_filter(filled,1.,order=(0,1),mode=("nearest","wrap"))
    gy=ndi.gaussian_filter(filled,1.,order=(1,0),mode=("nearest","wrap"))
    xx=ndi.gaussian_filter(gx*gx,1.4,mode=("nearest","wrap"))
    xy=ndi.gaussian_filter(gx*gy,1.4,mode=("nearest","wrap"))
    yy=ndi.gaussian_filter(gy*gy,1.4,mode=("nearest","wrap"))
    angle=.5*np.arctan2(2*xy,xx-yy)
    coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-10)
    return np.stack((np.cos(angle),np.sin(angle),coherence))

def direction_support(p, mat, orientation, ys, xs, settings, endpoint=False):
    if orientation is None:
        return np.ones(np.broadcast_shapes(np.shape(ys),np.shape(xs)),bool)
    period=len(mat)-1 if endpoint else len(mat)
    slope=2*np.pi/period*(p[1]*mat[xs,2]-p[2]*mat[xs,1])
    nx,ny,coherence=orientation[:,ys,xs]
    cosine=np.abs(ny-nx*slope)/np.sqrt(1+slope*slope)
    return (cosine>=settings["min_direction_cosine"]) & (coherence>=settings["min_orientation_coherence"])


def seeds(evidence, settings, endpoint=False, orientation=None):
    h,w = evidence.shape
    xs = np.unique(np.linspace(0,w-1,min(w,64)).astype(int))
    yy, xi = np.nonzero(evidence[:,xs] > 0)
    if len(yy) < 8:
        return []
    mat = design(w, endpoint)
    candidates = []
    amplitude_values = np.arange(0, settings["max_amplitude_px"]+.001,
                                 settings["amplitude_step_px"])
    for amp in amplitude_values:
        phases = [0] if amp == 0 else np.arange(0,360,settings["phase_step_deg"])
        for phase in phases:
            p = np.array([0.,amp*np.cos(np.deg2rad(phase)),amp*np.sin(np.deg2rad(phase))])
            offsets = np.rint(mat[xs]@p).astype(int)
            centers = yy-offsets[xi]
            good = (centers>=0)&(centers<h)
            good &= direction_support(p,mat,orientation,yy,xs[xi],settings,endpoint)
            votes = np.bincount(centers[good], minlength=h).astype(float)
            # +/-3 px coarse tolerance avoids losing steep curves between phase bins.
            # Votes only generate hypotheses; full-resolution checks decide acceptance.
            votes = ndi.convolve1d(votes, np.ones(7), mode="constant")/len(xs)
            peaks,_ = find_peaks(votes, height=settings["min_support"]*.65, distance=5)
            for y in peaks:
                candidates.append((float(votes[y]), np.array([float(y),p[1],p[2]])))
    candidates.sort(key=lambda a: a[0], reverse=True)
    selected = []
    probe = mat[np.unique(np.linspace(0,w-1,16).astype(int))]
    for score,p in candidates:
        if any(np.mean(np.abs(probe@(p-q))) < 5 for q in selected):
            continue
        selected.append(p)
        if len(selected) >= settings["max_seed_candidates"]:
            break
    return selected

def refine(p, evidence, valid, settings, endpoint=False, orientation=None):
    h,w = evidence.shape
    mat = design(w,endpoint)
    xs = np.arange(w)
    offsets = np.arange(-settings["fit_radius_px"], settings["fit_radius_px"]+1)
    p = np.asarray(p,float).copy()
    chosen = supported = None
    for _ in range(4):
        pred = mat@p
        ys = np.rint(pred).astype(int)[None,:]+offsets[:,None]
        inside = (ys>=0)&(ys<h)
        yc = np.clip(ys,0,h-1)
        strengths = evidence[yc,xs]
        usable = inside & valid[yc,xs] & (strengths>0)
        usable &= direction_support(p,mat,orientation,yc,xs,settings,endpoint)
        # Prefer nearby ridges; strength only breaks near-distance ties.
        merit = np.where(usable, -np.abs(ys-pred)+.15*np.minimum(strengths,8), -1e9)
        best = np.argmax(merit, axis=0)
        chosen = ys[best,xs]
        supported = usable[best,xs]
        if supported.sum() < max(10,w*.25):
            return None
        xfit,yfit = mat[supported],chosen[supported]
        if np.linalg.cond(xfit)>20:
            return None
        fit = least_squares(lambda q:xfit@q-yfit,p,loss="soft_l1",f_scale=1.2)
        p = fit.x
    pred = mat@p
    residual = np.abs(chosen-pred)
    supported &= residual<=2.5
    iy = np.rint(pred).astype(int)
    in_bounds = (iy>=0)&(iy<h)
    vc = in_bounds & valid[np.clip(iy,0,h-1),xs]
    supported &= vc
    supported &= direction_support(p,mat,orientation,np.clip(chosen,0,h-1),xs,settings,endpoint)
    # A candidate must stand out from nearby, similarly oriented traces.
    baseline=[]
    for shift in (-14,-8,8,14):
        b=p.copy(); b[0]+=shift
        by=np.rint(mat@b).astype(int)
        hit=np.zeros(w,bool)
        for delta in (-2,-1,0,1,2):
            rows=by+delta
            within=(rows>=0)&(rows<h)
            rows=np.clip(rows,0,h-1)
            hit |= within & valid[rows,xs] & (evidence[rows,xs]>0) & direction_support(b,mat,orientation,rows,xs,settings,endpoint)
        baseline.append(hit.mean())
    excess=float(supported.mean()-np.mean(baseline))
    # Bridge single missing columns, then require a substantial connected arc.
    bridged=supported | (np.roll(supported,1)&np.roll(supported,-1))
    chain=circular_gap(~bridged)
    fraction = float(supported.mean())
    nsectors = sum(np.mean(b) >= .25 for b in np.array_split(supported,8))
    gap = circular_gap(supported)
    if supported.sum()<3:
        return None
    rmse = float(np.sqrt(np.mean(residual[supported]**2)))
    amp = float(np.hypot(p[1],p[2]))
    if (fraction<settings["min_support"] or vc.mean()<settings["min_valid_fraction"]
        or nsectors<settings["min_sectors"] or gap>settings["max_gap_fraction"]
        or rmse>settings["max_rmse_px"] or amp>settings["max_amplitude_px"]
        or not 0<=p[0]<h or excess<settings["min_support_excess"]
        or chain<settings["min_chain_fraction"]):
        return None
    # A ranking statistic, never a probability of geological correctness.
    quality = fraction*(1-gap)*math.exp(-rmse/3)*min(1.,max(0.,excess)*2)
    return dict(center_row_px=float(p[0]),sin_coefficient_px=float(p[1]),
                cos_coefficient_px=float(p[2]),amplitude_px=amp,
                phase_deg=float(np.degrees(np.arctan2(p[2],p[1]))%360) if amp>.5 else None,
                support_fraction=fraction,valid_fraction=float(vc.mean()),
                sectors=int(nsectors),max_gap_fraction=gap,rmse_px=rmse,
                quality_score=float(quality), support_excess=excess, longest_chain_fraction=chain,
                review_flag="near_horizontal_check_acquisition_artifact" if amp<2 else "",
                feature_type="unclassified_planar_candidate",
                _support_rows=np.where(supported,chosen.astype(float),np.nan))

def params(d):
    return np.array([d["center_row_px"],d["sin_coefficient_px"],d["cos_coefficient_px"]])

def deduplicate(results, width, distance=5, endpoint=False):
    accepted=[]
    mat=design(width,endpoint)
    for d in sorted(results,key=lambda x:x["quality_score"],reverse=True):
        if any(np.mean(np.abs(mat@(params(d)-params(q))))<distance for q in accepted):
            continue
        if "_support_rows" in d:
            rows=d["_support_rows"]
            used=np.zeros(width,bool)
            for previous in accepted:
                other=previous.get("_support_rows")
                if other is not None:
                    used |= np.abs(rows-other)<=2.5
            if used.sum()>np.isfinite(rows).sum()*.35:
                continue
        accepted.append(d)
    return sorted(accepted,key=lambda d:d["center_row_px"])

def detect(evidence, valid, settings=None, endpoint=False, orientation=None):
    cfg=DEFAULTS | (settings or {})
    h,w=valid.shape
    if evidence.ndim==2:
        evidence=evidence[None,:,:]
    margin=int(math.ceil(cfg["max_amplitude_px"]))+cfg["fit_radius_px"]+12
    results=[]
    for core0 in range(0,h,cfg["tile_rows"]):
        core1=min(h,core0+cfg["tile_rows"])
        lo=max(0,core0-margin)
        hi=min(h,core1+margin)
        if not valid[lo:hi].any():
            continue
        for polarity,channel in enumerate(evidence):
            local_orientation=None if orientation is None else orientation[:,lo:hi]
            for p in seeds(channel[lo:hi],cfg,endpoint,local_orientation):
                d=refine(p,channel[lo:hi],valid[lo:hi],cfg,endpoint,local_orientation)
                if d is None:
                    continue
                d["center_row_px"]+=lo
                d["_support_rows"]+=lo
                if core0<=d["center_row_px"]<core1:
                    d["evidence_kind"]="dark_ridge" if polarity==0 else "bright_ridge"
                    results.append(d)
    return deduplicate(results,w,cfg["duplicate_distance_px"],endpoint)

def calibration(spec):
    anchors=spec.get("depth_anchors")
    if anchors is None:
        return None
    (y0,z0),(y1,z1)=np.asarray(anchors,float)
    if not np.isfinite([y0,z0,y1,z1]).all() or y1<=y0 or z1<=z0:
        raise ValueError("Depth anchors must be finite and increase in both row and depth.")
    scale=(z1-z0)/(y1-y0)
    return float(scale),float(z0-y0*scale)

def geometry(d,spec):
    out=dict(depth_m=None,amplitude_m=None,borehole_relative_dip_deg=None,
             true_dip_deg=None,true_dip_azimuth_deg=None)
    cal=calibration(spec)
    if cal is None:
        return out
    scale,intercept=cal
    out["depth_m"]=d["center_row_px"]*scale+intercept
    out["amplitude_m"]=d["amplitude_px"]*scale
    diameter=spec.get("diameter_m")
    if diameter is None:
        return out
    if not np.isfinite(diameter) or diameter<=0:
        raise ValueError("diameter_m must be positive, in metres.")
    radius=diameter/2
    out["borehole_relative_dip_deg"]=float(np.degrees(np.arctan(out["amplitude_m"]/radius)))
    # Columns of rotation are image theta=0 radial, theta=90 radial,
    # and downhole-axis unit vectors, all in North/East/Down coordinates.
    rotation=spec.get("image_frame_to_ned")
    if rotation is not None:
        r=np.asarray(rotation,float)
        if (r.shape!=(3,3) or not np.isfinite(r).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(r),1,atol=1e-5)):
            raise ValueError("image_frame_to_ned must be a proper orthonormal 3x3 rotation.")
        n=r@np.array([-d["cos_coefficient_px"]*scale/radius,
                      -d["sin_coefficient_px"]*scale/radius,1.])
        if n[2]<0:
            n=-n
        out["true_dip_deg"]=float(np.degrees(np.arctan2(np.hypot(n[0],n[1]),n[2])))
        if np.hypot(n[0],n[1])>1e-8:
            out["true_dip_azimuth_deg"]=float(np.degrees(np.arctan2(-n[1],-n[0]))%360)
    return out

def write_csv(path,rows,fields=None):
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def evaluate(predictions,truth,reviewed,width,endpoint=False,tolerance=5.):
    """One-to-one trace matching, restricted to fully reviewed center-row intervals.
    Truth must include every target feature centered within reviewed intervals.
    This does not infer truth from report annotations.
    """
    intervals=np.asarray(reviewed,float)
    if intervals.ndim!=2 or intervals.shape[1]!=2 or np.any(intervals[:,1]<=intervals[:,0]):
        raise ValueError("Provide nonempty [start_row, end_row) reviewed intervals.")
    def inside(d):
        return any(lo<=float(d["center_row_px"])<hi for lo,hi in intervals)
    pred=[d for d in predictions if inside(d)]
    gt=[d for d in truth if inside(d)]
    pcur=np.array([curve(params(d),width,endpoint) for d in pred])
    gcur=np.array([curve(params(d),width,endpoint) for d in gt])
    pairs=[]
    if len(pred) and len(gt):
        cost=np.mean(np.abs(pcur[:,None,:]-gcur[None,:,:]),axis=2)
        # Dummy assignments allow unmatched objects. Valid matches dominate total cost,
        # so assignment maximizes match count before minimizing trace error.
        n,m=cost.shape
        penalty=(n+m+1)*(tolerance+1)
        augmented=np.full((n+m,n+m),penalty)
        augmented[n:,m:]=0
        augmented[:n,:m]=np.where(cost<=tolerance,cost,penalty*3)
        ri,ci=linear_sum_assignment(augmented)
        pairs=[(i,j,float(cost[i,j])) for i,j in zip(ri,ci) if i<n and j<m and cost[i,j]<=tolerance]
    tp=len(pairs); fp=len(pred)-tp; fn=len(gt)-tp
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    result=dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                trace_mae_px=float(np.mean([x[2] for x in pairs])) if pairs else None)
    for key in ("center_row_px","amplitude_px","depth_m","borehole_relative_dip_deg",
                "true_dip_deg","true_dip_azimuth_deg","phase_deg"):
        errors=[]
        for i,j,_ in pairs:
            a,b=pred[i].get(key),gt[j].get(key)
            if a is None or b is None or a=="" or b=="":
                continue
            error=abs(float(a)-float(b))
            if key in ("phase_deg","true_dip_azimuth_deg"):
                error=abs((float(a)-float(b)+180)%360-180)
            errors.append(error)
        result[key+"_mae"]=float(np.mean(errors)) if errors else None
        result[key+"_matched_count"]=len(errors)
    return result

def plot_pages(rgb,evidence,valid,picks,outdir,spec,track,page_rows=850,max_pages=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h,w=valid.shape
    present=np.flatnonzero(valid.any(axis=1))
    if not len(present):
        return []
    paths=[]
    evidence2=evidence.max(axis=0) if evidence.ndim==3 else evidence
    for page,lo in enumerate(range(int(present[0]),int(present[-1])+1,page_rows)):
        if max_pages and page>=max_pages:
            break
        hi=min(lo+page_rows,h)
        fig,axes=plt.subplots(1,3,figsize=(10,12),sharey=True)
        for ax in axes:
            ax.set_xlim(0,w-1)
            ax.set_ylim(hi,lo)
            ax.set_xlabel("Azimuth column (pixels)")
        axes[0].imshow(rgb[lo:hi],extent=(-.5,w-.5,hi-.5,lo-.5),aspect="auto")
        axes[1].imshow(np.where(valid[lo:hi],evidence2[lo:hi],np.nan),
                       extent=(-.5,w-.5,hi-.5,lo-.5),aspect="auto",cmap="magma",vmin=0,vmax=8)
        axes[2].imshow(rgb[lo:hi],extent=(-.5,w-.5,hi-.5,lo-.5),aspect="auto")
        axes[0].set_title("Input report track")
        axes[1].set_title("Ridge evidence; missing pixels excluded")
        axes[2].set_title("Unclassified candidates")
        axes[0].set_ylabel("Original report row (pixels)")
        cal=calibration(spec)
        if cal:
            scale,intercept=cal
            sec=axes[0].secondary_yaxis("left",functions=(lambda y:y*scale+intercept,
                                                        lambda z:(z-intercept)/scale))
            sec.spines["left"].set_position(("outward",52))
            sec.set_ylabel("Depth (m), provisional report calibration")
        for d in picks:
            ys=curve(params(d),w,track.get("endpoint",False))
            if ys.max()<lo or ys.min()>hi:
                continue
            axes[2].plot(np.arange(w),ys,color="#00e5ff",lw=.8)
            if lo<=d["center_row_px"]<hi:
                axes[2].text(w+2,d["center_row_px"],str(d["candidate_id"]),fontsize=6)
        fig.suptitle(f'{spec["borehole"]} / {track["name"]} / page {page+1}\n'
                     'Exploratory output: expert review required; score is not a probability')
        fig.tight_layout()
        dest=outdir/f'page_{page+1:03}.png'
        fig.savefig(dest,dpi=120)
        plt.close(fig)
        paths.append(dest)
    return paths

def run(args):
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_hashes={}
    segmentation_runs={}
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    base=config_path.parent
    # A command-line checkpoint enables the existing learned segmentation route.
    # It changes the evidence image, not the original Hough and curve-fit rules.
    checkpoint=getattr(args,"unet_checkpoint",None)
    if checkpoint:
        checkpoint=str(Path(checkpoint).resolve())
        from unet_segmentation import load_checkpoint
        _,model_info=load_checkpoint(checkpoint,getattr(args,"unet_device","cpu"),False)
        selected=0
        for spec in config["logs"]:
            if args.boreholes and spec["borehole"] not in args.boreholes: continue
            for track in spec["tracks"]:
                if args.channels and track["name"] not in args.channels: continue
                if track.get("annotated",False):
                    raise ValueError("Selected track contains expert ink. Select clean tracks with --boreholes/--channels.")
                if track["name"] not in model_info["provenance"]["channels"]:
                    raise ValueError("Checkpoint modality mismatch. Select its trained modality with --channels.")
                if track.get("evidence_npy"):
                    raise ValueError("Remove evidence_npy before enabling a U-Net checkpoint.")
                track["unet_checkpoint"]=checkpoint
                track["unet_device"]=getattr(args,"unet_device","cpu")
                track["allow_synthetic_checkpoint"]=False
                selected+=1
        if not selected: raise ValueError("No tracks selected for U-Net segmentation.")
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg=DEFAULTS|config.get("detector",{})
    if (cfg["max_amplitude_px"]<=0 or cfg["amplitude_step_px"]<=0
        or not 0<cfg["phase_step_deg"]<=360 or cfg["tile_rows"]<32):
        raise ValueError("Invalid amplitude, phase, or tile settings.")
    (out/"config_used.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    summaries=[]; allpicks=[]; links=[]
    for spec in config["logs"]:
        if args.boreholes and spec["borehole"] not in args.boreholes:
            continue
        image_path=(base/spec["image"]).resolve()
        input_hashes[spec["borehole"]]=dict(path=str(image_path),sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
        rgb=np.asarray(Image.open(image_path).convert("RGB"))
        if list(rgb.shape[1::-1])!=spec["expected_size"]:
            raise ValueError(f"{image_path.name}: size differs from calibrated report layout.")
        for track in spec["tracks"]:
            if args.channels and track["name"] not in args.channels:
                continue
            start=time.perf_counter()
            name=spec["borehole"]+"_"+track["name"]
            folder=out/name
            folder.mkdir(exist_ok=True)
            x0,x1=track["x_bounds"]
            if not 0<=x0<x1<=rgb.shape[1]:
                raise ValueError("Track bounds outside image.")
            strip=rgb[:,x0:x1].copy()
            # Preserve native rows; no hidden crop offset in exported parameters.
            strip[:spec["header_end_row"]]=255
            strip[spec.get("data_end_row",len(strip)):]=255
            ev,valid,ink,gray=prepare(strip,track.get("annotated",False),cfg["ridge_threshold"])
            if ev.ndim==2:
                ev=ev[None,:,:]
            if track["name"]!="optical":
                ev=ev[:1]
            if track.get("evidence_npy"):
                supplied=np.load(base/track["evidence_npy"],allow_pickle=False)
                if supplied.shape!=valid.shape or not np.isfinite(supplied).all() or supplied.min()<0:
                    raise ValueError("External evidence must be a finite nonnegative HxW array.")
                ev=np.where(valid,supplied,0)[None,:,:]
            segmentation_metadata=None
            if track.get("unet_checkpoint"):
                if track.get("evidence_npy"):
                    raise ValueError("Choose unet_checkpoint or evidence_npy, not both.")
                if track.get("annotated",False):
                    raise ValueError("U-Net requires a clean image track; annotated inputs leak expert picks.")
                from unet_segmentation import segment_checkpoint
                print(f"{name}: U-Net segmentation...",flush=True)
                probability,learned_evidence,segmentation_mask,segmentation_metadata=segment_checkpoint(
                    strip,valid,base/track["unet_checkpoint"],track["name"],
                    track.get("unet_device","cpu"),track.get("allow_synthetic_checkpoint",False))
                ev=learned_evidence[None,:,:]
                np.save(folder/"segmentation_probability.npy",probability)
                Image.fromarray(segmentation_mask.astype(np.uint8)*255).save(folder/"segmentation_mask.png")
                (folder/"segmentation_model.json").write_text(json.dumps(segmentation_metadata,indent=2),encoding="utf-8")
                segmentation_runs[name]=segmentation_metadata
                normals=ridge_orientation(probability,valid)
            else:
                normals=ridge_orientation(gray,valid)
            print(f"{name}: {valid.sum()} usable pixels; sinusoidal Hough fitting...",flush=True)
            picks=detect(ev,valid,cfg,track.get("endpoint",False),normals)
            for i,d in enumerate(picks,1):
                d.pop("_support_rows",None)
                if segmentation_metadata is not None:
                    d["evidence_kind"]="unet_segmentation"
                d.update(candidate_id=i,borehole=spec["borehole"],channel=track["name"],
                         annotated_input=track.get("annotated",False),
                         calibration_status="provisional_report_ticks",
                         geographic_orientation_status="provided" if spec.get("image_frame_to_ned") else "unknown")
                d.update(geometry(d,spec))
            fields=list(picks[0]) if picks else ["candidate_id","borehole","channel","center_row_px",
                     "sin_coefficient_px","cos_coefficient_px","amplitude_px","phase_deg","quality_score",
                     "depth_m","borehole_relative_dip_deg","true_dip_deg","true_dip_azimuth_deg"]
            write_csv(folder/"candidates.csv",picks,fields)
            np.savez_compressed(folder/"evidence.npz",evidence=ev,valid=valid,annotation_mask=ink)
            pages=plot_pages(strip,ev,valid,picks,folder,spec,track,max_pages=args.max_pages)
            links.extend((name,p.relative_to(out).as_posix()) for p in pages)
            summary=dict(borehole=spec["borehole"],channel=track["name"],candidate_count=len(picks),
                         valid_pixels=int(valid.sum()),masked_annotation_pixels=int(ink.sum()),
                         annotated_input=track.get("annotated",False),
                         physical_dip_available=spec.get("diameter_m") is not None,
                         seconds=round(time.perf_counter()-start,2))
            if track.get("truth_csv"):
                if track.get("annotated",False):
                    raise ValueError("Independent validation is disabled on annotated input. Use clean exports.")
                with (base/track["truth_csv"]).open(encoding="utf-8-sig",newline="") as f:
                    truth=list(csv.DictReader(f))
                for d in truth:
                    for k in ("center_row_px","sin_coefficient_px","cos_coefficient_px"):
                        d[k]=float(d[k])
                    d["amplitude_px"]=float(np.hypot(d["sin_coefficient_px"],d["cos_coefficient_px"]))
                    d["phase_deg"]=float(np.degrees(np.arctan2(d["cos_coefficient_px"],d["sin_coefficient_px"]))%360) if d["amplitude_px"]>.5 else None
                    derived=geometry(d,spec)
                    for key,value in derived.items():
                        if d.get(key) in (None,""):
                            d[key]=value
                metrics=evaluate(picks,truth,track["reviewed_intervals"],x1-x0,
                                 track.get("endpoint",False),track.get("match_tolerance_px",5.))
                (folder/"validation.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
            summaries.append(summary); allpicks.extend(picks)
            print(f"{name}: {len(picks)} candidates; {summary['seconds']} s",flush=True)
    write_csv(out/"summary.csv",summaries, list(summaries[0]) if summaries else ["borehole","channel","candidate_count"])
    write_csv(out/"all_candidates.csv",allpicks,list(allpicks[0]) if allpicks else ["candidate_id","borehole","channel"])
    import html
    body="".join(f'<li>{html.escape(name)}: <a href="{html.escape(path)}">{html.escape(Path(path).name)}</a></li>'
                 for name,path in links)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Televiewer review</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;line-height:1.6}</style>'
        '<h1>Televiewer candidate review</h1><p>These are unclassified planar candidates, not confirmed fractures. '
        'Quality scores rank curve support; they are not calibrated confidence probabilities. '
        'Annotated acoustic reports are exploratory inputs only. Physical dip requires diameter; '
        'geographic dip requires verified orientation.</p><p><a href="summary.csv">Summary CSV</a> | '
        '<a href="all_candidates.csv">All candidates</a></p><ul>'+body+'</ul>',encoding="utf-8")
    manifest=dict(created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                  numpy=np.__version__,scipy=__import__("scipy").__version__,
                  script_sha256=script_hash,inputs=input_hashes,segmentation_models=segmentation_runs,
                  validation_status="No accuracy claim without independent expert labels.")
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("rgl_config.json")))
    parser.add_argument("--output",default=str(RESULTS/("televiewer_"+time.strftime("%Y%m%d_%H%M%S"))))
    parser.add_argument("--unet-checkpoint",help="RGL-trained checkpoint for the selected clean tracks; preserves the original fitter.")
    parser.add_argument("--unet-device",default="cpu",help="cpu or cuda")
    parser.add_argument("--boreholes",nargs="+")
    parser.add_argument("--channels",nargs="+")
    parser.add_argument("--max-pages",type=int,default=0,help="0 saves every page; detection always processes the entire track.")
    run(parser.parse_args())

