"""Separate RGB-D localization evaluation; truth labels never enter inference."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .rgbd import RGBDFrame
from .rgbd_localization import RGBDLocalizer,LocalizationFailure,select_visible_five


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def audit(frame, spec, labels=None):
    scene=RGBDLocalizer(spec).observe(frame)
    try:selected=select_visible_five(scene)
    except LocalizationFailure:selected=()
    report=dict(localization=asdict(scene),selected_five=[asdict(c) for c in selected],
        object_truth_access_during_inference=False,segmentation_ids_used=False,
        assumptions=["horizontal table","pale bins","orange upright cookies",
                     "four touching source columns; source x uses fitted packing prior"],
        accepted_source_cookies=len(scene.source_cookies),accepted_target_cookies=len(scene.target_cookies),
        status="goal_available" if selected else "needs_another_view")
    if labels is not None:
        # Evaluator runs only AFTER the scene estimate and selected goal exist.
        source=np.asarray([c.position for c in scene.source_cookies])
        truth=np.asarray(labels["cookie_positions"])
        metrics={}
        if len(source):
            cost=np.linalg.norm(source[:,None]-truth[None],axis=2)
            row,col=linear_sum_assignment(cost)
            errors=source[row]-truth[col]
            metrics.update(source_matched_count=len(row),source_position_error_mm=(1000*np.linalg.norm(errors,axis=1)).tolist(),
                source_mean_error_mm=float(1000*np.linalg.norm(errors,axis=1).mean()),
                source_p95_error_mm=float(1000*np.quantile(np.linalg.norm(errors,axis=1),.95)),
                source_axis_mae_mm=(1000*abs(errors).mean(0)).tolist(),
                source_matches_within_3mm=int(np.sum(np.linalg.norm(errors,axis=1)<.003)))
        for kind in ("source","target"):
            estimate=getattr(scene,kind+"_box")
            metrics[kind+"_box_xy_error_mm"]=float(1000*np.linalg.norm(
                estimate.position[:2]-np.asarray(labels[kind+"_bin_position"])[:2]))
        rotation=np.asarray(labels["target_bin_rotation"])
        metrics["target_yaw_error_deg"]=float(np.rad2deg(np.arccos(np.clip(
            scene.target_box.rotation[:,0]@rotation[:,0],-1,1))))
        report["evaluation_only"]=metrics
    return report


def save_overlay(frame, report, path):
    from PIL import Image,ImageDraw
    image=Image.fromarray(frame.rgb);draw=ImageDraw.Draw(image)
    inverse=np.linalg.inv(frame.world_from_optical)
    def pixel(point):
        q=inverse[:3,:3]@np.asarray(point)+inverse[:3,3]
        if q[2]<=0:return None
        v=frame.intrinsic@q
        return tuple((v[:2]/v[2]-.5).tolist())
    for cookie in report["localization"]["source_cookies"]:
        uv=pixel(cookie["position"])
        if uv:draw.ellipse((uv[0]-3,uv[1]-3,uv[0]+3,uv[1]+3),outline="cyan",width=2)
    for cookie in report["selected_five"]:
        uv=pixel(cookie["position"])
        if uv:draw.ellipse((uv[0]-6,uv[1]-6,uv[0]+6,uv[1]+6),outline="lime",width=2)
    for name in ("source_box","target_box"):
        box=report["localization"][name]
        point=np.asarray(box["position"]).copy();point[2]=box["rim_z"]
        uv=pixel(point)
        if uv:draw.text(uv,name,fill="magenta")
    image.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--audit-truth",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((args.capture/"spec.json").read_text())
    with np.load(args.capture/"frame.npz") as data:frame=RGBDFrame(**dict(data))
    labels=json.loads((args.capture/"truth_audit_only.json").read_text()) if args.audit_truth else None
    result=audit(frame,spec,labels)
    (args.output/"result.json").write_text(json.dumps(result,default=json_value,indent=2))
    save_overlay(frame,result,args.output/"localization.png")
    print(json.dumps({k:v for k,v in result.items() if k not in ("localization","selected_five")},default=json_value))


if __name__=="__main__":main()
