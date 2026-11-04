"""Fuse the declared cameras: global bin fit + consistent visible instances.

The old localizer discards a wrist view unless BOTH entire bins can be fitted
in that view. Here a valid global view supplies the measured bin transforms;
the wrist contributes RGB-D surface samples in that calibrated frame.
No invisible cookie is created from the packing prior.
"""
import numpy as np
from a3_batch_skills.rgbd import RGBDViews, unproject
from a3_batch_skills.rgbd_localization import (
    RGBDLocalizer, LocalizationFailure, VisualScene,
)

LOCALIZER_VERSION = "three_native_rgbd_consistent_instances_v2"


class ThreeViewLocalizer(RGBDLocalizer):
    def _cookies(self,frame,points,box,*,source):
        from .slab_instances import slab_instances
        return slab_instances(frame,points,box,self.spec,source=source,
            min_coverage=self.min_coverage,source_columns=self.source_columns)

    def observe(self, frame):
        if not isinstance(frame, RGBDViews) or len(frame.frames) != 3:
            raise LocalizationFailure("Expected the three synchronized native cameras")
        # A geometrically supported image, rather than a simulator pose, anchors
        # the coordinates. Do not accept mutually inconsistent full-bin fits.
        scenes = []
        for image in frame.frames:
            try:
                scenes.append(super().observe(image))
            except LocalizationFailure:
                pass
        if not scenes:
            # Calibrated views may independently see different bins. Both boxes
            # still need current metric rim support; no cached layout is filled in.
            boxes={}
            for kind in ('source','target'):
                clone=dict(self.spec)
                if kind=='source':
                    clone.update(target_inner_half_size=self.spec['source_inner_half_size'],
                        wall_height_m=self.spec['source_wall_top_offset_m'])
                else:
                    clone.update(source_inner_half_size=self.spec['target_inner_half_size'],
                        source_wall_top_offset_m=self.spec['wall_height_m'])
                candidates=[]
                for image in frame.frames:
                    try:
                        a,b=RGBDLocalizer(clone)._boxes(image,unproject(image),.75)
                        candidates.append(a if kind=='source' else b)
                    except LocalizationFailure:pass
                if not candidates:raise LocalizationFailure('No calibrated view supports '+kind+' bin rim')
                boxes[kind]=min(candidates,key=lambda b:b.extent_error_m+.003*(4-b.supported_sides))
            base=VisualScene(boxes['source'],boxes['target'],(),(),frame.timestamp,.75)
        else:
            scenes.sort(key=lambda s: (s.source_box.extent_error_m+s.target_box.extent_error_m))
            base = scenes[0]
        for other in scenes[1:]:
            for name in ("source_box", "target_box"):
                a, b = getattr(base, name), getattr(other, name)
                angle = np.arccos(np.clip((np.trace(a.rotation.T@b.rotation)-1)/2, -1, 1))
                if np.linalg.norm(a.position-b.position) > .005 or angle > np.deg2rad(2):
                    raise LocalizationFailure("Conflicting visual bin estimates; recalibrate/reobserve")
        if len({f.rgb.shape[1] for f in frame.frames}) != 1:
            raise LocalizationFailure("Rectify RGB-D widths before surface pooling")
        def merge(name,source):
            box=base.source_box if source else base.target_box
            result=list(getattr(base,name))
            def same_instance(a,b):
                if a.column!=b.column:return False
                delta=(a.position-b.position)@box.rotation
                return (abs(delta[1])<.0045 if source else np.linalg.norm(delta[:2])<.0045)
            for image in frame.frames:
                # Keep complete visible instances separate. Millimetre camera
                # errors must not widen/merge thin slabs in a pooled histogram.
                proposals=self._cookies(image,unproject(image),box,source=source)
                matched=[]
                for cookie in proposals:
                    same=[c for c in result if same_instance(c,cookie)]
                    if same:matched.append(cookie.position-same[0].position)
                if matched and np.linalg.norm(np.median(matched,axis=0))>.003:
                    continue  # calibration inconsistency; do not accumulate it
                for cookie in proposals:
                    same=[c for c in result if same_instance(c,cookie)]
                    if not same:result.append(cookie)
            return tuple(sorted(result,key=lambda c:(c.column or 0,float((c.position-box.position)@box.rotation[:,1]))))
        return VisualScene(base.source_box, base.target_box,
            merge('source_cookies',True),merge('target_cookies',False),
            frame.timestamp, base.table_z, source=LOCALIZER_VERSION)
