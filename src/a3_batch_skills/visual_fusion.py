"""Global measured bin anchors plus local wrist surface evidence.

No view must contain both complete bins. Rims must agree before pooling.
"""
import numpy as np
from .rgbd import RGBDViews,unproject
from .rgbd_localization import RGBDLocalizer,LocalizationFailure,VisualScene


class FusedRGBDLocalizer(RGBDLocalizer):
    def __init__(self,spec,*,allow_fallen=False,**kwargs):
        super().__init__(spec,**kwargs);self.allow_fallen=allow_fallen
        self.target_unknown=False;self.diagnostics={}

    def observe(self,packet):
        frames=packet.frames if isinstance(packet,RGBDViews) else (packet,)
        points=[unproject(f) for f in frames];boxes={};counts={}
        # Both anchors may come from separate synchronized views.
        for kind in ('source','target'):
            clone=dict(self.spec)
            if kind=='source':clone.update(target_inner_half_size=self.spec['source_inner_half_size'],
                wall_height_m=self.spec['source_wall_top_offset_m'])
            else:clone.update(source_inner_half_size=self.spec['target_inner_half_size'],
                source_wall_top_offset_m=self.spec['wall_height_m'])
            proposals=[]
            for f,q in zip(frames,points):
                try:
                    a,b=RGBDLocalizer(clone)._boxes(f,q,self._table_z(q))
                    proposals.append(a if kind=='source' else b)
                except LocalizationFailure:continue
            if not proposals:raise LocalizationFailure('No calibrated view supports '+kind+' bin rim')
            proposals.sort(key=lambda b:b.extent_error_m+.003*(4-b.supported_sides))
            anchor=proposals[0]
            for other in proposals[1:]:
                angle=np.arccos(np.clip((np.trace(anchor.rotation.T@other.rotation)-1)/2,-1,1))
                if np.linalg.norm(anchor.position-other.position)>.005 or angle>np.deg2rad(2):
                    raise LocalizationFailure('Conflicting visual bin estimates; recalibrate/reobserve')
            boxes[kind]=anchor;counts[kind]=len(proposals)
        objects={}
        self.target_unknown=False
        for kind in ('source','target'):
            source=kind=='source';box=boxes[kind];result=[]
            for f,q in zip(frames,points):
                for cookie in self._cookies(f,q,box,source=source):
                    if not any(c.column==cookie.column and np.linalg.norm(c.position-cookie.position)<.0035 for c in result):
                        result.append(cookie)
            if not source and self.allow_fallen:
                from .visual_slabs import oriented_targets
                result,unknown=oriented_targets(frames,points,box,self.spec,result)
                self.target_unknown=unknown
                from .visual_goals import floor_visibility,rectangle_samples
                from .visual_acceptance import cookie_corners
                xy=rectangle_samples(-box.inner_half_size+.002,box.inner_half_size-.002,n=21)
                visible=floor_visibility(packet,box,xy,self.spec['floor_offset_m'],return_mask=True)
                # Rasterize only measured footprints. Missing depth is never
                # made visible by a packing/count prior.
                from scipy.spatial import ConvexHull
                for c in result:
                    corners=(cookie_corners(c,self.half)-box.position)@box.rotation
                    try:
                        hull=ConvexHull(corners[:,:2]);eq=hull.equations
                        visible|=np.all(xy@eq[:,:2].T+eq[:,2]<=.001,axis=1)
                    except Exception:continue
                if visible.mean()<.90:self.target_unknown=True
            objects[kind]=tuple(result)
        self.diagnostics=dict(anchor_views=counts,target_unknown=self.target_unknown,
            source_instances=len(objects['source']),target_instances=len(objects['target']))
        return VisualScene(boxes['source'],boxes['target'],objects['source'],objects['target'],
            packet.timestamp,self._table_z(points[0]),source='fused_native_metric_surfaces_v1')

    @staticmethod
    def _table_z(points):
        good=np.isfinite(points).all(2)&(points[:,:,0]>-.12)&(points[:,:,0]<.55)&(points[:,:,1]>-.20)&(points[:,:,1]<.60)
        z=points[:,:,2][good]
        if len(z)<100:raise LocalizationFailure('Too little valid tabletop depth')
        counts,edges=np.histogram(z,bins=np.arange(.50,1.10,.001));i=int(counts.argmax())
        selected=z[(z>=edges[i])&(z<edges[i+1])]
        if not len(selected):raise LocalizationFailure('No tabletop height support')
        return float(np.median(selected))
