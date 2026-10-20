"""Task-specific, calibrated RGB-D geometry. No environment or object IDs.

Assumptions are explicit: pale rectangular bins, orange upright cookies, a
horizontal table, bin yaw within 30 degrees, and four touching source columns.
The packing prior constrains source x coordinates; occluded cookies are rejected,
not filled in from that prior. This is not a general object-pose network.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import ConvexHull
from .rgbd import unproject,RGBDViews

LOCALIZER_VERSION = "rgbd_packing_geometry_v1"
TARGET_DETECTOR_VERSION = "rgbd_free_target_ridges_v2"


class LocalizationFailure(RuntimeError):
    pass


def rotation_z(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


@dataclass(frozen=True)
class BoxEstimate:
    position: np.ndarray
    rotation: np.ndarray
    inner_half_size: np.ndarray
    rim_z: float
    supported_sides: int
    extent_error_m: float


@dataclass(frozen=True)
class CookieEstimate:
    position: np.ndarray
    rotation: np.ndarray
    column: int | None
    coverage: float
    top_residual_m: float
    # x is constrained by the fitted packing layout, not independently measured.
    position_model: str = "packing_x_rgbd_yz"


@dataclass(frozen=True)
class VisualScene:
    source_box: BoxEstimate
    target_box: BoxEstimate
    source_cookies: tuple[CookieEstimate,...]
    target_cookies: tuple[CookieEstimate,...]
    timestamp: float
    table_z: float
    source: str = LOCALIZER_VERSION


def minimum_rectangle(xy):
    """Minimum-area enclosing rectangle; orientation resolved by table-axis prior."""
    hull=xy[ConvexHull(xy).vertices]
    edge=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.arctan2(edge[:,1],edge[:,0])%(np.pi/2))
    best=None
    for angle in angles:
        # The scene randomizer rotates bins by a small yaw, never a quarter turn.
        angle=(angle+np.pi/4)%(np.pi/2)-np.pi/4
        rot=rotation_z(angle)[:2,:2]
        q=hull@rot
        lo,hi=q.min(0),q.max(0)
        area=np.prod(hi-lo)
        if best is None or area<best[0]:
            best=(area,rot,lo,hi)
    return best[1:]


class RGBDLocalizer:
    def __init__(self, spec, *, min_coverage=.90, source_columns=4):
        if spec.get("schema") != "a3_rgbd_geometry_v2":
            raise ValueError("Use v2 geometry: source wall-centre and inner extents differ")
        self.spec=dict(spec)
        self.half=np.asarray(spec["cookie_half_size"],dtype=float)
        self.wall=float(spec["wall_thickness_m"])
        self.min_coverage=float(min_coverage)
        self.source_columns=int(source_columns)
        if not 0<self.min_coverage<=1 or self.source_columns<1:
            raise ValueError("Invalid visibility or packing configuration")

    def _boxes(self, frame, points, table_z):
        color=frame.rgb.astype(float)
        white=(color.min(2)>80)&(np.ptp(color,axis=2)<40)
        white &= np.isfinite(points).all(2)&(points[:,:,2]>table_z+.008)&(points[:,:,2]<table_z+.080)
        labels,n=ndimage.label(white)
        candidates=[]
        for identifier,region in enumerate(ndimage.find_objects(labels),1):
            if region is None:continue
            q=points[region][labels[region]==identifier]
            if len(q)<300:continue
            zhist,edges=np.histogram(q[:,2],bins=np.arange(table_z+.008,table_z+.081,.0004))
            peaks,_=signal.find_peaks(np.r_[0,zhist,0],height=max(80,len(q)*.025))
            if not len(peaks):continue
            rim_z=edges[min(int(peaks[-1]-1),len(edges)-2)]+.0002
            top=q[abs(q[:,2]-rim_z)<.00065]
            if len(top)<150:continue
            rim_z=float(np.median(top[:,2]))
            try:rot,lo,hi=minimum_rectangle(top[:,:2])
            except Exception:continue
            candidates.append((top,rot,lo,hi,rim_z))
        result={}
        for kind in ("source","target"):
            inner=np.asarray(self.spec[kind+"_inner_half_size"],dtype=float)
            outer=inner+2*self.wall
            ranked=[]
            for top,rot,lo,hi,rim_z in candidates:
                ext=hi-lo
                # At least three supported sides; no complete reconstruction of
                # a bin hidden on two axes is accepted.
                complete=abs(ext-2*outer)<.0025
                if not np.any(complete) or np.any(ext>2*outer+.003):continue
                if np.min(ext/(2*outer))<.55:continue
                center=(lo+hi)/2
                q=top[:,:2]@rot
                supported=int(2*complete.sum())
                for axis in np.flatnonzero(~complete):
                    # Only one end is occluded. A complete visible rim side has
                    # many more pixels than an arbitrary occlusion boundary.
                    support_lo=np.sum(q[:,axis]<lo[axis]+2*self.wall)
                    support_hi=np.sum(q[:,axis]>hi[axis]-2*self.wall)
                    if max(support_lo,support_hi)<2*min(support_lo,support_hi):
                        break
                    center[axis]=hi[axis]-outer[axis] if support_hi>support_lo else lo[axis]+outer[axis]
                    supported+=1
                else:
                    error=float(np.max(abs(ext[complete]-2*outer[complete])))
                    offset=(self.spec["source_wall_top_offset_m"] if kind=="source"
                            else self.spec["wall_height_m"])
                    position=np.r_[center@rot.T,rim_z-offset]
                    ranked.append((error+.003*(4-supported),BoxEstimate(position,
                        np.block([[rot,np.zeros((2,1))],[np.zeros((1,2)),np.ones((1,1))]]),
                        inner,rim_z,supported,error)))
            if not ranked:raise LocalizationFailure("Insufficient visible "+kind+" bin rim")
            result[kind]=min(ranked,key=lambda row:row[0])[1]
        return result["source"],result["target"]

    def _free_targets(self,q,box,*,allow_internal_occlusion=False):
        """Fit visible upright top ridges anywhere in the box, not at slot seeds.

        A misplaced object must still be detected. Slot agreement is evaluated
        separately by the controller and must never be a detector prerequisite.
        """
        if not len(q):return ()
        resolution=.0005
        lower=-box.inner_half_size
        shape=np.ceil(2*box.inner_half_size/resolution).astype(int)+1
        index=np.floor((q[:,:2]-lower)/resolution).astype(int)
        valid=(index>=0).all(1)&(index<shape).all(1)
        q,index=q[valid],index[valid]
        grid=np.zeros(tuple(shape[::-1]),dtype=bool)
        grid[index[:,1],index[:,0]]=True
        # Components support rotated ridges. Raster samples can be disconnected
        # in a sparse oblique view, so independent measured row/length support
        # below also proposes patches; neither proposal uses nominal slot x's.
        filled=ndimage.binary_closing(grid,structure=np.ones((1,3)))
        labels,count=ndimage.label(filled,structure=np.ones((3,3)))
        identifiers=labels[index[:,1],index[:,0]]
        patches=[q[identifiers==identifier] for identifier in range(1,count+1)]
        yedges=np.arange(-box.inner_half_size[1],box.inner_half_size[1]+.0004,.0004)
        density=ndimage.gaussian_filter1d(np.histogram(q[:,1],yedges)[0].astype(float),1.3)
        rows=ndimage.binary_closing(density>max(3,float(density.max())*.15),structure=np.ones(2))
        row_labels,_=ndimage.label(rows)
        for region in ndimage.find_objects(row_labels):
            segment=region[0]
            width=(segment.stop-segment.start)*.0004
            if not .0035<width<.0085:continue
            row=q[(q[:,1]>yedges[segment.start]-.0002)&(q[:,1]<yedges[segment.stop]+.0002)]
            # Held-object sensing may see both ends with the gripper hiding
            # the middle. Length/coverage still need actual measured support.
            if allow_internal_occlusion:patches.append(row)
            xedges=np.arange(-box.inner_half_size[0],box.inner_half_size[0]+.001,.001)
            density=ndimage.gaussian_filter1d(np.histogram(row[:,0],xedges)[0].astype(float),.65)
            spans=ndimage.binary_closing(density>max(.5,float(density.max())*.08),structure=np.ones(2))
            span_labels,_=ndimage.label(spans)
            for span in ndimage.find_objects(span_labels):
                s=span[0]
                patches.append(row[(row[:,0]>xedges[s.start]-.0005)&(row[:,0]<xedges[s.stop]+.0005)])
        estimates=[]
        for patch in sorted(patches,key=len,reverse=True):
            if len(patch)<25:continue
            values,vectors=np.linalg.eigh(np.cov(patch[:,:2].T))
            axis=vectors[:,-1]
            if axis[0]<0:axis=-axis
            perpendicular=np.array([-axis[1],axis[0]])
            basis=np.column_stack((axis,perpendicular))
            planar=patch[:,:2]@basis
            bounds=np.quantile(planar,[.01,.99],axis=0)
            extent=bounds[1]-bounds[0]
            if not 1.80*self.half[0]<extent[0]<2.15*self.half[0]:continue
            if not .0005<extent[1]<2.5*self.half[1]:continue
            bins=np.linspace(bounds[0,0],bounds[1,0],25)
            coverage=float(np.mean(np.histogram(planar[:,0],bins)[0]>=2))
            if coverage<self.min_coverage:continue
            center=(bounds[0]+bounds[1])/2@basis.T
            z=float(np.quantile(patch[:,2],.9)-self.half[2])
            residual=float(np.std(patch[:,2]))
            if residual>.0015:continue
            rotation=box.rotation@rotation_z(float(np.arctan2(axis[1],axis[0])))
            position=box.position+box.rotation@np.r_[center,z]
            if not any(np.linalg.norm(position-c.position)<.0035 for c in estimates):
                estimates.append(CookieEstimate(position,rotation,None,coverage,residual,TARGET_DETECTOR_VERSION))
        return tuple(sorted(estimates,key=lambda c:float((c.position-box.position)@box.rotation[:,1])))

    def _cookies(self, frame, points, box, *, source):
        r,g,b=frame.rgb.astype(float).transpose(2,0,1)
        local=(points-box.position)@box.rotation
        orange=(r>70)&(r>g*1.13)&(g>b*1.25)&np.isfinite(points).all(2)
        inside=(np.abs(local[:,:,:2])<box.inner_half_size).all(2)
        # Accept only the upright top band; fallen or stacked objects do not
        # masquerade as valid instances. Boundary verification treats absence
        # as unknown/failure, not proof of an empty target region.
        if source:
            floor=self.spec["source_floor_top_offset_m"]
        else:floor=self.spec["floor_offset_m"]
        expected_top=floor+2*self.half[2]
        mask=orange&inside&(abs(local[:,:,2]-expected_top)<(.004 if source else .001))
        q=local[mask]
        if not len(q):return ()
        if not source:return self._free_targets(q,box)
        xs=(np.arange(self.source_columns)-(self.source_columns-1)/2)*2*self.half[0]
        estimates=[]
        for column,x in enumerate(xs):
            # A known packing prior proposes cells; RGB-D must visibly support
            # every returned object. No SOURCE_POSITIONS or object IDs enter.
            strip=q[abs(q[:,0]-x)<self.half[0]-.001]
            if len(strip)<30:continue
            step=.0004
            edges=np.arange(-box.inner_half_size[1],box.inner_half_size[1]+step,step)
            counts,_=np.histogram(strip[:,1],edges)
            smooth=ndimage.gaussian_filter1d(counts.astype(float),1.3)
            occupied=smooth>max(3,float(smooth.max())*.15)
            occupied=ndimage.binary_closing(occupied,structure=np.ones(2))
            row_labels,_=ndimage.label(occupied)
            for row_slice in ndimage.find_objects(row_labels):
                segment=row_slice[0]
                width=(segment.stop-segment.start)*step
                if not .0035<width<.0085:continue
                lower,upper=edges[segment.start]-.0002,edges[segment.stop]+.0002
                patch=strip[(strip[:,1]>lower)&(strip[:,1]<upper)]
                if len(patch)<25:continue
                # Use occupied longitudinal bins to detect occlusion, rather
                # than mistaking one visible corner for a complete cookie.
                bins=np.linspace(x-self.half[0]+.001,x+self.half[0]-.001,25)
                support=np.histogram(patch[:,0],bins)[0]
                coverage=float(np.mean(support>=2))
                if coverage<self.min_coverage:continue
                # Symmetric visible bevels locate y; retain measured z and a
                # separate roughness diagnostic for independent audit.
                yl,yh=np.quantile(patch[:,1],[.02,.98])
                y=float((yl+yh)/2)
                z=float(np.quantile(patch[:,2],.9)-self.half[2])
                top=patch[patch[:,2]>np.quantile(patch[:,2],.65)]
                residual=float(np.std(top[:,2]))
                if residual>.0015:continue
                estimates.append(CookieEstimate(box.position+box.rotation@np.array([x,y,z]),
                    box.rotation.copy(),column+1,coverage,residual,"packing_x_rgbd_yz"))
        return tuple(sorted(estimates,key=lambda c:(c.column or 0,float((c.position-box.position)@box.rotation[:,1]))))

    def observe(self, frame):
        if isinstance(frame,RGBDViews):
            scenes=[]
            for image in frame.frames:
                try:scenes.append(self.observe(image))
                except LocalizationFailure:pass
            if not scenes:raise LocalizationFailure("No calibrated view supports both bin estimates")
            scenes.sort(key=lambda s:len(s.source_cookies)+len(s.target_cookies),reverse=True)
            base=scenes[0]
            def merge(name):
                result=list(getattr(base,name))
                for scene in scenes[1:]:
                    for cookie in getattr(scene,name):
                        same=[c for c in result if c.column==cookie.column and
                              ((abs(c.position[0]-cookie.position[0])<.018
                                and abs(c.position[1]-cookie.position[1])<.0045)
                               if name=="source_cookies" else np.linalg.norm(c.position-cookie.position)<.0035)]
                        if not same:result.append(cookie)
                return tuple(sorted(result,key=lambda c:(c.column or 0,float(c.position[1]))))
            return VisualScene(base.source_box,base.target_box,merge("source_cookies"),merge("target_cookies"),
                               frame.timestamp,base.table_z)
        points=unproject(frame)
        finite=np.isfinite(points).all(2)
        if finite.mean()<.40:raise LocalizationFailure("Too little valid metric depth")
        # Calibrated tabletop workspace excludes the robot base and distant room.
        work=finite&(points[:,:,0]>-.12)&(points[:,:,0]<.55)&(points[:,:,1]>-.20)&(points[:,:,1]<.60)
        z=points[:,:,2][work]
        bins=np.arange(.50,1.10,.001)
        counts,edges=np.histogram(z,bins)
        k=int(counts.argmax())
        table_z=float(np.median(z[(z>=edges[k])&(z<edges[k+1])]))
        source,target=self._boxes(frame,points,table_z)
        return VisualScene(source,target,self._cookies(frame,points,source,source=True),
            self._cookies(frame,points,target,source=False),frame.timestamp,table_z)


def select_visible_five(scene, *, column="auto"):
    """Return a measured exposed run. No invisible cells are filled in."""
    for col in sorted({c.column for c in scene.source_cookies}):
        if column!="auto" and col!=column:continue
        cookies=[c for c in scene.source_cookies if c.column==col]
        cookies.sort(key=lambda c:float(c.position@scene.source_box.rotation[:,1]))
        if len(cookies)<5:continue
        positions=np.asarray([c.position for c in cookies])@scene.source_box.rotation
        gaps=np.diff(positions[:,1])
        # Only an observed contiguous end run; an occlusion or a removed object
        # never bridges two short runs into a fictitious five-object grasp.
        for start in (0,len(cookies)-5):
            if np.all((gaps[start:start+4]>.0045)&(gaps[start:start+4]<.0115)):
                return tuple(cookies[start:start+5])
    raise LocalizationFailure("No fully visible contiguous five-cookie run; acquire another view")
