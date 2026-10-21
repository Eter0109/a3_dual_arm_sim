"""Zero-initialized learned geometry residual on the existing A3 state token.

Process-local PI05 wrappers only. Existing weights/config/state normalization
remain intact. A checkpoint with this head must use GeometryPi05Adapter.
"""
from .geometry_context import GEOMETRY_DIM,GEOMETRY_KEY,GEOMETRY_VERSION


def attach_geometry_head(model,hidden_dim=128):
    import torch
    if hasattr(model,'a3_geometry_proj'):raise RuntimeError('Geometry head already attached')
    width=model.state_proj.weight.shape[0]
    head=torch.nn.Sequential(torch.nn.Linear(GEOMETRY_DIM,hidden_dim),torch.nn.SiLU(),
                             torch.nn.Linear(hidden_dim,width,bias=False))
    torch.nn.init.zeros_(head[-1].weight)
    head.to(device=model.state_proj.weight.device,dtype=model.state_proj.weight.dtype)
    model.add_module('a3_geometry_proj',head)
    return head


def add_geometry_to_state_token(model,embeddings,geometry):
    import math
    import torch
    if geometry is None:return embeddings
    if geometry.ndim!=2 or geometry.shape!=(embeddings.shape[0],GEOMETRY_DIM):
        raise ValueError('Expected B,30 geometry separate from left8 state')
    if not torch.isfinite(geometry).all():raise ValueError('Nonfinite geometry context')
    flags=geometry[:,27:]
    if not torch.all((flags==0)|(flags==1)):raise ValueError('Geometry validity flags must be binary')
    # Invalid poses must be zero, preventing unavailable labels leaking into
    # training and making all-zero original-data replay exactly the old path.
    poses=geometry[:,:27].reshape(-1,3,9)
    if torch.any(poses.masked_select(~flags.bool().unsqueeze(-1).expand_as(poses))!=0):
        raise ValueError('Invalid geometry pose must contain zeros')
    weight=model.a3_geometry_proj[0].weight
    residual=model.a3_geometry_proj(geometry.to(device=weight.device,dtype=weight.dtype))
    gate=flags.any(-1).to(device=residual.device,dtype=residual.dtype).unsqueeze(-1)
    token=embeddings[:,-1:]+(residual*gate*math.sqrt(embeddings.shape[-1]))[:,None].to(embeddings.dtype)
    return torch.cat([embeddings[:,:-1],token],dim=1)


def install_geometry_model():
    from lerobot.policies.pi05 import modeling_pi05 as module
    cls=module.PI05Pytorch;policy_cls=module.PI05Policy
    if getattr(cls.embed_prefix,'_a3_geometry',False):return
    old_embed,old_forward,old_chunk=cls.embed_prefix,policy_cls.forward,policy_cls.predict_action_chunk

    def embed_prefix(self,*args,**kwargs):
        state_present=getattr(self,'_a3_state',None) is not None
        embeddings,pad,attention=old_embed(self,*args,**kwargs)
        geometry=getattr(self,'_a3_geometry',None)
        self._a3_geometry=None
        if geometry is not None:
            if not state_present or not hasattr(self,'a3_geometry_proj'):
                raise RuntimeError('Geometry requires the trained A3 state token and registered head')
            embeddings=add_geometry_to_state_token(self,embeddings,geometry)
        return embeddings,pad,attention

    def forward(self,batch,*args,**kwargs):
        self.model._a3_geometry=batch.get(GEOMETRY_KEY)
        try:return old_forward(self,batch,*args,**kwargs)
        finally:self.model._a3_geometry=None

    def chunk(self,batch,*args,**kwargs):
        self.model._a3_geometry=batch.get(GEOMETRY_KEY)
        try:return old_chunk(self,batch,*args,**kwargs)
        finally:self.model._a3_geometry=None

    embed_prefix._a3_geometry=True
    cls.embed_prefix=embed_prefix;policy_cls.forward=forward;policy_cls.predict_action_chunk=chunk


def geometry_descriptor(**extra):
    return dict(geometry_conditioning=GEOMETRY_VERSION,geometry_dim=GEOMETRY_DIM,
        hidden_dim=128,injection='zero_initialized_residual_on_trained_state_token',
        translation_scale_m=.1,geometry_normalization='explicit_units_no_dataset_normalizer',
        observation_source='calibrated_RGBD_and_robot_FK_only',**extra)
