"""Load genuinely geometry-trained weights and emit learned left8 actions only."""
import json
from pathlib import Path
import numpy as np
from .geometry_context import GEOMETRY_KEY,GEOMETRY_VERSION,GEOMETRY_DIM
from .geometry_model import attach_geometry_head,install_geometry_model


class GeometryPi05Adapter:
    action_mode='joint_position'
    name='pi05_left8_learned_geometry'

    def __init__(self,checkpoint,job,device='cuda',inference_seed=1000):
        import torch
        import pi05_adapter as legacy
        metadata=json.loads((Path(checkpoint)/'geometry_conditioning.json').read_text())
        if (metadata.get('geometry_conditioning')!=GEOMETRY_VERSION
                or metadata.get('geometry_dim')!=GEOMETRY_DIM
                or metadata.get('hidden_dim')!=128
                or metadata.get('training_completed') is not True):
            raise ValueError('Requires geometry weights that have completed real training')
        legacy.install_state_projector(job)
        self.config,self.pre,self.post=legacy.load_processors(checkpoint,job,device)
        install_geometry_model()
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        class GeometryPolicy(PI05Policy):
            def __init__(instance,config):
                super().__init__(config);attach_geometry_head(instance.model,metadata['hidden_dim'])
            def load_state_dict(instance,*args,**kwargs):
                result=super().load_state_dict(*args,**kwargs)
                instance.loaded_completely=not result.missing_keys and not result.unexpected_keys
                return result
        self.policy=GeometryPolicy.from_pretrained(checkpoint,config=self.config,strict=True)
        if not getattr(self.policy,'loaded_completely',False):raise RuntimeError('Incomplete geometry weight load')
        self.policy.to(device).eval().requires_grad_(False)
        self.inference_seed=inference_seed;self.reset()

    def reset(self,context=None):
        import torch
        self._right_hold=None;self.last_raw_action=None;self.steps=0
        torch.manual_seed(self.inference_seed+(context.seed if context is not None else 0))
        self.policy.reset()

    def act(self,observation,task=''):
        import torch
        from pi05_adapter import observation_batch
        state=np.asarray(observation['observation.state'])
        if state.shape!=(16,):raise ValueError('Expected canonical environment state')
        geometry=np.asarray(observation[GEOMETRY_KEY],dtype=np.float32)
        if geometry.shape!=(GEOMETRY_DIM,):raise ValueError('Missing measured geometry30')
        if self._right_hold is None:self._right_hold=state[8:].copy()
        batch=self.pre(observation_batch(observation,task))
        batch[GEOMETRY_KEY]=torch.from_numpy(geometry.copy())[None].to(self.config.device)
        with torch.inference_mode():
            action=self.post(self.policy.select_action(batch)).detach().float().cpu().numpy().reshape(-1)
        if action.shape!=(8,) or not np.isfinite(action).all():raise ValueError('Invalid learned left8 action')
        full=np.r_[action,self._right_hold];self.last_raw_action=full.copy();self.steps+=1
        return full

    def close(self):
        import torch
        self.policy=None
        if torch.cuda.is_available():torch.cuda.empty_cache()
