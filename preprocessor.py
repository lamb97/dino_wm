import torch
from einops import rearrange

class Preprocessor:
    def __init__(self, 
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform

    def normalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return actions * self.action_std + self.action_mean
    
    def normalize_proprios(self, proprio):
        '''
        input shape (..., proprio_dim)
        '''
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_states(self, state):
        '''
        input shape (..., state_dim)
        '''
        return (state - self.state_mean) / self.state_std

    def preprocess_obs_visual(self, obs_visual):
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        x = torch.as_tensor(obs_visual)

        # Planning inputs are typically (b, t, h, w, c). torchvision Resize expects
        # batched 4D tensors, so flatten (b, t) -> (b*t), transform, then restore.
        if x.ndim == 5:
            b, t = x.shape[:2]
            x = rearrange(x, "b t h w c -> (b t) c h w").float() / 255.0
            x = self.transform(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)
            return x

        # Fallback for already-batched visuals.
        if x.ndim == 4:
            if x.shape[-1] in (1, 3, 4):
                x = rearrange(x, "b h w c -> b c h w").float() / 255.0
            else:
                x = x.float()
                if x.max() > 1:
                    x = x / 255.0
            return self.transform(x)

        raise ValueError(f"Unsupported visual obs shape: {tuple(x.shape)}")
    
    def transform_obs(self, obs):
        '''
        np arrays to tensors
        '''
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
