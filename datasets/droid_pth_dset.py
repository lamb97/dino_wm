import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from collections import OrderedDict

from .traj_dset import TrajDataset, get_train_val_sliced


class DroidPTHDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        image_cache_size: int = 32,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.image_cache_size = int(image_cache_size)

        states = self._torch_load_cpu(self.data_path / "states.pth")
        actions = self._torch_load_cpu(self.data_path / "actions.pth")
        seq_lengths = self._torch_load_cpu(self.data_path / "seq_lengths.pth")

        if isinstance(seq_lengths, list):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        else:
            seq_lengths = seq_lengths.long()

        n = len(seq_lengths) if n_rollout is None else min(int(n_rollout), len(seq_lengths))
        self.states = states[:n]
        self.actions = actions[:n]
        self.seq_lengths = seq_lengths[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.state_dim
        self.image_paths = [
            self.data_path / "obses" / f"episode_{idx:03d}.pth"
            for idx in range(n)
        ]
        self._image_cache = OrderedDict()

        if normalize_action:
            self.action_mean, self.action_std = self._get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self._get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean = self.state_mean
            self.proprio_std = self.state_std
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

    def _torch_load_cpu(self, path: Path):
        try:
            return torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            # Older torch versions may not support mmap argument.
            return torch.load(path, map_location="cpu")

    def _get_data_mean_std(self, data, traj_lengths):
        total_count = 0
        data_sum = torch.zeros(data.shape[-1], dtype=torch.float64)
        data_sq_sum = torch.zeros(data.shape[-1], dtype=torch.float64)
        for traj in range(len(traj_lengths)):
            t = int(traj_lengths[traj].item())
            if t <= 0:
                continue
            chunk = data[traj, :t].to(dtype=torch.float64)
            data_sum += chunk.sum(dim=0)
            data_sq_sum += (chunk * chunk).sum(dim=0)
            total_count += t
        if total_count == 0:
            raise ValueError("No valid timesteps found to compute dataset statistics.")
        mean = data_sum / total_count
        var = data_sq_sum / total_count - mean * mean
        var = torch.clamp(var, min=1e-12)
        std = torch.sqrt(var)
        return mean.float(), (std + 1e-6).float()

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx].item())

    def __len__(self):
        return len(self.seq_lengths)

    def get_all_actions(self):
        chunks = []
        for i in range(len(self.seq_lengths)):
            t = int(self.seq_lengths[i].item())
            action = self.actions[i, :t].float()
            action = (action - self.action_mean) / (self.action_std + 1e-6)
            chunks.append(action)
        return torch.cat(chunks, dim=0)

    def get_frames(self, idx, frames):
        image = self._get_images(idx)[frames]  # THWC uint8
        action = self.actions[idx, frames].float()
        state = self.states[idx, frames].float()
        proprio = state

        image = rearrange(image.float() / 255.0, "t h w c -> t c h w")
        if self.transform:
            image = self.transform(image)
        action = (action - self.action_mean) / (self.action_std + 1e-6)
        proprio = (proprio - self.proprio_mean) / (self.proprio_std + 1e-6)
        obs = {"visual": image, "proprio": proprio}
        return obs, action, state, {}

    def get_slice(self, i, start, end, frameskip, num_frames):
        t = self.get_seq_length(i)
        need = num_frames * frameskip
        if t < need:
            raise IndexError(f"Trajectory {i} too short for slice: len={t}, required={need}")
        max_start = t - need
        start = int(min(max(int(start), 0), max_start))
        end = start + need

        image = self._get_images(i)[start:end:frameskip]  # THWC
        state = self.states[i, start:end:frameskip].float()
        proprio = state
        action = self.actions[i, start:end].float()  # keep dense actions for concat
        if image.shape[0] != num_frames or state.shape[0] != num_frames or proprio.shape[0] != num_frames:
            raise RuntimeError(
                f"Invalid sliced length: traj={i}, start={start}, end={end}, "
                f"image={tuple(image.shape)}, state={tuple(state.shape)}, proprio={tuple(proprio.shape)}"
            )
        action = (action - self.action_mean) / (self.action_std + 1e-6)
        proprio = (proprio - self.proprio_mean) / (self.proprio_std + 1e-6)
        action = rearrange(action, "(n f) d -> n (f d)", n=num_frames)

        image = rearrange(image.float() / 255.0, "t h w c -> t c h w")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, action, state

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def _get_images(self, idx):
        # Per-process LRU cache to avoid loading all episode images into RAM.
        if idx in self._image_cache:
            self._image_cache.move_to_end(idx)
            return self._image_cache[idx]

        img = torch.load(self.image_paths[idx])
        self._image_cache[idx] = img
        if len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        return img

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w").float() / 255.0
        raise NotImplementedError


def load_droid_pth_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/droid_pth",
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    max_slices_per_traj=None,
    image_cache_size=32,
):
    dset = DroidPTHDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        image_cache_size=image_cache_size,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
        max_slices_per_traj=max_slices_per_traj,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
