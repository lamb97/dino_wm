import numpy as np
import torch
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional, Sequence

from .traj_dset import TrajDataset, get_train_val_sliced


class DroidDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        camera_key: str = "wrist_image_left",
        action_key: str = "action",
        proprio_keys: Sequence[str] = ("joint_position", "gripper_position"),
        state_keys: Sequence[str] = ("joint_position", "gripper_position"),
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.normalize_action = normalize_action
        self.camera_key = camera_key
        self.action_key = action_key
        self.proprio_keys = list(proprio_keys)
        self.state_keys = list(state_keys)

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow_datasets is required for DroidDataset. "
                "Please install tensorflow-datasets in your env."
            ) from exc

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        builder = tfds.builder_from_directory(str(self.data_path))
        ds = builder.as_dataset(split=self.split, shuffle_files=False)

        self.images = []
        self.actions = []
        self.proprios = []
        self.states = []
        seq_lengths = []

        max_rollouts = n_rollout if n_rollout is not None else None
        for i, episode in enumerate(ds):
            if max_rollouts is not None and i >= max_rollouts:
                break

            traj_image = []
            traj_action = []
            traj_proprio = []
            traj_state = []

            steps_ds = episode["steps"]
            for step in tfds.as_numpy(steps_ds):
                obs_step = step["observation"]
                if self.camera_key not in obs_step:
                    raise KeyError(
                        f"camera_key '{self.camera_key}' not found in DROID observation keys."
                    )
                traj_image.append(np.asarray(obs_step[self.camera_key], dtype=np.uint8))
                traj_action.append(np.asarray(step[self.action_key], dtype=np.float32))
                traj_proprio.append(self._concat_step_obs(obs_step, self.proprio_keys))
                traj_state.append(self._concat_step_obs(obs_step, self.state_keys))

            if len(traj_image) == 0:
                continue

            image = np.stack(traj_image, axis=0)
            action = np.stack(traj_action, axis=0)
            proprio = np.stack(traj_proprio, axis=0)
            state = np.stack(traj_state, axis=0)

            traj_len = min(image.shape[0], action.shape[0], proprio.shape[0], state.shape[0])
            if traj_len < 2:
                continue

            self.images.append(torch.from_numpy(image[:traj_len]).contiguous())
            self.actions.append(torch.from_numpy(action[:traj_len]).contiguous())
            self.proprios.append(torch.from_numpy(proprio[:traj_len]).contiguous())
            self.states.append(torch.from_numpy(state[:traj_len]).contiguous())
            seq_lengths.append(traj_len)

        if len(self.actions) == 0:
            raise ValueError(
                "No valid DROID episodes loaded. Check data_path/split and selected keys."
            )

        self.seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        self.action_dim = self.actions[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]

        if self.normalize_action:
            self.action_mean, self.action_std = self._get_data_mean_std(self.actions)
            self.proprio_mean, self.proprio_std = self._get_data_mean_std(self.proprios)
            self.state_mean, self.state_std = self._get_data_mean_std(self.states)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)

        self.actions = [
            (a - self.action_mean) / (self.action_std + 1e-6) for a in self.actions
        ]
        self.proprios = [
            (p - self.proprio_mean) / (self.proprio_std + 1e-6) for p in self.proprios
        ]

    def _concat_step_obs(self, obs_step, keys: Sequence[str]) -> np.ndarray:
        parts = []
        for key in keys:
            if key not in obs_step:
                raise KeyError(f"Observation key '{key}' not found in DROID episode.")
            arr = np.asarray(obs_step[key], dtype=np.float32).reshape(-1)
            parts.append(arr)
        return np.concatenate(parts, axis=0)

    def _get_data_mean_std(self, traj_list):
        all_data = torch.cat(traj_list, dim=0).float()
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0) + 1e-6
        return data_mean, data_std

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx].item())

    def __len__(self):
        return len(self.seq_lengths)

    def get_all_actions(self):
        return torch.cat(self.actions, dim=0)

    def get_frames(self, idx, frames):
        image = self.images[idx][frames]  # THWC uint8
        act = self.actions[idx][frames]
        state = self.states[idx][frames]
        proprio = self.proprios[idx][frames]

        image = rearrange(image.float() / 255.0, "t h w c -> t c h w")
        if self.transform:
            image = self.transform(image)

        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w").float() / 255.0
        raise NotImplementedError


def load_droid_slice_train_val(
    transform,
    data_path,
    split="train",
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    camera_key="wrist_image_left",
    action_key="action",
    proprio_keys=("joint_position", "gripper_position"),
    state_keys=("joint_position", "gripper_position"),
):
    dset = DroidDataset(
        data_path=data_path,
        split=split,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        camera_key=camera_key,
        action_key=action_key,
        proprio_keys=proprio_keys,
        state_keys=state_keys,
    )

    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
