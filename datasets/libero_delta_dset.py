import re
import json
import numpy as np
import torch
from pathlib import Path
from bisect import bisect_right
from typing import Optional, Callable, Sequence, Union
from collections import OrderedDict
from omegaconf import ListConfig

from .traj_dset import TrajDataset, get_train_val_sliced


_EPISODE_RE = re.compile(r"episode_(\d+)\.(pth|npy)$")


class LiberoDeltaDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        use_mmap: bool = False,
        mmap_mode: str = "r",
        image_use_mmap: bool = True,
        image_mmap_mode: str = "r",
        image_cache_size: int = 8,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.use_mmap = use_mmap
        self.mmap_mode = mmap_mode

        self.image_use_mmap = image_use_mmap
        self.image_mmap_mode = image_mmap_mode
        self.image_cache_size = max(int(image_cache_size), 1)
        self._image_cache = OrderedDict()
        self._image_lengths = {}
        self._episode_meta = {}
        self._sim_state_dir = self.data_path / "sim_states"

        self.states = None
        self.actions = None
        self.seq_lengths = None

        states_npy = self.data_path / "states.npy"
        actions_npy = self.data_path / "actions.npy"
        seq_npy = self.data_path / "seq_lengths.npy"

        if use_mmap and states_npy.exists() and actions_npy.exists() and seq_npy.exists():
            self.states = np.load(states_npy, mmap_mode=mmap_mode)
            self.actions = np.load(actions_npy, mmap_mode=mmap_mode)
            self.seq_lengths = np.load(seq_npy, mmap_mode=mmap_mode).astype(np.int64)
            self.storage_mode = "mmap"
        else:
            self.states = torch.load(self.data_path / "states.pth").float()
            self.actions = torch.load(self.data_path / "actions.pth").float()
            self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth").long().numpy()
            self.storage_mode = "memory"

        total_rollouts = int(self.seq_lengths.shape[0])
        if n_rollout is not None:
            self.n_rollout = min(int(n_rollout), total_rollouts)
        else:
            self.n_rollout = total_rollouts

        if self.storage_mode == "mmap":
            self.states = self.states[: self.n_rollout]
            self.actions = self.actions[: self.n_rollout]
        else:
            self.states = self.states[: self.n_rollout]
            self.actions = self.actions[: self.n_rollout]
        self.seq_lengths = self.seq_lengths[: self.n_rollout]

        self.state_dim = int(self._shape_last_dim(self.states))
        self.action_dim = int(self._shape_last_dim(self.actions))
        self.proprio_dim = self.state_dim

        self._pth_obs_index = self._build_obs_index(self.data_path / "obses", suffix=".pth")
        self._npy_obs_index = self._build_obs_index(self.data_path / "obses_npy", suffix=".npy")
        self._load_episode_meta()

        if self.image_use_mmap and len(self._npy_obs_index) > 0:
            self.image_storage_mode = "mmap_npy"
        elif len(self._pth_obs_index) > 0:
            if self.image_use_mmap:
                print(
                    f"[LiberoDeltaDataset] image_use_mmap=True but no files found in "
                    f"{(self.data_path / 'obses_npy')}. Falling back to torch .pth images."
                )
            self.image_storage_mode = "torch_pth"
        else:
            raise FileNotFoundError(
                f"No image files found in {(self.data_path / 'obses')} or {(self.data_path / 'obses_npy')}"
            )

        if normalize_action:
            self.action_mean, self.action_std = self._compute_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self._compute_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.state_mean.clone(), self.state_std.clone()
            self.action_std = torch.clamp(self.action_std, min=1e-6)
            self.state_std = torch.clamp(self.state_std, min=1e-6)
            self.proprio_std = torch.clamp(self.proprio_std, min=1e-6)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        print(
            f"Loaded {self.n_rollout} LIBERO trajectories from {self.data_path} "
            f"(state/action={self.storage_mode}, images={self.image_storage_mode}, normalize_action={self.normalize_action})"
        )

    def _load_episode_meta(self):
        meta_path = self.data_path / "episode_meta.jsonl"
        if not meta_path.exists():
            return
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "converted_index" in entry:
                    self._episode_meta[int(entry["converted_index"])] = entry

    @staticmethod
    def _shape_last_dim(arr):
        return arr.shape[-1]

    @staticmethod
    def _build_obs_index(dir_path: Path, suffix: str):
        index = {}
        if not dir_path.exists():
            return index
        for p in sorted(dir_path.glob(f"*{suffix}")):
            m = _EPISODE_RE.match(p.name)
            if m is None:
                continue
            ep_idx = int(m.group(1))
            index[ep_idx] = p
        return index

    def _traj_slice(self, arr, idx: int, end_t: int):
        if isinstance(arr, torch.Tensor):
            return arr[idx, :end_t].cpu().numpy()
        return np.asarray(arr[idx, :end_t], dtype=np.float32)

    def _frame_slice(self, arr, idx: int, frames):
        if isinstance(arr, torch.Tensor):
            return arr[idx, frames].float()
        return torch.from_numpy(np.asarray(arr[idx, frames], dtype=np.float32))

    def _compute_mean_std(self, arr, traj_lengths):
        dim = int(self._shape_last_dim(arr))
        total_count = 0
        total_sum = np.zeros(dim, dtype=np.float64)
        total_sumsq = np.zeros(dim, dtype=np.float64)

        for traj_idx in range(self.n_rollout):
            t = int(traj_lengths[traj_idx])
            if t <= 0:
                continue
            x = self._traj_slice(arr, traj_idx, t).astype(np.float64)
            total_count += x.shape[0]
            total_sum += x.sum(axis=0)
            total_sumsq += (x * x).sum(axis=0)

        if total_count == 0:
            mean = np.zeros(dim, dtype=np.float32)
            std = np.ones(dim, dtype=np.float32)
        else:
            mean = total_sum / total_count
            var = total_sumsq / total_count - mean * mean
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            mean = mean.astype(np.float32)
            std = std.astype(np.float32)

        return torch.from_numpy(mean), torch.from_numpy(std)

    def _get_image_memmap(self, idx: int):
        if idx not in self._npy_obs_index:
            raise FileNotFoundError(f"No npy image file for idx={idx}")
        path = self._npy_obs_index[idx]
        key = str(path)
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached

        arr = np.load(path, mmap_mode=self.image_mmap_mode)
        self._image_cache[key] = arr
        if len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        return arr

    def _load_episode_obs(self, idx: int):
        if self.image_storage_mode == "mmap_npy":
            return self._get_image_memmap(idx)

        if idx not in self._pth_obs_index:
            raise FileNotFoundError(f"Cannot find obs file for idx={idx} in {self.data_path / 'obses'}")
        return torch.load(self._pth_obs_index[idx])

    def _get_image_length(self, idx: int) -> int:
        cached = self._image_lengths.get(idx)
        if cached is not None:
            return cached
        raw_image = self._load_episode_obs(idx)
        length = int(raw_image.shape[0])
        self._image_lengths[idx] = length
        return length

    def get_seq_length(self, idx):
        # Robust against mismatched metadata: use the shortest available modality.
        declared = int(self.seq_lengths[idx])
        state_t = int(self.states.shape[1])
        action_t = int(self.actions.shape[1])
        image_t = self._get_image_length(idx)
        return min(declared, state_t, action_t, image_t)

    def get_all_actions(self):
        chunks = []
        for i in range(self.n_rollout):
            t = self.get_seq_length(i)
            act = self._frame_slice(self.actions, i, slice(0, t))
            act = (act - self.action_mean) / self.action_std
            chunks.append(act)
        return torch.cat(chunks, dim=0)

    def get_frames(self, idx, frames):
        raw_image = self._load_episode_obs(idx)  # either np.memmap/ndarray or torch.Tensor
        max_t = int(raw_image.shape[0])
        if isinstance(frames, range):
            if len(frames) > 0 and frames.stop > max_t:
                raise IndexError(
                    f"Requested frame range [{frames.start}, {frames.stop}) exceeds image length "
                    f"{max_t} for idx={idx}. Check seq_lengths/image file alignment."
                )
        if isinstance(raw_image, torch.Tensor):
            image = raw_image[frames]
        else:
            image_np = np.asarray(raw_image[frames])  # keep uint8 until torch conversion
            image = torch.from_numpy(image_np)

        image = image.permute(0, 3, 1, 2).contiguous().float().div_(255.0)  # [T, C, H, W]
        if self.transform:
            image = self.transform(image)

        state = self._frame_slice(self.states, idx, frames)
        act = self._frame_slice(self.actions, idx, frames)

        if self.normalize_action:
            proprio = (state - self.proprio_mean) / self.proprio_std
            act = (act - self.action_mean) / self.action_std
        else:
            proprio = state

        obs = {"visual": image, "proprio": proprio}
        env_info = self._build_env_info(idx)
        return obs, act, state, env_info

    def _build_env_info(self, idx):
        env_info = {"episode_idx": int(idx)}
        meta = self._episode_meta.get(int(idx))
        if meta is not None:
            # Include task / scene metadata for env reconstruction in planning.
            for key in [
                "source_hdf5",
                "demo_key",
                "task",
                "env_name",
                "env_args",
                "problem_info",
                "has_sim_states",
                "sim_states_shape",
            ]:
                if key in meta:
                    env_info[key] = meta[key]

        sim_state_path = self._sim_state_dir / f"episode_{idx:06d}.pth"
        if sim_state_path.exists():
            env_info["sim_state_path"] = str(sim_state_path)
            # Keep low-dim state trajectory for matching random sampled offsets back to sim-states.
            traj_len = self.get_seq_length(idx)
            env_info["state_traj"] = self._traj_slice(self.states, idx, traj_len)
        return env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.n_rollout

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        if isinstance(imgs, torch.Tensor):
            return imgs.permute(0, 3, 1, 2).contiguous() / 255.0
        raise TypeError(f"Unsupported image type: {type(imgs)}")


class MultiLiberoDeltaDataset(TrajDataset):
    def __init__(
        self,
        data_paths: Sequence[Union[str, Path]],
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        use_mmap: bool = False,
        mmap_mode: str = "r",
        image_use_mmap: bool = True,
        image_mmap_mode: str = "r",
        image_cache_size: int = 8,
    ):
        if len(data_paths) == 0:
            raise ValueError("data_paths must contain at least one dataset path")

        self.datasets = []
        for data_path in data_paths:
            self.datasets.append(
                LiberoDeltaDataset(
                    data_path=str(data_path),
                    n_rollout=None,
                    transform=transform,
                    normalize_action=False,
                    use_mmap=use_mmap,
                    mmap_mode=mmap_mode,
                    image_use_mmap=image_use_mmap,
                    image_mmap_mode=image_mmap_mode,
                    image_cache_size=image_cache_size,
                )
            )

        self._dataset_lengths = [len(d) for d in self.datasets]
        self._dataset_offsets = [0]
        for length in self._dataset_lengths:
            self._dataset_offsets.append(self._dataset_offsets[-1] + int(length))

        total_rollouts = self._dataset_offsets[-1]
        if n_rollout is not None:
            self.n_rollout = min(int(n_rollout), total_rollouts)
        else:
            self.n_rollout = total_rollouts

        self.state_dim = self.datasets[0].state_dim
        self.action_dim = self.datasets[0].action_dim
        self.proprio_dim = self.datasets[0].proprio_dim
        for dset in self.datasets[1:]:
            if dset.state_dim != self.state_dim or dset.action_dim != self.action_dim:
                raise ValueError(
                    "All datasets must share the same state/action dimensions to be concatenated"
                )

        self.transform = transform
        self.normalize_action = normalize_action
        self.action_mean, self.action_std = self._compute_combined_mean_std(kind="action")
        self.state_mean, self.state_std = self._compute_combined_mean_std(kind="state")
        self.proprio_mean, self.proprio_std = self.state_mean.clone(), self.state_std.clone()
        self.action_std = torch.clamp(self.action_std, min=1e-6)
        self.state_std = torch.clamp(self.state_std, min=1e-6)
        self.proprio_std = torch.clamp(self.proprio_std, min=1e-6)

        if not normalize_action:
            self.action_mean.zero_()
            self.action_std.fill_(1.0)
            self.state_mean.zero_()
            self.state_std.fill_(1.0)
            self.proprio_mean.zero_()
            self.proprio_std.fill_(1.0)

        joined_paths = ", ".join(str(Path(p)) for p in data_paths)
        print(f"Loaded {self.n_rollout} LIBERO trajectories from [{joined_paths}]")

    def _compute_combined_mean_std(self, kind: str):
        dim = self.action_dim if kind == "action" else self.state_dim
        total_count = 0
        total_sum = np.zeros(dim, dtype=np.float64)
        total_sumsq = np.zeros(dim, dtype=np.float64)

        remaining = self.n_rollout
        for dset in self.datasets:
            if remaining <= 0:
                break
            take = min(len(dset), remaining)
            remaining -= take
            for traj_idx in range(take):
                t = dset.get_seq_length(traj_idx)
                if t <= 0:
                    continue
                arr = dset.actions if kind == "action" else dset.states
                x = dset._traj_slice(arr, traj_idx, t).astype(np.float64)
                total_count += x.shape[0]
                total_sum += x.sum(axis=0)
                total_sumsq += (x * x).sum(axis=0)

        if total_count == 0:
            mean = np.zeros(dim, dtype=np.float32)
            std = np.ones(dim, dtype=np.float32)
        else:
            mean = total_sum / total_count
            var = total_sumsq / total_count - mean * mean
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            mean = mean.astype(np.float32)
            std = std.astype(np.float32)

        return torch.from_numpy(mean), torch.from_numpy(std)

    def _locate_index(self, idx: int):
        if idx < 0 or idx >= self.n_rollout:
            raise IndexError(f"Index {idx} out of range for {self.n_rollout} trajectories")
        dataset_idx = bisect_right(self._dataset_offsets, idx) - 1
        local_idx = idx - self._dataset_offsets[dataset_idx]
        return dataset_idx, local_idx

    def get_seq_length(self, idx):
        dataset_idx, local_idx = self._locate_index(int(idx))
        return self.datasets[dataset_idx].get_seq_length(local_idx)

    def get_frames(self, idx, frames):
        dataset_idx, local_idx = self._locate_index(int(idx))
        obs, act, state, env_info = self.datasets[dataset_idx].get_frames(local_idx, frames)

        if self.normalize_action:
            obs["proprio"] = (state - self.proprio_mean) / self.proprio_std
            act = (act - self.action_mean) / self.action_std
        else:
            obs["proprio"] = state

        env_info = dict(env_info)
        env_info["episode_idx"] = int(idx)
        env_info["source_dataset_idx"] = int(dataset_idx)
        env_info["source_episode_idx"] = int(local_idx)
        return obs, act, state, env_info

    def get_all_actions(self):
        chunks = []
        for idx in range(self.n_rollout):
            t = self.get_seq_length(idx)
            _, act, _, _ = self.get_frames(idx, slice(0, t))
            chunks.append(act)
        return torch.cat(chunks, dim=0)

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.n_rollout


def load_libero_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    val_indices=None,
    num_hist=0,
    num_pred=0,
    frameskip=1,
    random_seed=42,
    use_mmap=False,
    mmap_mode="r",
    image_use_mmap=True,
    image_mmap_mode="r",
    image_cache_size=8,
):
    if isinstance(data_path, (list, tuple, ListConfig)):
        data_paths = [str(Path(path)) for path in data_path]
        dset = MultiLiberoDeltaDataset(
            data_paths=data_paths,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_mmap=use_mmap,
            mmap_mode=mmap_mode,
            image_use_mmap=image_use_mmap,
            image_mmap_mode=image_mmap_mode,
            image_cache_size=image_cache_size,
        )
    else:
        dset = LiberoDeltaDataset(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_mmap=use_mmap,
            mmap_mode=mmap_mode,
            image_use_mmap=image_use_mmap,
            image_mmap_mode=image_mmap_mode,
            image_cache_size=image_cache_size,
        )

    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        random_seed=random_seed,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
        val_indices=val_indices,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
