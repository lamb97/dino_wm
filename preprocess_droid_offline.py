#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch


def parse_csv_keys(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def concat_obs_keys(obs_step, keys: List[str]) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in obs_step:
            raise KeyError(f"Observation key '{key}' not found in episode.")
        arr = np.asarray(obs_step[key], dtype=np.float32).reshape(-1)
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def export_split_subsets(
    output_dir: Path,
    actions_padded: torch.Tensor,
    states_padded: torch.Tensor,
    seq_lengths_tensor: torch.Tensor,
    meta: dict,
    test_ratio: float,
    split_seed: int,
):
    num_eps = int(seq_lengths_tensor.shape[0])
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in (0,1), got {test_ratio}")
    n_test = int(round(num_eps * test_ratio))
    n_test = min(max(n_test, 1), num_eps - 1)

    rng = np.random.default_rng(split_seed)
    indices = np.arange(num_eps)
    rng.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    split_map = {"train": train_idx, "test": test_idx}
    src_obs_dir = output_dir / "obses"

    for split_name, split_idx in split_map.items():
        split_dir = output_dir / split_name
        split_obs_dir = split_dir / "obses"
        split_obs_dir.mkdir(parents=True, exist_ok=True)

        split_idx_t = torch.from_numpy(split_idx).long()
        split_actions = actions_padded.index_select(0, split_idx_t)
        split_states = states_padded.index_select(0, split_idx_t)
        split_seq = seq_lengths_tensor.index_select(0, split_idx_t)

        torch.save(split_actions, split_dir / "actions.pth")
        torch.save(split_states, split_dir / "states.pth")
        torch.save(split_seq, split_dir / "seq_lengths.pth")
        torch.save(split_idx_t, split_dir / "episode_indices.pth")

        for new_i, old_i in enumerate(split_idx.tolist()):
            src = src_obs_dir / f"episode_{old_i:03d}.pth"
            dst = split_obs_dir / f"episode_{new_i:03d}.pth"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        split_meta = dict(meta)
        split_meta.update(
            {
                "split_name": split_name,
                "num_episodes": int(split_seq.shape[0]),
                "split_seed": int(split_seed),
                "test_ratio": float(test_ratio),
            }
        )
        with open(split_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)


def _reshape_flat_feature(values: np.ndarray, t: int, key: str) -> np.ndarray:
    if t <= 0:
        raise ValueError(f"Invalid sequence length t={t} for key={key}")
    if values.size % t != 0:
        raise ValueError(
            f"Feature '{key}' length {values.size} is not divisible by traj_len {t}"
        )
    return values.reshape(t, -1)


def _compute_actions_from_mode(
    action_key: str,
    obs_cartesian: np.ndarray,
    actions_raw: Optional[np.ndarray] = None,
    gripper_action: Optional[np.ndarray] = None,
    obs_gripper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Supported modes:
      - standard action keys (use actions_raw directly)
      - delta_cartesian_position: obs_cartesian[t+1] - obs_cartesian[t], shape (T-1, 6)
    """
    if action_key == "delta_cartesian_position":
        if obs_cartesian.ndim != 2 or obs_cartesian.shape[1] != 6:
            raise ValueError(
                f"Expected observation/cartesian_position to have shape (T, 6), "
                f"got {tuple(obs_cartesian.shape)}"
            )
        if obs_cartesian.shape[0] < 2:
            return np.zeros((0, 6), dtype=np.float32)
        return np.diff(obs_cartesian, axis=0).astype(np.float32)

    if action_key == "delta_end_effector_state":
        if obs_cartesian.ndim != 2 or obs_cartesian.shape[1] != 6:
            raise ValueError(
                f"Expected observation/cartesian_position to have shape (T, 6), "
                f"got {tuple(obs_cartesian.shape)}"
            )
        if obs_gripper is None:
            raise ValueError(
                "obs_gripper is required for action_key='delta_end_effector_state'"
            )
        if obs_gripper.ndim == 1:
            obs_gripper = obs_gripper[:, None]
        if obs_gripper.ndim != 2 or obs_gripper.shape[1] != 1:
            raise ValueError(
                f"Expected obs_gripper shape (T, 1), got {tuple(obs_gripper.shape)}"
            )
        t = min(obs_cartesian.shape[0], obs_gripper.shape[0])
        if t < 2:
            return np.zeros((0, 7), dtype=np.float32)
        full_state = np.concatenate([obs_cartesian[:t], obs_gripper[:t]], axis=1)
        return np.diff(full_state, axis=0).astype(np.float32)

    if action_key == "delta_cartesian_position_gripper_position":
        if obs_cartesian.ndim != 2 or obs_cartesian.shape[1] != 6:
            raise ValueError(
                f"Expected observation/cartesian_position to have shape (T, 6), "
                f"got {tuple(obs_cartesian.shape)}"
            )
        if gripper_action is None:
            raise ValueError(
                "gripper_action is required for action_key='delta_cartesian_position_gripper_position'"
            )
        if gripper_action.ndim == 1:
            gripper_action = gripper_action[:, None]
        if gripper_action.ndim != 2 or gripper_action.shape[1] != 1:
            raise ValueError(
                f"Expected gripper_action shape (T, 1), got {tuple(gripper_action.shape)}"
            )
        t = min(obs_cartesian.shape[0], gripper_action.shape[0])
        if t < 2:
            return np.zeros((0, 7), dtype=np.float32)
        delta = np.diff(obs_cartesian[:t], axis=0).astype(np.float32)
        grip = gripper_action[: t - 1].astype(np.float32)
        return np.concatenate([delta, grip], axis=1)

    if actions_raw is None:
        raise ValueError(f"actions_raw is required for action_key='{action_key}'")
    return actions_raw


def _get_step_value(step, key: str):
    if key in step:
        return step[key]
    cur = step
    for tok in key.split("/"):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return None
    return cur


def iterate_tfds_episodes(
    input_dir: Path,
    split: str,
    camera_key: str,
    action_key: str,
    state_keys: List[str],
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(input_dir))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    for episode in ds:
        traj_imgs = []
        traj_states = []
        traj_cartesian = []
        traj_obs_gripper = []
        traj_actions_raw = []
        traj_gripper_action = []
        for step in tfds.as_numpy(episode["steps"]):
            obs_step = step["observation"]
            if camera_key not in obs_step:
                raise KeyError(f"camera_key '{camera_key}' not found in observation.")
            if "cartesian_position" not in obs_step:
                raise KeyError(
                    "Observation key 'cartesian_position' not found; "
                    "required for delta_cartesian_position mode."
                )
            traj_imgs.append(np.asarray(obs_step[camera_key], dtype=np.uint8))
            traj_cartesian.append(
                np.asarray(obs_step["cartesian_position"], dtype=np.float32).reshape(-1)
            )
            if "gripper_position" not in obs_step:
                raise KeyError(
                    "Observation key 'gripper_position' not found; required for delta_end_effector_state mode."
                )
            traj_obs_gripper.append(
                np.asarray(obs_step["gripper_position"], dtype=np.float32).reshape(-1)
            )
            if action_key not in (
                "delta_cartesian_position",
                "delta_cartesian_position_gripper_position",
                "delta_end_effector_state",
            ):
                action_step = _get_step_value(step, action_key)
                if action_step is None:
                    raise KeyError(f"Action key '{action_key}' not found in step.")
                traj_actions_raw.append(np.asarray(action_step, dtype=np.float32))
            if action_key == "delta_cartesian_position_gripper_position":
                g = _get_step_value(step, "action_dict/gripper_position")
                if g is None:
                    raise KeyError(
                        "Action key 'action_dict/gripper_position' not found in step."
                    )
                traj_gripper_action.append(np.asarray(g, dtype=np.float32).reshape(-1))
            traj_states.append(concat_obs_keys(obs_step, state_keys))
        if len(traj_imgs) == 0:
            continue
        cartesian = np.stack(traj_cartesian, axis=0)
        obs_gripper = np.stack(traj_obs_gripper, axis=0)
        if action_key == "delta_cartesian_position":
            actions = _compute_actions_from_mode(
                action_key=action_key,
                obs_cartesian=cartesian,
                actions_raw=None,
            )
        elif action_key == "delta_end_effector_state":
            actions = _compute_actions_from_mode(
                action_key=action_key,
                obs_cartesian=cartesian,
                actions_raw=None,
                obs_gripper=obs_gripper,
            )
        elif action_key == "delta_cartesian_position_gripper_position":
            gripper_action = np.stack(traj_gripper_action, axis=0)
            actions = _compute_actions_from_mode(
                action_key=action_key,
                obs_cartesian=cartesian,
                actions_raw=None,
                gripper_action=gripper_action,
            )
        else:
            actions = _compute_actions_from_mode(
                action_key=action_key,
                obs_cartesian=cartesian,
                actions_raw=np.stack(traj_actions_raw, axis=0),
            )
        yield (
            np.stack(traj_imgs, axis=0),
            actions,
            np.stack(traj_states, axis=0),
        )


def iterate_legacy_tfrecord_episodes(
    input_dir: Path,
    split: str,
    camera_key: str,
    action_key: str,
    state_keys: List[str],
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import tensorflow as tf

    split_files = sorted(input_dir.glob(f"*-{split}.tfrecord-*"))
    if len(split_files) == 0:
        split_files = sorted(input_dir.glob("*.tfrecord-*"))
    if len(split_files) == 0:
        raise FileNotFoundError(f"No tfrecord shards found under {input_dir}")

    cam_feat_key = (
        camera_key if camera_key.startswith("steps/observation/") else f"steps/observation/{camera_key}"
    )
    use_delta_cartesian = action_key in (
        "delta_cartesian_position",
        "delta_cartesian_position_gripper_position",
        "delta_end_effector_state",
    )
    act_feat_key = action_key if action_key.startswith("steps/") else f"steps/{action_key}"
    gripper_feat_key = "steps/action_dict/gripper_position"
    state_feat_keys = [
        k if k.startswith("steps/observation/") else f"steps/observation/{k}" for k in state_keys
    ]
    obs_cartesian_key = "steps/observation/cartesian_position"
    obs_gripper_key = "steps/observation/gripper_position"
    t_ref_key = "steps/is_first"

    for shard in split_files:
        for raw in tf.compat.v1.io.tf_record_iterator(str(shard)):
            ex = tf.train.Example.FromString(raw)
            feats = ex.features.feature
            if t_ref_key not in feats:
                raise KeyError(f"Missing '{t_ref_key}' in record from {shard}")
            t = len(feats[t_ref_key].int64_list.value)
            if t < 2:
                continue

            if cam_feat_key not in feats:
                raise KeyError(f"Missing camera feature '{cam_feat_key}' in record.")
            img_bytes = feats[cam_feat_key].bytes_list.value
            if len(img_bytes) != t:
                t = min(t, len(img_bytes))
            imgs = []
            for b in img_bytes[:t]:
                img = tf.io.decode_image(
                    b, channels=3, expand_animations=False
                ).numpy().astype(np.uint8)
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)

            if obs_cartesian_key not in feats:
                raise KeyError(
                    "Missing 'steps/observation/cartesian_position' in record; "
                    "required for delta_cartesian_position mode."
                )
            obs_cartesian_vals = np.asarray(
                feats[obs_cartesian_key].float_list.value, dtype=np.float32
            )
            obs_cartesian = _reshape_flat_feature(
                obs_cartesian_vals, t, obs_cartesian_key
            )

            if use_delta_cartesian:
                gripper_action = None
                obs_gripper = None
                if action_key == "delta_end_effector_state":
                    if obs_gripper_key not in feats:
                        raise KeyError(
                            "Missing 'steps/observation/gripper_position' in record; "
                            "required for delta_end_effector_state mode."
                        )
                    obs_grip_vals = np.asarray(
                        feats[obs_gripper_key].float_list.value, dtype=np.float32
                    )
                    obs_gripper = _reshape_flat_feature(
                        obs_grip_vals, t, obs_gripper_key
                    )
                if action_key == "delta_cartesian_position_gripper_position":
                    if gripper_feat_key not in feats:
                        raise KeyError(
                            "Missing 'steps/action_dict/gripper_position' in record; "
                            "required for delta_cartesian_position_gripper_position mode."
                        )
                    grip_vals = np.asarray(
                        feats[gripper_feat_key].float_list.value, dtype=np.float32
                    )
                    gripper_action = _reshape_flat_feature(
                        grip_vals, t, gripper_feat_key
                    )
                actions = _compute_actions_from_mode(
                    action_key=action_key,
                    obs_cartesian=obs_cartesian,
                    actions_raw=None,
                    gripper_action=gripper_action,
                    obs_gripper=obs_gripper,
                )
            else:
                if act_feat_key not in feats:
                    raise KeyError(f"Missing action feature '{act_feat_key}' in record.")
                act_vals = np.asarray(feats[act_feat_key].float_list.value, dtype=np.float32)
                actions_raw = _reshape_flat_feature(act_vals, t, act_feat_key)
                actions = _compute_actions_from_mode(
                    action_key=action_key,
                    obs_cartesian=obs_cartesian,
                    actions_raw=actions_raw,
                )

            state_parts = []
            for s_key in state_feat_keys:
                if s_key not in feats:
                    raise KeyError(f"Missing state feature '{s_key}' in record.")
                s_vals = np.asarray(feats[s_key].float_list.value, dtype=np.float32)
                s_arr = _reshape_flat_feature(s_vals, t, s_key)
                state_parts.append(s_arr)
            states = np.concatenate(state_parts, axis=1)
            yield imgs, actions, states


def main():
    parser = argparse.ArgumentParser(
        description="Convert DROID TFDS directory into offline .pth files for dino_wm."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="TFDS builder directory, e.g. /.../droid_100/1.0.0",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output folder, e.g. /.../datasets/droid_pth",
    )
    parser.add_argument("--split", type=str, default="train", help="TFDS split name.")
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="If set, only convert first N episodes.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="wrist_image_left",
        help="Observation image key to export into obses/episode_xxx.pth",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help=(
            "Action source. Example: 'action', 'action_dict/cartesian_velocity', "
            "'delta_cartesian_position' (6D), or "
            "'delta_cartesian_position_gripper_position' (7D), or "
            "'delta_end_effector_state' (7D: delta(cartesian_position+gripper_position))."
        ),
    )
    parser.add_argument(
        "--state-keys",
        type=str,
        default="joint_position,gripper_position",
        help="Comma-separated observation keys to concatenate as state.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="If set, additionally export episode-level train/test subsets under output-dir/{train,test}.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for episode-level train/test split.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    obses_dir = output_dir / "obses"
    obses_dir.mkdir(parents=True, exist_ok=True)

    state_keys = parse_csv_keys(args.state_keys)
    if len(state_keys) == 0:
        raise ValueError("--state-keys must contain at least one key.")

    import tensorflow as tf

    # Keep TF from reserving GPU memory when only decoding records.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    actions_list = []
    states_list = []
    seq_lengths = []

    num_eps = 0
    max_len = 0
    action_dim = None
    state_dim = None

    use_tfds = False
    tfds_err = None
    try:
        import tensorflow_datasets as tfds

        _ = tfds.builder_from_directory(str(input_dir))
        use_tfds = True
    except Exception as e:
        tfds_err = e

    if use_tfds:
        episode_iter = iterate_tfds_episodes(
            input_dir=input_dir,
            split=args.split,
            camera_key=args.camera_key,
            action_key=args.action_key,
            state_keys=state_keys,
        )
        source_format = "tfds_builder"
    else:
        print(
            f"TFDS builder_from_directory failed ({type(tfds_err).__name__}: {tfds_err}). "
            "Falling back to legacy TFRecord parser."
        )
        episode_iter = iterate_legacy_tfrecord_episodes(
            input_dir=input_dir,
            split=args.split,
            camera_key=args.camera_key,
            action_key=args.action_key,
            state_keys=state_keys,
        )
        source_format = "legacy_tfrecord"

    for ep_idx, (imgs, actions, states) in enumerate(episode_iter):
        if args.max_rollouts is not None and ep_idx >= args.max_rollouts:
            break
        traj_len = min(len(imgs), len(actions), len(states))
        if traj_len < 2:
            continue

        imgs = imgs[:traj_len]
        actions = actions[:traj_len]
        states = states[:traj_len]

        if action_dim is None:
            action_dim = actions.shape[-1]
            state_dim = states.shape[-1]
        else:
            if actions.shape[-1] != action_dim:
                raise ValueError(
                    f"Inconsistent action dim at episode {ep_idx}: "
                    f"{actions.shape[-1]} vs {action_dim}"
                )
            if states.shape[-1] != state_dim:
                raise ValueError(
                    f"Inconsistent state dim at episode {ep_idx}: "
                    f"{states.shape[-1]} vs {state_dim}"
                )

        torch.save(torch.from_numpy(imgs).contiguous(), obses_dir / f"episode_{num_eps:03d}.pth")
        actions_list.append(torch.from_numpy(actions).contiguous())
        states_list.append(torch.from_numpy(states).contiguous())
        seq_lengths.append(traj_len)
        max_len = max(max_len, traj_len)
        num_eps += 1

        if num_eps % 10 == 0:
            print(f"Converted {num_eps} episodes ...")

    if num_eps == 0:
        raise ValueError("No valid episodes converted.")

    actions_padded = torch.zeros((num_eps, max_len, action_dim), dtype=torch.float32)
    states_padded = torch.zeros((num_eps, max_len, state_dim), dtype=torch.float32)
    for i in range(num_eps):
        t = seq_lengths[i]
        actions_padded[i, :t] = actions_list[i]
        states_padded[i, :t] = states_list[i]

    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
    torch.save(actions_padded, output_dir / "actions.pth")
    torch.save(states_padded, output_dir / "states.pth")
    torch.save(seq_lengths_tensor, output_dir / "seq_lengths.pth")

    meta = {
        "input_dir": str(input_dir),
        "split": args.split,
        "source_format": source_format,
        "num_episodes": num_eps,
        "max_seq_len": int(max_len),
        "action_dim": int(action_dim),
        "state_dim": int(state_dim),
        "camera_key": args.camera_key,
        "action_key": args.action_key,
        "state_keys": state_keys,
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.test_ratio is not None:
        export_split_subsets(
            output_dir=output_dir,
            actions_padded=actions_padded,
            states_padded=states_padded,
            seq_lengths_tensor=seq_lengths_tensor,
            meta=meta,
            test_ratio=float(args.test_ratio),
            split_seed=int(args.split_seed),
        )

    print("Done.")
    print(f"Output: {output_dir}")
    print(f"Episodes: {num_eps}, max_seq_len: {max_len}")


if __name__ == "__main__":
    main()
