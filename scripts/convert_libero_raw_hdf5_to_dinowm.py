#!/usr/bin/env python3
"""Convert raw LIBERO HDF5 demos to DINO-WM-friendly tensors.

Output format:
- <output_dir>/obses/episode_XXXXXX.pth      (uint8 tensor, [T, H, W, C])
- <output_dir>/states.pth                    (float32 tensor, [N, T_max, 7])
- <output_dir>/actions.pth                   (float32 tensor, [N, T_max, 7])
- <output_dir>/seq_lengths.pth               (int64 tensor, [N])
- <output_dir>/metadata.json

State definition (7D):
- [ee_pos_x, ee_pos_y, ee_pos_z, ee_euler_x, ee_euler_y, ee_euler_z, gripper_width_m]

Action definition (7D, base frame):
- [dpos_x, dpos_y, dpos_z, deuler_x, deuler_y, deuler_z, gripper_action]
  where dpos/deuler are derived from consecutive states, not copied from command action.
  In `delta` mode, gripper_action is delta gripper width in meters.
"""

import argparse
import json
from pathlib import Path

import numpy as np


torch = None
h5py = None
R = None


def ensure_dependencies():
    global torch, h5py, R
    try:
        import torch as _torch
    except ImportError as exc:
        raise SystemExit("torch is required.") from exc
    try:
        import h5py as _h5py
    except ImportError as exc:
        raise SystemExit("h5py is required.") from exc
    try:
        from scipy.spatial.transform import Rotation as _R
    except ImportError as exc:
        raise SystemExit("scipy is required.") from exc

    torch = _torch
    h5py = _h5py
    R = _R


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing raw LIBERO hdf5 files (e.g., .../datasets/Libero).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted tensors.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="agentview_rgb",
        choices=["agentview_rgb", "eye_in_hand_rgb"],
        help="Visual stream key under demo/obs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize output images to square size (HxW).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on converted episodes.",
    )
    parser.add_argument(
        "--gripper-mode",
        type=str,
        default="delta",
        choices=["delta", "binary", "command"],
        help=(
            "How to define action dim7. "
            "`delta`: gripper width delta; `binary`: sign of width delta; "
            "`command`: copy raw command action[:,6]."
        ),
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=1e-4,
        help="Threshold for binary gripper mode.",
    )
    parser.add_argument(
        "--pad-last-action",
        type=str,
        default="zero",
        choices=["zero", "repeat"],
        help="How to pad last action so actions length matches T.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Output directory is not empty. Use --overwrite or choose a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "obses").mkdir(parents=True, exist_ok=True)


def numeric_demo_sort_key(name: str):
    # demo_0, demo_1, ..., demo_49
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def parse_problem_info(attrs):
    raw = attrs.get("problem_info", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"raw_problem_info": raw}
    return {}


def compute_actions_from_state(ee_pos, ee_ori_rotvec, gripper_states, raw_actions, args):
    # Position delta in base frame.
    dpos = ee_pos[1:] - ee_pos[:-1]  # [T-1, 3]

    # Orientation delta in base frame:
    # R_next = R_delta * R_curr  =>  R_delta = R_next * R_curr^{-1}
    rot = R.from_rotvec(ee_ori_rotvec)
    drot = (rot[1:] * rot[:-1].inv()).as_euler("xyz", degrees=False)  # [T-1, 3]

    # Width in meters: distance between left and right finger positions.
    # LIBERO gripper states are typically signed mirror values.
    width = np.abs(gripper_states[:, 0] - gripper_states[:, 1])  # [T]

    if args.gripper_mode == "delta":
        g = width[1:] - width[:-1]  # [T-1]
    elif args.gripper_mode == "binary":
        dw = width[1:] - width[:-1]
        g = np.zeros_like(dw)
        g[dw > args.binary_threshold] = 1.0
        g[dw < -args.binary_threshold] = -1.0
    elif args.gripper_mode == "command":
        # raw command action space (useful for ablations).
        g = raw_actions[:-1, 6]
    else:
        raise ValueError("Unknown gripper mode: {}".format(args.gripper_mode))

    action = np.concatenate([dpos, drot, g[:, None]], axis=1).astype(np.float32)  # [T-1, 7]

    if args.pad_last_action == "zero":
        last = np.zeros((1, action.shape[1]), dtype=np.float32)
    else:
        last = action[-1:, :].copy()
    action = np.concatenate([action, last], axis=0)  # [T, 7]
    return action


def convert_demo(demo_group, camera_key, args):
    obs_group = demo_group["obs"]
    image = obs_group[camera_key][...]  # [T, H, W, C], uint8
    ee_pos = obs_group["ee_pos"][...]  # [T, 3]
    ee_ori = obs_group["ee_ori"][...]  # [T, 3], rotvec
    gripper_states = obs_group["gripper_states"][...]  # [T, 2]
    raw_actions = demo_group["actions"][...]  # [T, 7], command

    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("Unexpected image shape: {}".format(image.shape))
    if ee_pos.shape[0] < 2:
        raise ValueError("Too short demo, need at least 2 frames.")

    if args.image_size is not None:
        if image.shape[1] != args.image_size or image.shape[2] != args.image_size:
            image_t = torch.from_numpy(image).permute(0, 3, 1, 2).float()  # [T,C,H,W]
            image_t = torch.nn.functional.interpolate(
                image_t,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )
            image = image_t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

    euler = R.from_rotvec(ee_ori).as_euler("xyz", degrees=False).astype(np.float32)
    width = np.abs(gripper_states[:, 0] - gripper_states[:, 1]).astype(np.float32)[:, None]
    state = np.concatenate([ee_pos, euler, width], axis=1).astype(np.float32)  # [T,7]
    action = compute_actions_from_state(
        ee_pos=ee_pos,
        ee_ori_rotvec=ee_ori,
        gripper_states=gripper_states,
        raw_actions=raw_actions,
        args=args,
    )

    if state.shape[0] != action.shape[0] or state.shape[0] != image.shape[0]:
        raise ValueError(
            "Length mismatch image/state/action: {} {} {}".format(
                image.shape[0], state.shape[0], action.shape[0]
            )
        )
    return image, state, action


def main():
    args = parse_args()
    ensure_dependencies()
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    hdf5_files = sorted(args.input_root.rglob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError("No .hdf5 files found under {}".format(args.input_root))

    all_states = []
    all_actions = []
    seq_lengths = []
    converted = []
    skipped = []

    out_idx = 0
    for file_idx, hdf5_path in enumerate(hdf5_files):
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                skipped.append(
                    {
                        "source_hdf5": str(hdf5_path),
                        "demo_key": None,
                        "reason": "Missing /data group",
                    }
                )
                continue

            data_group = f["data"]
            problem_info = parse_problem_info(data_group.attrs)
            instruction = problem_info.get("language_instruction")
            env_name = data_group.attrs.get("env_name", "")
            env_args = data_group.attrs.get("env_args", "")

            demo_keys = sorted(
                [k for k in data_group.keys() if k.startswith("demo_")],
                key=numeric_demo_sort_key,
            )
            for demo_key in demo_keys:
                if args.max_episodes is not None and out_idx >= args.max_episodes:
                    break
                demo_group = data_group[demo_key]
                try:
                    image, state, action = convert_demo(demo_group, args.camera_key, args)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(
                        {
                            "source_hdf5": str(hdf5_path),
                            "demo_key": demo_key,
                            "reason": str(exc),
                        }
                    )
                    continue

                torch.save(
                    torch.from_numpy(image),
                    args.output_dir / "obses" / "episode_{:06d}.pth".format(out_idx),
                )
                all_states.append(torch.from_numpy(state))
                all_actions.append(torch.from_numpy(action))
                seq_lengths.append(int(state.shape[0]))
                converted.append(
                    {
                        "converted_index": out_idx,
                        "source_hdf5": str(hdf5_path),
                        "demo_key": demo_key,
                        "task": instruction,
                        "env_name": str(env_name),
                        "num_frames": int(state.shape[0]),
                        "camera_key": args.camera_key,
                    }
                )
                out_idx += 1

                if out_idx % 50 == 0:
                    print(
                        "converted={} skipped={} files={}/{}".format(
                            out_idx, len(skipped), file_idx + 1, len(hdf5_files)
                        ),
                        flush=True,
                    )

            if args.max_episodes is not None and out_idx >= args.max_episodes:
                break

    if not converted:
        raise RuntimeError("No episodes converted.")

    max_len = max(seq_lengths)
    n = len(converted)
    states_tensor = torch.zeros((n, max_len, 7), dtype=torch.float32)
    actions_tensor = torch.zeros((n, max_len, 7), dtype=torch.float32)
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)

    for i, (s, a) in enumerate(zip(all_states, all_actions)):
        t = s.shape[0]
        states_tensor[i, :t] = s
        actions_tensor[i, :t] = a

    torch.save(states_tensor, args.output_dir / "states.pth")
    torch.save(actions_tensor, args.output_dir / "actions.pth")
    torch.save(seq_lengths_tensor, args.output_dir / "seq_lengths.pth")

    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "camera_key": args.camera_key,
        "image_size": args.image_size,
        "gripper_mode": args.gripper_mode,
        "binary_threshold": args.binary_threshold,
        "pad_last_action": args.pad_last_action,
        "converted_episodes": len(converted),
        "skipped_episodes": len(skipped),
        "max_seq_len": int(max_len),
        "state_definition": "ee_pos(3)+ee_euler_xyz(3)+gripper_width_m(1)",
        "action_definition": "delta_ee_pos_base(3)+delta_ee_euler_xyz_base(3)+gripper_action(1)",
        "converted_index_map": converted,
        "skipped": skipped,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nConversion complete.")
    print("Converted episodes:", len(converted))
    print("Skipped episodes:", len(skipped))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
