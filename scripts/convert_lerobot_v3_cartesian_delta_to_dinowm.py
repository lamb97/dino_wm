#!/usr/bin/env python3
"""Convert a LeRobot v3 parquet/video dataset to DINO-WM Cartesian-delta format.

Output format:
- <output_dir>/states.npy                    float32 [N, T_max, 7]
- <output_dir>/actions.npy                   float32 [N, T_max, 7]
- <output_dir>/seq_lengths.npy               int64   [N]
- <output_dir>/states.pth
- <output_dir>/actions.pth
- <output_dir>/seq_lengths.pth
- <output_dir>/obses_npy/episode_XXXXXX.npy  uint8   [T, H, W, C]
- <output_dir>/episode_meta.jsonl
- <output_dir>/metadata.json

The state is eef_xyz + eef_rotvec + gripper_position from observation.state.
The action is observed delta xyz + observed delta rotvec + original gripper cmd.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


torch = None
imageio = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="LeRobot v3 dataset root containing data/, videos/, and meta/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for DINO-WM-readable files.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="observation.images.world.scene_1",
        help="Video feature key under videos/.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to image-size x image-size.",
    )
    parser.add_argument(
        "--skip-episodes",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Original episode_index values to skip.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on converted episodes after skipping.",
    )
    parser.add_argument(
        "--save-pth-images",
        action="store_true",
        help="Also write obses/episode_XXXXXX.pth images for non-mmap loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory contents if it already exists.",
    )
    return parser.parse_args()


def ensure_dependencies():
    global torch, imageio
    try:
        import torch as _torch
    except ImportError as exc:
        raise SystemExit("torch is required to write DINO-WM tensors.") from exc
    try:
        import imageio.v3 as _imageio
    except ImportError as exc:
        raise SystemExit("imageio is required to read LeRobot mp4 videos.") from exc

    torch = _torch
    imageio = _imageio


def ensure_clean_dir(path: Path, overwrite: bool, save_pth_images: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path / "obses_npy").mkdir(parents=True, exist_ok=True)
    if save_pth_images:
        (path / "obses").mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_text(input_dir: Path, task_index: int):
    tasks_path = input_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    table = pq.read_table(tasks_path)
    rows = table.to_pylist()
    for row in rows:
        if int(row.get("task_index", -1)) == int(task_index):
            return row.get("task")
    if rows:
        return rows[0].get("task")
    return None


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    return axis_angle_matrix(np.asarray(rotvec, dtype=np.float64) / theta, theta)


def matrix_to_rotvec(rotation):
    trace = np.trace(rotation)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - theta < 1e-6:
        diag = np.diag(rotation)
        axis = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0.0))
        if axis[0] > 1e-6:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif axis[1] > 1e-6:
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis = skew / (2.0 * math.sin(theta))
    return axis * theta


def cartesian_state_from_observation(observation_state):
    state = np.zeros((observation_state.shape[0], 7), dtype=np.float32)
    state[:, :6] = observation_state[:, 6:12]
    state[:, 6] = observation_state[:, 12]
    return state


def observed_delta_action(cartesian_state, raw_action):
    num_steps = cartesian_state.shape[0]
    action = np.zeros((num_steps, 7), dtype=np.float32)
    if num_steps > 1:
        action[:-1, :3] = cartesian_state[1:, :3] - cartesian_state[:-1, :3]
        rel_rotvecs = []
        for idx in range(num_steps - 1):
            cur_rot = rotvec_to_matrix(cartesian_state[idx, 3:6])
            next_rot = rotvec_to_matrix(cartesian_state[idx + 1, 3:6])
            rel_rotvecs.append(matrix_to_rotvec(cur_rot.T @ next_rot))
        action[:-1, 3:6] = np.asarray(rel_rotvecs, dtype=np.float32)

    action[:, 6] = raw_action[:, 6].astype(np.float32)
    return action


def resize_video(frames, image_size):
    if frames.shape[1] == image_size and frames.shape[2] == image_size:
        return frames.astype(np.uint8, copy=False)
    video = torch.from_numpy(np.asarray(frames, dtype=np.uint8))
    video = video.permute(0, 3, 1, 2).float()
    video = torch.nn.functional.interpolate(
        video,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    video = video.round().clamp_(0.0, 255.0).to(torch.uint8)
    return video.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def read_video(video_path: Path, expected_frames: int, image_size: int):
    resized_chunks = []
    chunk = []
    chunk_size = 64
    for frame in imageio.imiter(video_path):
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError(f"Expected video frame [H,W,C], got {frame.shape} from {video_path}")
        chunk.append(np.asarray(frame[..., :3], dtype=np.uint8))
        if len(chunk) >= chunk_size:
            resized_chunks.append(resize_video(np.stack(chunk, axis=0), image_size))
            chunk = []
    if chunk:
        resized_chunks.append(resize_video(np.stack(chunk, axis=0), image_size))

    if not resized_chunks:
        raise ValueError(f"No frames read from {video_path}")

    frames = np.concatenate(resized_chunks, axis=0)
    if frames.shape[0] != expected_frames:
        raise ValueError(
            f"Video/parquet length mismatch for {video_path}: "
            f"video={frames.shape[0]} parquet={expected_frames}"
        )
    return frames


def source_paths(input_dir: Path, image_key: str, episode_index: int):
    chunk = episode_index // 1000
    file_name = f"file-{episode_index:03d}"
    parquet_path = input_dir / "data" / f"chunk-{chunk:03d}" / f"{file_name}.parquet"
    video_path = (
        input_dir
        / "videos"
        / image_key
        / f"chunk-{chunk:03d}"
        / f"{file_name}.mp4"
    )
    return parquet_path, video_path


def convert_episode(input_dir: Path, image_key: str, episode_index: int, image_size: int):
    parquet_path, video_path = source_paths(input_dir, image_key, episode_index)
    table = pq.read_table(
        parquet_path,
        columns=["observation.state", "action", "episode_index", "frame_index", "task_index"],
    )
    observation_state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    raw_action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    episode_indices = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)

    if observation_state.ndim != 2 or observation_state.shape[1] < 13:
        raise ValueError(f"Expected observation.state [T, >=13], got {observation_state.shape}")
    if raw_action.ndim != 2 or raw_action.shape[1] < 7:
        raise ValueError(f"Expected action [T, >=7], got {raw_action.shape}")
    if not np.all(episode_indices == int(episode_index)):
        raise ValueError(f"{parquet_path} contains episode indices other than {episode_index}")
    if not np.array_equal(frame_indices, np.arange(frame_indices.shape[0])):
        raise ValueError(f"{parquet_path} frame_index is not contiguous from zero")

    state = cartesian_state_from_observation(observation_state)
    action = observed_delta_action(state, raw_action)
    images = read_video(video_path, expected_frames=state.shape[0], image_size=image_size)
    task_index = int(task_indices[0]) if task_indices.shape[0] else 0
    return images, state, action, parquet_path, video_path, task_index


def main():
    args = parse_args()
    ensure_dependencies()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ensure_clean_dir(output_dir, overwrite=args.overwrite, save_pth_images=args.save_pth_images)

    info = load_json(input_dir / "meta" / "info.json") or {}
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        parquet_files = sorted((input_dir / "data").glob("chunk-*/*.parquet"))
        episode_indices = [int(p.stem.split("-")[-1]) for p in parquet_files]
    else:
        episode_indices = list(range(total_episodes))

    skip_episodes = {int(x) for x in args.skip_episodes}
    episode_indices = [idx for idx in episode_indices if idx not in skip_episodes]
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    states_list = []
    actions_list = []
    seq_lengths = []
    episode_meta = []
    skipped = []

    for source_episode_index in episode_indices:
        try:
            images, state, action, parquet_path, video_path, task_index = convert_episode(
                input_dir=input_dir,
                image_key=args.image_key,
                episode_index=source_episode_index,
                image_size=args.image_size,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"episode_index": source_episode_index, "reason": str(exc)})
            print(f"skip episode {source_episode_index}: {exc}", flush=True)
            continue

        converted_index = len(states_list)
        np.save(output_dir / "obses_npy" / f"episode_{converted_index:06d}.npy", images)
        if args.save_pth_images:
            torch.save(
                torch.from_numpy(images),
                output_dir / "obses" / f"episode_{converted_index:06d}.pth",
            )

        states_list.append(state)
        actions_list.append(action)
        seq_lengths.append(int(state.shape[0]))
        episode_meta.append(
            {
                "converted_index": converted_index,
                "original_episode_index": int(source_episode_index),
                "task_index": task_index,
                "task": load_task_text(input_dir, task_index),
                "num_frames": int(state.shape[0]),
                "source_parquet": str(parquet_path),
                "source_video": str(video_path),
                "image_key": args.image_key,
            }
        )

        print(
            f"converted episode {source_episode_index} -> {converted_index} "
            f"frames={state.shape[0]}",
            flush=True,
        )

    if not states_list:
        raise RuntimeError("No episodes converted.")

    max_len = max(seq_lengths)
    num_episodes = len(states_list)
    states = np.zeros((num_episodes, max_len, 7), dtype=np.float32)
    actions = np.zeros((num_episodes, max_len, 7), dtype=np.float32)
    for idx, (state, action) in enumerate(zip(states_list, actions_list)):
        seq_len = state.shape[0]
        states[idx, :seq_len] = state
        actions[idx, :seq_len] = action

    seq_lengths_np = np.asarray(seq_lengths, dtype=np.int64)
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "seq_lengths.npy", seq_lengths_np)
    torch.save(torch.from_numpy(states), output_dir / "states.pth")
    torch.save(torch.from_numpy(actions), output_dir / "actions.pth")
    torch.save(torch.from_numpy(seq_lengths_np), output_dir / "seq_lengths.pth")

    with (output_dir / "episode_meta.jsonl").open("w", encoding="utf-8") as handle:
        for record in episode_meta:
            handle.write(json.dumps(record))
            handle.write("\n")

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_key": args.image_key,
        "image_size": int(args.image_size),
        "skip_episodes": sorted(skip_episodes),
        "converted_episodes": int(num_episodes),
        "skipped_episodes": skipped,
        "max_seq_length": int(max_len),
        "state_dim": 7,
        "action_dim": 7,
        "state_definition": "observation.state[6:9] eef_xyz + observation.state[9:12] eef_rotvec + observation.state[12] gripper_position",
        "action_definition": "observed delta xyz + observed relative delta rotvec + original action[6] gripper command",
        "source_info": info,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print("\nConversion complete.")
    print("Converted episodes:", num_episodes)
    print("Skipped episodes:", len(skipped))
    print("Max sequence length:", max_len)
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
