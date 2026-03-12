#!/usr/bin/env python3
"""Convert LIBERO LeRobot parquet dataset into DINO-WM-friendly torch tensors.

Output format:
- <output_dir>/obses/episode_XXXXXX.pth      (uint8 tensor, shape [T, H, W, C])
- <output_dir>/states.pth                    (float32 tensor, shape [N, T_max, 8])
- <output_dir>/actions.pth                   (float32 tensor, shape [N, T_max, 7])
- <output_dir>/seq_lengths.pth               (int64 tensor, shape [N])
- <output_dir>/metadata.json                 (conversion metadata + index mapping)

Expected input layout (LeRobot):
- <input_dir>/meta/episodes.jsonl
- <input_dir>/data/chunk-XXX/episode_XXXXXX.parquet
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np


torch = None
pq = None
Image = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to LeRobot dataset root (contains meta/ and data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to converted output directory.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="image",
        choices=["image", "wrist_image"],
        help="Which visual stream to export.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for number of episodes to convert.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Episode chunk size used in data/chunk-XXX naming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory contents if it already exists.",
    )
    return parser.parse_args()


def ensure_dependencies():
    global torch, pq, Image
    try:
        import torch as _torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required. Install it in your runtime environment first."
        ) from exc
    try:
        import pyarrow.parquet as _pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. Install it first, e.g. `pip install pyarrow`."
        ) from exc
    try:
        from PIL import Image as _Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required. Install it first, e.g. `pip install pillow`."
        ) from exc

    torch = _torch
    pq = _pq
    Image = _Image


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def decode_image_bytes(image_struct):
    # LeRobot image field is a struct {bytes, path}.
    image_bytes = image_struct.get("bytes") if isinstance(image_struct, dict) else None
    if image_bytes is None:
        raise ValueError("Missing image bytes in parquet row.")
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(pil_img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Decoded image is not HxWx3 RGB.")
    return arr


def ensure_output_dir(output_dir, overwrite=False):
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Output directory is not empty. Use --overwrite or choose a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "obses").mkdir(parents=True, exist_ok=True)


def to_tensor_2d(pylist, expected_dim, dtype=np.float32):
    arr = np.asarray(pylist, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(
            "Expected 2D array with second dim {}, got {}".format(
                expected_dim, tuple(arr.shape)
            )
        )
    return torch.from_numpy(arr)


def convert_episode(parquet_path, image_key):
    table = pq.read_table(str(parquet_path), columns=[image_key, "state", "actions"])

    image_col = table.column(image_key).to_pylist()
    state_col = table.column("state").to_pylist()
    action_col = table.column("actions").to_pylist()

    if not (len(image_col) == len(state_col) == len(action_col)):
        raise ValueError("Mismatched lengths among image/state/actions columns.")

    frames = [decode_image_bytes(x) for x in image_col]
    obs = torch.from_numpy(np.stack(frames, axis=0))  # [T, H, W, C], uint8
    state = to_tensor_2d(state_col, expected_dim=8)
    actions = to_tensor_2d(action_col, expected_dim=7)

    if state.shape[0] != obs.shape[0] or actions.shape[0] != obs.shape[0]:
        raise ValueError("Temporal length mismatch after decode.")

    return obs, state, actions


def build_parquet_path(input_dir, episode_index, chunk_size):
    chunk = episode_index // chunk_size
    return (
        input_dir
        / "data"
        / "chunk-{:03d}".format(chunk)
        / "episode_{:06d}.parquet".format(episode_index)
    )


def main():
    args = parse_args()
    ensure_dependencies()

    episodes_path = args.input_dir / "meta" / "episodes.jsonl"
    info_path = args.input_dir / "meta" / "info.json"

    if not episodes_path.exists():
        raise FileNotFoundError("Missing episodes file: {}".format(episodes_path))

    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    episodes = read_jsonl(episodes_path)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    dataset_info = {}
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    all_states = []
    all_actions = []
    seq_lengths = []
    converted = []
    skipped = []

    for i, episode in enumerate(episodes):
        ep_idx = int(episode["episode_index"])
        parquet_path = build_parquet_path(args.input_dir, ep_idx, args.chunk_size)

        try:
            obs, state, actions = convert_episode(parquet_path, image_key=args.image_key)
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "episode_index": ep_idx,
                    "parquet": str(parquet_path),
                    "reason": str(exc),
                }
            )
            continue

        out_idx = len(converted)
        obs_path = args.output_dir / "obses" / "episode_{:06d}.pth".format(out_idx)
        torch.save(obs, obs_path)

        all_states.append(state)
        all_actions.append(actions)
        seq_lengths.append(state.shape[0])
        converted.append(
            {
                "converted_index": out_idx,
                "original_episode_index": ep_idx,
                "task": episode.get("tasks", [None])[0],
                "length": int(state.shape[0]),
                "source_parquet": str(parquet_path),
            }
        )

        if (i + 1) % 50 == 0:
            print(
                "processed {:5d}/{:5d} episodes | converted={} skipped={}".format(
                    i + 1, len(episodes), len(converted), len(skipped)
                ),
                flush=True,
            )

    if not converted:
        raise RuntimeError("No valid episodes converted.")

    max_len = max(seq_lengths)
    n = len(converted)
    states_tensor = torch.zeros((n, max_len, 8), dtype=torch.float32)
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
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "image_key": args.image_key,
        "chunk_size": args.chunk_size,
        "requested_episodes": len(episodes),
        "converted_episodes": len(converted),
        "skipped_episodes": len(skipped),
        "max_seq_length": int(max_len),
        "state_dim": 8,
        "action_dim": 7,
        "source_info": dataset_info,
        "converted_index_map": converted,
        "skipped": skipped,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nConversion complete.")
    print("Converted episodes:", len(converted))
    print("Skipped episodes:", len(skipped))
    print("Output:", args.output_dir)
    if skipped:
        print("Example skipped reason:", skipped[0]["reason"])


if __name__ == "__main__":
    main()
