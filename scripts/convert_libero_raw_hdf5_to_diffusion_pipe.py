#!/usr/bin/env python3
"""Convert raw LIBERO HDF5 demos to a diffusion-pipe video dataset.

Output format:
- <output_dir>/<base_name>.mp4
- <output_dir>/<base_name>.txt
- <output_dir>/metadata.json

Each `.txt` contains the task language instruction for the corresponding demo.
The `.mp4` contains the selected camera stream for that demo.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


h5py = None
imageio = None
Image = None


def ensure_dependencies():
    global h5py, imageio, Image

    try:
        import h5py as _h5py
    except ImportError as exc:
        raise SystemExit("h5py is required. Try `pip install h5py`.") from exc

    try:
        import imageio.v2 as _imageio
    except ImportError as exc:
        raise SystemExit("imageio is required. Try `pip install imageio imageio-ffmpeg`.") from exc

    try:
        from PIL import Image as _Image
    except ImportError as exc:
        raise SystemExit("Pillow is required. Try `pip install Pillow`.") from exc

    h5py = _h5py
    imageio = _imageio
    Image = _Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing raw LIBERO .hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write diffusion-pipe media files into.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="agentview_rgb",
        choices=["agentview_rgb", "eye_in_hand_rgb"],
        help="Image stream stored under demo/obs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="FPS to use when encoding mp4 files.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square resize for exported frames.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame from each trajectory.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames per exported video after frame_stride.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on total exported demos.",
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
            f"Output directory {output_dir} is not empty. Use --overwrite or choose a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def numeric_demo_sort_key(name: str):
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


def normalize_instruction(instruction):
    if instruction is None:
        return None
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8", errors="ignore")
    if isinstance(instruction, (list, tuple)):
        instruction = "".join(
            item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)
            for item in instruction
        )
    instruction = str(instruction).strip()
    return instruction.strip('"').strip("'")


def instruction_from_filename(path: Path):
    stem = path.stem
    stem = re.sub(r"_demo$", "", stem)
    parts = stem.split("_")
    task_tokens = []
    for token in parts:
        if token.startswith("SCENE") or token == "demo":
            continue
        if token.isupper() and "SCENE" not in token:
            continue
        task_tokens.append(token)
    if not task_tokens:
        return stem.replace("_", " ")
    return " ".join(task_tokens)


def resize_frames(images, image_size):
    if image_size is None:
        return images
    if images.shape[1] == image_size and images.shape[2] == image_size:
        return images

    resized = []
    for frame in images:
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        resized.append(np.asarray(pil_image, dtype=np.uint8))
    return np.stack(resized, axis=0)


def load_demo_images(demo_group, camera_key, args):
    obs_group = demo_group["obs"]
    if camera_key not in obs_group:
        raise KeyError(f"Missing camera key {camera_key!r} under demo/obs.")

    images = obs_group[camera_key][...]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    images = images[:: args.frame_stride]
    if args.max_frames is not None:
        images = images[: args.max_frames]
    images = resize_frames(images, args.image_size)

    if images.shape[0] < 2:
        raise ValueError(f"Too short after filtering, got {images.shape[0]} frames.")

    return images


def write_video(video_path: Path, images, fps: int):
    writer = imageio.get_writer(str(video_path), fps=fps)
    try:
        for frame in images:
            writer.append_data(frame)
    finally:
        writer.close()


def print_progress(file_idx, total_files, hdf5_path, demo_idx, total_demos, converted_count, skipped_count):
    percent = 100.0 * (file_idx + 1) / max(total_files, 1)
    message = (
        f"\r[{file_idx + 1}/{total_files} files | {percent:5.1f}%] "
        f"{hdf5_path.name} demo {demo_idx}/{total_demos} "
        f"converted={converted_count} skipped={skipped_count}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def main():
    args = parse_args()
    ensure_dependencies()
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2 when provided")

    hdf5_files = sorted(args.input_root.rglob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_root}")

    print(f"found {len(hdf5_files)} hdf5 files under {args.input_root}")

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
            instruction = normalize_instruction(problem_info.get("language_instruction"))
            if not instruction:
                instruction = instruction_from_filename(hdf5_path)

            demo_keys = sorted(
                [key for key in data_group.keys() if key.startswith("demo_")],
                key=numeric_demo_sort_key,
            )
            total_demos = len(demo_keys)

            for demo_pos, demo_key in enumerate(demo_keys, start=1):
                if args.max_episodes is not None and out_idx >= args.max_episodes:
                    break

                demo_group = data_group[demo_key]
                try:
                    images = load_demo_images(demo_group, args.camera_key, args)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(
                        {
                            "source_hdf5": str(hdf5_path),
                            "demo_key": demo_key,
                            "reason": str(exc),
                        }
                    )
                    print_progress(
                        file_idx,
                        len(hdf5_files),
                        hdf5_path,
                        demo_pos,
                        total_demos,
                        out_idx,
                        len(skipped),
                    )
                    continue

                base_name = f"episode_{out_idx:06d}"
                video_path = args.output_dir / f"{base_name}.mp4"
                caption_path = args.output_dir / f"{base_name}.txt"

                write_video(video_path, images, fps=args.fps)
                caption_path.write_text(instruction + "\n", encoding="utf-8")

                converted.append(
                    {
                        "id": out_idx,
                        "source_hdf5": str(hdf5_path),
                        "demo_key": demo_key,
                        "task": instruction,
                        "camera_key": args.camera_key,
                        "num_frames": int(images.shape[0]),
                        "video_path": str(video_path),
                        "caption_path": str(caption_path),
                    }
                )
                out_idx += 1
                print_progress(
                    file_idx,
                    len(hdf5_files),
                    hdf5_path,
                    demo_pos,
                    total_demos,
                    out_idx,
                    len(skipped),
                )

        if total_demos > 0:
            print()

        if args.max_episodes is not None and out_idx >= args.max_episodes:
            break

    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "camera_key": args.camera_key,
        "fps": args.fps,
        "image_size": args.image_size,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "converted_count": len(converted),
        "skipped_count": len(skipped),
        "converted": converted,
        "skipped": skipped,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"done: converted={len(converted)} skipped={len(skipped)} output={args.output_dir}")


if __name__ == "__main__":
    main()
