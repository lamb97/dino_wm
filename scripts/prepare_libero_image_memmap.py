#!/usr/bin/env python3
"""Prepare mmap-friendly image files for LIBERO converted dataset.

Reads:
- <data_dir>/obses/episode_XXXXXX.pth (uint8, [T,H,W,C])
Writes:
- <data_dir>/obses_npy/episode_XXXXXX.npy (uint8, [T,H,W,C])
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    obses_dir = args.data_dir / "obses"
    out_dir = args.data_dir / "obses_npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    pths = sorted(obses_dir.glob("episode_*.pth"))
    if not pths:
        raise FileNotFoundError(f"No episode_*.pth found in {obses_dir}")

    for i, p in enumerate(pths, 1):
        out = out_dir / (p.stem + ".npy")
        if out.exists() and not args.overwrite:
            continue

        x = torch.load(p)
        if isinstance(x, torch.Tensor):
            arr = x.cpu().numpy()
        else:
            arr = np.asarray(x)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        np.save(out, arr)

        if i % 200 == 0:
            print(f"processed {i}/{len(pths)}")

    print("Done")
    print(f"input episodes: {len(pths)}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
