#!/usr/bin/env python3
"""Prepare mmap-friendly .npy files from converted LIBERO tensors.

Reads:
- states.pth
- actions.pth
- seq_lengths.pth

Writes:
- states.npy
- actions.npy
- seq_lengths.npy
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
    data_dir = args.data_dir

    states_pth = data_dir / "states.pth"
    actions_pth = data_dir / "actions.pth"
    seq_pth = data_dir / "seq_lengths.pth"

    if not (states_pth.exists() and actions_pth.exists() and seq_pth.exists()):
        raise FileNotFoundError("Expected states.pth/actions.pth/seq_lengths.pth in data dir")

    states_npy = data_dir / "states.npy"
    actions_npy = data_dir / "actions.npy"
    seq_npy = data_dir / "seq_lengths.npy"

    if not args.overwrite:
        for p in [states_npy, actions_npy, seq_npy]:
            if p.exists():
                raise FileExistsError(f"{p} already exists. Use --overwrite.")

    states = torch.load(states_pth).cpu().numpy().astype(np.float32, copy=False)
    actions = torch.load(actions_pth).cpu().numpy().astype(np.float32, copy=False)
    seq = torch.load(seq_pth).cpu().numpy().astype(np.int64, copy=False)

    np.save(states_npy, states)
    np.save(actions_npy, actions)
    np.save(seq_npy, seq)

    print("Saved:")
    print(states_npy)
    print(actions_npy)
    print(seq_npy)
    print("Shapes:")
    print("states", states.shape, states.dtype)
    print("actions", actions.shape, actions.dtype)
    print("seq", seq.shape, seq.dtype)


if __name__ == "__main__":
    main()
