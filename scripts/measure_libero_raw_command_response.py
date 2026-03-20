import argparse
import json
import sys
from pathlib import Path

import numpy as np

from replay_gt_actions import (
    find_matching_segment,
    load_model_cfg,
    load_plan_targets,
    load_traj_dataset,
    make_env,
)
from utils import seed


def log(msg):
    print(f"[measure_raw_cmd] {msg}", file=sys.stderr, flush=True)


def parse_command(text: str):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 7:
        raise ValueError(f"Expected 7 comma-separated values, got {len(vals)} from: {text}")
    return np.asarray(vals, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure observed LIBERO motion from a raw 7D controller command."
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--plan-targets", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    parser.add_argument("--traj-id", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--command", type=str, default="1,0,0,0,0,0,0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def unwrap_env(env):
    cur = env
    visited = set()
    while hasattr(cur, "env") and id(cur) not in visited:
        visited.add(id(cur))
        cur = cur.env
    return cur


def main():
    args = parse_args()
    seed(args.seed)

    command = parse_command(args.command)
    plan_targets_path = Path(args.plan_targets).resolve()
    model_dir = Path(args.model_dir).resolve()

    log(f"loading plan targets from {plan_targets_path}")
    plan_targets = load_plan_targets(plan_targets_path)
    log(f"loading model config from {model_dir}")
    model_cfg = load_model_cfg(model_dir, args.dataset_path)
    log(f"loading dataset split={args.split}")
    dset = load_traj_dataset(model_cfg, split=args.split)
    log(f"dataset loaded: size={len(dset)}")

    log("matching trajectory segment for env initialization")
    match = find_matching_segment(
        dset=dset,
        target_actions=plan_targets["gt_actions"],
        target_state_0=plan_targets["state_0"],
        target_state_g=plan_targets["state_g"],
        frameskip=int(model_cfg.frameskip),
        traj_id=args.traj_id,
        offset=args.offset,
    )
    log(f"matched traj_id={match['traj_id']} offset={match['offset']}")

    env = make_env(model_cfg)
    try:
        base_env = unwrap_env(env)
        log("calling env.update_env")
        env.update_env(match["env_info"])
        log("preparing initial state")
        obs0, state0 = base_env.prepare(int(args.seed), np.asarray(plan_targets["state_0"][0], dtype=np.float32))

        states = [np.asarray(state0, dtype=np.float32)]
        visuals = [np.asarray(obs0["visual"], dtype=np.uint8)]
        rewards = []
        dones = []
        infos = []

        log(f"stepping raw command {command.tolist()} repeat={args.repeat}")
        for step_idx in range(int(args.repeat)):
            obs, reward, done, info = base_env._env.step(command.astype(np.float32))
            dino_obs = base_env._obs_to_dino(obs)
            state = np.asarray(dino_obs["proprio"], dtype=np.float32)
            states.append(state)
            visuals.append(np.asarray(dino_obs["visual"], dtype=np.uint8))
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(dict(info))
            log(f"step {step_idx}: observed_dpos={(states[-1][:3] - states[-2][:3]).tolist()}")

        states = np.asarray(states, dtype=np.float32)
        observed_deltas = states[1:, :3] - states[:-1, :3]
        cumulative = states[-1, :3] - states[0, :3]

        summary = {
            "plan_targets": str(plan_targets_path),
            "model_dir": str(model_dir),
            "dataset_path": str(model_cfg.env.dataset.data_path),
            "split": args.split,
            "matched_traj_id": int(match["traj_id"]),
            "matched_offset": int(match["offset"]),
            "seed": int(args.seed),
            "raw_command": command.tolist(),
            "repeat": int(args.repeat),
            "initial_ee_pos": states[0, :3].tolist(),
            "final_ee_pos": states[-1, :3].tolist(),
            "observed_dpos_per_step": observed_deltas.tolist(),
            "observed_dpos_norm_per_step": [float(np.linalg.norm(x)) for x in observed_deltas],
            "cumulative_observed_dpos": cumulative.tolist(),
            "cumulative_observed_norm": float(np.linalg.norm(cumulative)),
            "rewards": rewards,
            "dones": dones,
        }

        text = json.dumps(summary, indent=2)
        print(text)
        if args.output is not None:
            out_path = Path(args.output).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
            log(f"wrote summary to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
