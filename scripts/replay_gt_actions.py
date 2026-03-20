import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import gym
import hydra
import imageio
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from torchvision import utils as vutils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import env  # noqa: F401  # Register gym environments.

from preprocessor import Preprocessor
from utils import seed


def log_progress(msg):
    print(f"[replay_gt_actions] {msg}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay saved gt actions with current LIBERO action conversion and compare against the source trajectory."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Training output dir containing hydra.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--plan-targets",
        type=str,
        required=True,
        help="Path to plan_targets.pkl saved by plan.py.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional dataset override. Defaults to model hydra.yaml dataset path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Trajectory split to search. Planning uses valid by default.",
    )
    parser.add_argument(
        "--traj-id",
        type=int,
        default=None,
        help="Optional fixed trajectory id. If omitted, the script matches from gt_actions.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Optional starting offset within the trajectory. If omitted, the script matches from gt_actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Env seed for replay. Does not affect the matched gt segment.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="FPS for exported videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save replay outputs. Defaults to <plan_targets_dir>/gt_replay_compare.",
    )
    parser.add_argument(
        "--match-tol",
        type=float,
        default=1e-5,
        help="Tolerance for action matching diagnostics.",
    )
    return parser.parse_args()


def load_plan_targets(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    required = ["obs_0", "obs_g", "state_0", "state_g", "gt_actions", "goal_H"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"plan_targets is missing keys: {missing}")
    if data["gt_actions"] is None:
        raise ValueError("plan_targets.pkl does not contain gt_actions.")
    return data


def load_model_cfg(model_dir: Path, dataset_path_override: Optional[str]):
    model_cfg_path = model_dir / "hydra.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Cannot find hydra.yaml at {model_cfg_path}")
    model_cfg = OmegaConf.load(model_cfg_path)
    with open_dict(model_cfg):
        if dataset_path_override is not None:
            model_cfg.env.dataset.data_path = dataset_path_override
    return model_cfg


def load_traj_dataset(model_cfg, split: str):
    _, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    return traj_dsets[split]


def build_preprocessor(dset):
    return Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )


def find_matching_segment(dset, target_actions, target_state_0, target_state_g, frameskip, traj_id=None, offset=None):
    target_actions = torch.as_tensor(target_actions).float()
    if target_actions.ndim != 3 or target_actions.shape[0] != 1:
        raise ValueError(
            f"Expected gt_actions to have shape [1, H, F*D], got {tuple(target_actions.shape)}"
        )
    target_actions = target_actions[0]
    exec_steps = int(target_actions.shape[0] * frameskip)

    traj_indices = [int(traj_id)] if traj_id is not None else range(len(dset))
    best = None

    for idx in traj_indices:
        obs, act, state, env_info = dset[int(idx)]
        state = torch.as_tensor(state).float()
        act = torch.as_tensor(act).float()
        if act.shape[0] < exec_steps:
            continue

        candidate_offsets = [int(offset)] if offset is not None else range(act.shape[0] - exec_steps + 1)
        for off in candidate_offsets:
            chunk = act[off : off + exec_steps]
            wm_chunk = rearrange(chunk, "(t f) d -> t (f d)", f=frameskip)
            action_max_abs = torch.max(torch.abs(wm_chunk - target_actions)).item()
            state0_dist = torch.norm(state[off] - torch.as_tensor(target_state_0[0]).float()).item()
            if off + exec_steps < state.shape[0]:
                end_state = state[off + exec_steps]
                stateg_dist = torch.norm(
                    end_state - torch.as_tensor(target_state_g[0]).float()
                ).item()
            else:
                # Some converted datasets store T states for T actions instead of T+1.
                # Use the saved planning goal state for scoring in that case.
                end_state = torch.as_tensor(target_state_g[0]).float()
                stateg_dist = 0.0
            score = (action_max_abs, state0_dist, stateg_dist)
            record = {
                "traj_id": int(idx),
                "offset": int(off),
                "action_max_abs": float(action_max_abs),
                "state0_dist": float(state0_dist),
                "stateg_dist": float(stateg_dist),
                "matched_end_state_from_dataset": bool(off + exec_steps < state.shape[0]),
                "obs": obs,
                "act": act,
                "state": state,
                "env_info": env_info,
                "exec_steps": exec_steps,
            }
            if best is None or score < (
                best["action_max_abs"],
                best["state0_dist"],
                best["stateg_dist"],
            ):
                best = record
                if action_max_abs == 0.0 and state0_dist == 0.0 and stateg_dist == 0.0:
                    return best

    if best is None:
        raise RuntimeError("Could not find any trajectory segment long enough for the saved gt_actions.")
    return best


def make_env(model_cfg):
    return gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)


def prepare_gt_visuals(obs_visual, goal_visual, offset, exec_steps):
    gt = torch.as_tensor(obs_visual[offset : offset + exec_steps + 1]).float()
    if gt.shape[0] == exec_steps:
        goal = torch.as_tensor(goal_visual[0]).float().permute(2, 0, 1).unsqueeze(0)
        gt = torch.cat([gt, goal], dim=0)
    if gt.ndim != 4:
        raise ValueError(f"Expected visual trajectory [T, C, H, W], got {tuple(gt.shape)}")
    return gt


def transform_replay_visuals(preprocessor: Preprocessor, replay_visuals):
    replay_visuals = np.asarray(replay_visuals, dtype=np.uint8)
    replay_visuals = preprocessor.transform_obs_visual(replay_visuals)
    return replay_visuals.cpu()






def unwrap_env(env):
    cur = env
    visited = set()
    while hasattr(cur, "env") and id(cur) not in visited:
        visited.add(id(cur))
        cur = cur.env
    return cur

def convert_pose_cmds(env, low_level_actions):
    base_env = unwrap_env(env)
    cmds = []
    for action in np.asarray(low_level_actions, dtype=np.float32):
        cmd = np.asarray(base_env._action_to_libero(action), dtype=np.float32)
        cmds.append(cmd)
    return np.asarray(cmds, dtype=np.float32)


def summarize_position_diagnostics(low_level_actions, converted_cmds, replay_states, max_steps=10):
    replay_states = np.asarray(replay_states, dtype=np.float32)
    observed_dpos = replay_states[1:, :3] - replay_states[:-1, :3]
    src_dpos = np.asarray(low_level_actions, dtype=np.float32)[:, :3]
    cmd_dpos = np.asarray(converted_cmds, dtype=np.float32)[:, :3]
    steps = min(max_steps, src_dpos.shape[0], observed_dpos.shape[0], cmd_dpos.shape[0])

    rows = []
    for idx in range(steps):
        rows.append({
            "step": int(idx),
            "dataset_dpos": src_dpos[idx].tolist(),
            "converted_cmd_pos": cmd_dpos[idx].tolist(),
            "replay_observed_dpos": observed_dpos[idx].tolist(),
            "dataset_norm": float(np.linalg.norm(src_dpos[idx])),
            "observed_norm": float(np.linalg.norm(observed_dpos[idx])),
        })

    src_total = np.sum(src_dpos, axis=0)
    obs_total = np.sum(observed_dpos, axis=0)
    src_norm = float(np.linalg.norm(src_total))
    obs_norm = float(np.linalg.norm(obs_total))
    ratio = obs_norm / src_norm if src_norm > 1e-8 else float("nan")

    return {
        "first_steps": rows,
        "dataset_total_dpos": src_total.tolist(),
        "replay_total_observed_dpos": obs_total.tolist(),
        "dataset_total_norm": src_norm,
        "replay_total_norm": obs_norm,
        "replay_to_dataset_total_norm_ratio": ratio,
    }

def to_uint8_frame(frame_chw: torch.Tensor):
    frame = frame_chw.detach().cpu().float()
    if frame.ndim != 3:
        raise ValueError(f"Expected frame [C, H, W], got {tuple(frame.shape)}")
    if frame.min() < 0.0:
        frame = ((frame.clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0
    else:
        frame = frame.clamp(0.0, 1.0) * 255.0
    return frame.byte().permute(1, 2, 0).numpy()


def save_video(path: Path, frames, fps: int):
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_side_by_side_video(path: Path, left_frames, right_frames, fps: int):
    sep = 255 * np.ones((left_frames[0].shape[0], 8, 3), dtype=np.uint8)
    merged = []
    for left, right in zip(left_frames, right_frames):
        merged.append(np.concatenate([left, sep, right], axis=1))
    save_video(path, merged, fps=fps)


def save_compare_grid(path: Path, gt_visuals: torch.Tensor, replay_visuals: torch.Tensor, step_stride: int):
    gt = gt_visuals[::step_stride]
    replay = replay_visuals[::step_stride]
    grid = torch.cat([gt, replay], dim=0)
    vutils.save_image(
        grid,
        str(path),
        nrow=gt.shape[0],
        normalize=True,
        value_range=(-1, 1) if grid.min() < 0 else (0, 1),
    )


def main():
    args = parse_args()
    seed(args.seed)

    plan_targets_path = Path(args.plan_targets).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else plan_targets_path.parent / "gt_replay_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_progress(f"loading plan targets from {plan_targets_path}")
    plan_targets = load_plan_targets(plan_targets_path)
    log_progress(f"loading model config from {model_dir}")
    model_cfg = load_model_cfg(model_dir, args.dataset_path)
    log_progress(f"loading trajectory dataset split={args.split}")
    dset = load_traj_dataset(model_cfg, split=args.split)
    log_progress(f"dataset loaded: size={len(dset)}")
    preprocessor = build_preprocessor(dset)
    frameskip = int(model_cfg.frameskip)

    log_progress("matching saved gt_actions back to dataset trajectory")
    match = find_matching_segment(
        dset=dset,
        target_actions=plan_targets["gt_actions"],
        target_state_0=plan_targets["state_0"],
        target_state_g=plan_targets["state_g"],
        frameskip=frameskip,
        traj_id=args.traj_id,
        offset=args.offset,
    )

    log_progress(f"matched traj_id={match['traj_id']} offset={match['offset']} exec_steps={match['exec_steps']}")
    gt_visuals = prepare_gt_visuals(
        match["obs"]["visual"],
        plan_targets["obs_g"]["visual"],
        match["offset"],
        match["exec_steps"],
    )
    gt_states = match["state"][match["offset"] : match["offset"] + match["exec_steps"] + 1]

    log_progress("creating gym env")
    env = make_env(model_cfg)
    log_progress(f"env created: {type(env).__name__}")
    try:
        log_progress("calling env.update_env")
        env.update_env(match["env_info"])
        log_progress("env.update_env finished")
        log_progress("denormalizing matched low-level actions")
        low_level_actions = preprocessor.denormalize_actions(
            match["act"][match["offset"] : match["offset"] + match["exec_steps"]]
        ).numpy()
        converted_cmds = convert_pose_cmds(env, low_level_actions)
        log_progress(f"starting env.rollout with {match['exec_steps']} actions")
        replay_obs, replay_states = env.rollout(
            int(args.seed),
            np.asarray(plan_targets["state_0"][0], dtype=np.float32),
            low_level_actions,
        )
        log_progress("env.rollout finished")
    finally:
        env.close()

    log_progress("transforming replay visuals")
    replay_visuals = transform_replay_visuals(preprocessor, replay_obs["visual"])
    if replay_visuals.shape[0] != gt_visuals.shape[0]:
        raise RuntimeError(
            f"Replay length mismatch: gt has {gt_visuals.shape[0]} frames, replay has {replay_visuals.shape[0]}."
        )

    gt_frames = [to_uint8_frame(frame) for frame in gt_visuals]
    replay_frames = [to_uint8_frame(frame) for frame in replay_visuals]

    log_progress(f"writing outputs to {output_dir}")
    save_video(output_dir / "ground_truth.mp4", gt_frames, fps=args.fps)
    save_video(output_dir / "replay.mp4", replay_frames, fps=args.fps)
    save_side_by_side_video(output_dir / "compare_side_by_side.mp4", gt_frames, replay_frames, fps=args.fps)

    step_stride = max(frameskip, 1)
    save_compare_grid(output_dir / "compare_grid.png", gt_visuals, replay_visuals, step_stride=step_stride)

    replay_states = np.asarray(replay_states, dtype=np.float32)
    gt_states_np = gt_states.detach().cpu().numpy().astype(np.float32)
    if gt_states_np.shape[0] == replay_states.shape[0] - 1:
        gt_states_np = np.concatenate(
            [gt_states_np, np.asarray(plan_targets["state_g"], dtype=np.float32)],
            axis=0,
        )
    if gt_states_np.shape[0] != replay_states.shape[0]:
        raise RuntimeError(
            f"State length mismatch: gt has {gt_states_np.shape[0]} states, replay has {replay_states.shape[0]}."
        )
    state_l2 = np.linalg.norm(replay_states - gt_states_np, axis=1)
    position_diag = summarize_position_diagnostics(low_level_actions, converted_cmds, replay_states)
    for row in position_diag["first_steps"]:
        log_progress(
            "pos_diag step={step} dataset_dpos={dataset_dpos} cmd_pos={converted_cmd_pos} observed_dpos={replay_observed_dpos}".format(**row)
        )
    log_progress(
        "pos_diag totals dataset_total={} replay_total={} ratio={:.4f}".format(
            position_diag["dataset_total_dpos"],
            position_diag["replay_total_observed_dpos"],
            position_diag["replay_to_dataset_total_norm_ratio"],
        )
    )
    summary = {
        "plan_targets": str(plan_targets_path),
        "model_dir": str(model_dir),
        "dataset_path": str(model_cfg.env.dataset.data_path),
        "split": args.split,
        "matched_traj_id": int(match["traj_id"]),
        "matched_offset": int(match["offset"]),
        "goal_h": int(plan_targets["goal_H"]),
        "frameskip": frameskip,
        "exec_steps": int(match["exec_steps"]),
        "action_max_abs_diff": float(match["action_max_abs"]),
        "action_match_within_tol": bool(match["action_max_abs"] <= args.match_tol),
        "state0_dist": float(match["state0_dist"]),
        "stateg_dist": float(match["stateg_dist"]),
        "matched_end_state_from_dataset": bool(match["matched_end_state_from_dataset"]),
        "mean_state_l2_replay_vs_gt": float(state_l2.mean()),
        "final_state_l2_replay_vs_gt": float(state_l2[-1]),
        "position_diagnostics": position_diag,
        "outputs": {
            "ground_truth_video": str(output_dir / "ground_truth.mp4"),
            "replay_video": str(output_dir / "replay.mp4"),
            "side_by_side_video": str(output_dir / "compare_side_by_side.mp4"),
            "compare_grid": str(output_dir / "compare_grid.png"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_progress("done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
