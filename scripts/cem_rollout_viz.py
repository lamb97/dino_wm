import argparse
import os
import sys
from pathlib import Path

import hydra
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import utils as vutils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning.cem import CEMPlanner
from planning.objectives import create_objective_fn
from utils import seed, move_to_device


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


class DummyWandbRun:
    def log(self, *args, **kwargs):
        pass


class IdentityPreprocessor:
    """Use already-normalized tensors from traj datasets directly."""

    def transform_obs(self, obs):
        out = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.float()
            else:
                out[key] = torch.tensor(value, dtype=torch.float32)
        return out


def load_ckpt(snapshot_path: Path, device: torch.device):
    payload = torch.load(snapshot_path, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload.get("epoch", -1)
    return result


def load_model(model_dir: Path, model_epoch: str, device: torch.device):
    model_cfg_path = model_dir / "hydra.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Cannot find hydra.yaml at {model_cfg_path}")
    model_cfg = OmegaConf.load(model_cfg_path)

    ckpt_name = "model_latest.pth" if model_epoch == "latest" else f"model_{model_epoch}.pth"
    model_ckpt = model_dir / "checkpoints" / ckpt_name
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {model_ckpt}")
    result = load_ckpt(model_ckpt, device=device)
    print(f"Loaded checkpoint from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(model_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint.")

    if model_cfg.has_decoder and "decoder" not in result:
        if model_cfg.env.decoder_path is None:
            raise ValueError("Decoder missing in checkpoint and env.decoder_path is not set.")
        decoder_path = Path(__file__).resolve().parents[1] / model_cfg.env.decoder_path
        ckpt = torch.load(decoder_path, map_location=device)
        result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
    elif not model_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        model_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=model_cfg.proprio_emb_dim,
        action_dim=model_cfg.action_emb_dim,
        concat_dim=model_cfg.concat_dim,
        num_action_repeat=model_cfg.num_action_repeat,
        num_proprio_repeat=model_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model_cfg, model, result["epoch"]


def sample_segments(
    dset,
    num_rollout,
    goal_h,
    frameskip,
    n_past,
    rng: np.random.RandomState,
):
    obs0_visuals = []
    obs0_proprios = []
    obsg_visuals = []
    obsg_proprios = []
    gt_visuals = []
    gt_proprios = []
    meta = []

    needed = goal_h * frameskip + 1
    for idx in range(num_rollout):
        valid = False
        while not valid:
            traj_idx = int(rng.randint(0, len(dset)))
            obs, act, state, _ = dset[traj_idx]
            t = obs["visual"].shape[0]
            if t < needed:
                continue
            start = int(rng.randint(0, t - needed + 1))
            valid = True

        sub = slice(start, start + needed)
        obs_visual_full = obs["visual"][sub][::frameskip]       # [H+1, C, H, W], normalized
        obs_proprio_full = obs["proprio"][sub][::frameskip]     # [H+1, D], normalized

        obs0_visuals.append(obs_visual_full[:n_past])
        obs0_proprios.append(obs_proprio_full[:n_past])
        obsg_visuals.append(obs_visual_full[-1:].clone())
        obsg_proprios.append(obs_proprio_full[-1:].clone())
        gt_visuals.append(obs_visual_full)
        gt_proprios.append(obs_proprio_full)
        meta.append({"traj_idx": traj_idx, "start": start})

    obs_0 = {
        "visual": torch.stack(obs0_visuals, dim=0),
        "proprio": torch.stack(obs0_proprios, dim=0),
    }
    obs_g = {
        "visual": torch.stack(obsg_visuals, dim=0),
        "proprio": torch.stack(obsg_proprios, dim=0),
    }
    gt_obs = {
        "visual": torch.stack(gt_visuals, dim=0),
        "proprio": torch.stack(gt_proprios, dim=0),
    }
    return obs_0, obs_g, gt_obs, meta


def plan_open_loop_rollout(planner, model, obs_0, obs_g):
    actions, _ = planner.plan(obs_0=obs_0, obs_g=obs_g, actions=None)
    trans_obs_0 = move_to_device(planner.preprocessor.transform_obs(obs_0), planner.device)
    z_obses, _ = model.rollout(obs_0=trans_obs_0, act=actions)
    imagined_visuals = model.decode_obs(z_obses)[0]["visual"].cpu()
    return actions.cpu(), imagined_visuals


def plan_segmented_rollout(planner, model, gt_obs, goal_h, n_past, subgoal_h):
    if subgoal_h <= 0:
        raise ValueError(f"subgoal_h must be positive, got {subgoal_h}.")

    b = gt_obs["visual"].shape[0]
    imagined_visuals = gt_obs["visual"].new_zeros(gt_obs["visual"].shape)
    actions = gt_obs["visual"].new_zeros((b, goal_h, planner.action_dim))
    segment_meta = []
    current_obs_idx = n_past - 1
    original_horizon = planner.horizon

    try:
        while current_obs_idx < goal_h:
            exec_h = min(subgoal_h, goal_h - current_obs_idx)
            history_start = current_obs_idx - n_past + 1
            segment_goal_idx = current_obs_idx + exec_h
            planner.horizon = segment_goal_idx - history_start

            obs_0 = {
                key: value[:, history_start : current_obs_idx + 1].clone()
                for key, value in gt_obs.items()
            }
            obs_g = {
                key: value[:, segment_goal_idx : segment_goal_idx + 1].clone()
                for key, value in gt_obs.items()
            }

            local_actions, local_visuals = plan_open_loop_rollout(
                planner=planner,
                model=model,
                obs_0=obs_0,
                obs_g=obs_g,
            )
            imagined_visuals[:, history_start : segment_goal_idx + 1] = local_visuals
            if history_start == 0:
                actions[:, : planner.horizon] = local_actions
            else:
                actions[:, current_obs_idx:segment_goal_idx] = local_actions[:, n_past - 1 :]

            segment_meta.append(
                {
                    "history_start_idx": int(history_start),
                    "current_obs_idx": int(current_obs_idx),
                    "segment_goal_idx": int(segment_goal_idx),
                    "planner_horizon": int(planner.horizon),
                    "exec_h": int(exec_h),
                    "obs_refresh_source": "ground_truth",
                }
            )
            current_obs_idx = segment_goal_idx
    finally:
        planner.horizon = original_horizon

    return actions, imagined_visuals, segment_meta


def save_rollout_plot(gt_visuals, imagined_visuals, out_png):
    """
    gt_visuals: [B, T, C, H, W], normalized to [-1, 1]
    imagined_visuals: [B, T, C, H, W], normalized to [-1, 1]
    """
    b, t = gt_visuals.shape[:2]
    grid = torch.cat([gt_visuals, imagined_visuals], dim=1)  # [B, 2T, C, H, W]
    grid = rearrange(grid, "b t c h w -> (b t) c h w")
    n_columns = t
    vutils.save_image(
        grid,
        out_png,
        nrow=n_columns,
        normalize=True,
        value_range=(-1, 1),
    )


def save_rollout_plot_per_traj(gt_visuals, imagined_visuals, out_dir, prefix):
    """
    Save one PNG per trajectory.
    Each image has 2 rows: GT (top) and imagined (bottom), T columns.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b, t = gt_visuals.shape[:2]
    for i in range(b):
        traj_grid = torch.cat([gt_visuals[i], imagined_visuals[i]], dim=0)  # [2T, C, H, W]
        out_png = out_dir / f"{prefix}_traj{i:03d}.png"
        vutils.save_image(
            traj_grid,
            str(out_png),
            nrow=t,
            normalize=True,
            value_range=(-1, 1),
        )


def main():
    parser = argparse.ArgumentParser(
        description="CEM-only rollout visualization (GT vs imagined) without simulator."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Training output dir containing hydra.yaml and checkpoints/",
    )
    parser.add_argument(
        "--model-epoch",
        type=str,
        default="latest",
        help="Checkpoint epoch to load. Use 'latest' or an integer string.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid"],
    )
    parser.add_argument("--num-rollout", type=int, default=10)
    parser.add_argument(
        "--plan-cfg",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "conf" / "plan.yaml"),
    )
    parser.add_argument(
        "--planner-cfg",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "conf" / "planner" / "cem.yaml"),
    )
    parser.add_argument(
        "--goal-h",
        type=int,
        default=None,
        help="Planning horizon / goal distance (in downsampled steps). Defaults to plan.yaml goal_H.",
    )
    parser.add_argument(
        "--subgoal-h",
        type=int,
        default=None,
        help=(
            "If set smaller than goal-h, replan every subgoal-h downsampled execution steps. "
            "After each chunk, the current observation is refreshed from GT."
        ),
    )
    parser.add_argument(
        "--n-past",
        type=int,
        default=None,
        help="History length for obs_0. Defaults to model_cfg.num_hist.",
    )
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save png and action tensors. Defaults to <model-dir>/cem_rollout_plots.",
    )
    args = parser.parse_args()

    seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir).resolve()
    model_cfg, model, loaded_epoch = load_model(model_dir, args.model_epoch, device)

    _, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets[args.split]

    plan_cfg = OmegaConf.load(args.plan_cfg)
    planner_cfg = OmegaConf.load(args.planner_cfg)
    goal_h = int(args.goal_h) if args.goal_h is not None else int(plan_cfg.goal_H)
    subgoal_h = int(args.subgoal_h) if args.subgoal_h is not None else None
    n_past = int(args.n_past) if args.n_past is not None else int(model_cfg.num_hist)
    use_segmented_replan = subgoal_h is not None and subgoal_h < goal_h

    if n_past > goal_h:
        raise ValueError(f"n_past ({n_past}) must be <= goal_h ({goal_h}).")
    if subgoal_h is not None and subgoal_h <= 0:
        raise ValueError(f"subgoal_h must be positive, got {subgoal_h}.")

    objective_fn = create_objective_fn(
        alpha=float(plan_cfg.objective.alpha),
        base=int(plan_cfg.objective.base),
        mode=str(plan_cfg.objective.mode),
    )

    action_dim = int(dset.action_dim * model_cfg.frameskip)
    planner_cfg = OmegaConf.to_container(planner_cfg, resolve=True)
    planner_cfg["horizon"] = goal_h

    planner = CEMPlanner(
        **planner_cfg,
        wm=model,
        action_dim=action_dim,
        objective_fn=objective_fn,
        preprocessor=IdentityPreprocessor(),
        evaluator=None,
        wandb_run=DummyWandbRun(),
        logging_prefix="cem_rollout",
        log_filename=None,
    )

    obs_0, obs_g, gt_obs, meta = sample_segments(
        dset=dset,
        num_rollout=args.num_rollout,
        goal_h=goal_h,
        frameskip=model_cfg.frameskip,
        n_past=n_past,
        rng=rng,
    )
    gt_visuals = gt_obs["visual"]

    with torch.no_grad():
        if use_segmented_replan:
            actions, imagined_visuals, replan_segments = plan_segmented_rollout(
                planner=planner,
                model=model,
                gt_obs=gt_obs,
                goal_h=goal_h,
                n_past=n_past,
                subgoal_h=subgoal_h,
            )
        else:
            planner.horizon = goal_h
            actions, imagined_visuals = plan_open_loop_rollout(
                planner=planner,
                model=model,
                obs_0=obs_0,
                obs_g=obs_g,
            )
            replan_segments = []

    out_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else model_dir / "cem_rollout_plots"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    epoch_tag = str(loaded_epoch) if loaded_epoch is not None else "unknown"
    mode_tag = f"_sg{subgoal_h}" if use_segmented_replan else ""
    out_png = out_root / f"cem_rollout_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.png"
    per_traj_dir = out_root / f"cem_rollout_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}_per_traj"
    out_actions = out_root / f"cem_actions_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.pt"
    out_meta = out_root / f"cem_meta_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.pt"

    save_rollout_plot(gt_visuals, imagined_visuals, str(out_png))
    save_rollout_plot_per_traj(
        gt_visuals=gt_visuals,
        imagined_visuals=imagined_visuals,
        out_dir=per_traj_dir,
        prefix=f"cem_rollout_e{epoch_tag}_{args.split}{mode_tag}",
    )
    torch.save(actions.cpu(), out_actions)
    torch.save(
        {
            "meta": meta,
            "goal_h": goal_h,
            "subgoal_h": subgoal_h,
            "n_past": n_past,
            "frameskip": int(model_cfg.frameskip),
            "model_epoch": loaded_epoch,
            "split": args.split,
            "use_segmented_replan": use_segmented_replan,
            "replan_obs_source": "ground_truth" if use_segmented_replan else None,
            "replan_segments": replan_segments,
        },
        out_meta,
    )

    print(f"Saved rollout plot: {out_png}")
    print(f"Saved per-trajectory plots under: {per_traj_dir}")
    print(f"Saved optimized actions: {out_actions}")
    print(f"Saved sampling metadata: {out_meta}")


if __name__ == "__main__":
    main()
