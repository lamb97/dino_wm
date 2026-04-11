import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import utils as vutils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning.cem import CEMPlanner
from planning.objectives import create_objective_fn
from utils import move_to_device, seed


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


class SVDVideoClient:
    def __init__(self, server_url: str, timeout_sec: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def healthcheck(self) -> dict:
        with urllib.request.urlopen(
            f"{self.server_url}/health", timeout=self.timeout_sec
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(
        self,
        image_tensor: torch.Tensor,
        *,
        num_frames: int,
        width: int,
        height: int,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        decode_chunk_size: int,
        num_inference_steps: int,
        seed_value: Optional[int],
    ):
        image = visual_tensor_to_pil(image_tensor)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = {
            "image_png_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "num_frames": int(num_frames),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "motion_bucket_id": int(motion_bucket_id),
            "noise_aug_strength": float(noise_aug_strength),
            "decode_chunk_size": int(decode_chunk_size),
            "num_inference_steps": int(num_inference_steps),
            "seed": None if seed_value is None else int(seed_value),
        }
        request = urllib.request.Request(
            f"{self.server_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SVD server returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach SVD server at {self.server_url}: {exc}") from exc

        frames = [
            png_base64_to_visual_tensor(frame_b64)
            for frame_b64 in result["frames_png_base64"]
        ]
        return torch.stack(frames, dim=0), result


def visual_tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor of shape [C, H, W], got {tuple(image.shape)}")

    img_min = float(image.min().item())
    img_max = float(image.max().item())
    if img_min >= 0.0 and img_max <= 1.0:
        image_01 = image.clamp(0.0, 1.0)
    else:
        image_01 = ((image.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)

    image_uint8 = image_01.mul(255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image_uint8)


def png_base64_to_visual_tensor(frame_b64: str) -> torch.Tensor:
    image = Image.open(io.BytesIO(base64.b64decode(frame_b64))).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    return image_tensor * 2.0 - 1.0


def select_generated_frames(
    generated_frames: torch.Tensor,
    needed_future: int,
    strategy: str,
) -> torch.Tensor:
    available = int(generated_frames.shape[0])
    if available < needed_future:
        raise ValueError(
            f"SVD returned only {available} frames, but planning needs {needed_future}."
        )
    if available == needed_future:
        return generated_frames
    if strategy == "first":
        return generated_frames[:needed_future]
    if strategy == "linspace":
        indices = np.linspace(0, available - 1, needed_future, dtype=np.int64)
        return generated_frames[torch.from_numpy(indices)]
    raise ValueError(f"Unknown frame selection strategy: {strategy}")


def resize_visual_sequence(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] == (height, width):
        return frames
    return F.interpolate(
        frames,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


def rotate_visual_sequence_180(frames: torch.Tensor) -> torch.Tensor:
    return torch.rot90(frames, k=2, dims=(-2, -1))


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
    for _ in range(num_rollout):
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
        obs_visual_full = obs["visual"][sub][::frameskip]
        obs_proprio_full = obs["proprio"][sub][::frameskip]

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


def get_goal_proprio(gt_obs, current_obs_idx: int, goal_idx: int, source: str) -> torch.Tensor:
    if source == "gt":
        return gt_obs["proprio"][:, goal_idx : goal_idx + 1].clone()
    if source == "current":
        return gt_obs["proprio"][:, current_obs_idx : current_obs_idx + 1].clone()
    raise ValueError(f"Unknown goal proprio source: {source}")


def build_svd_goal_visuals(
    client: SVDVideoClient,
    obs_0,
    gt_obs,
    goal_h: int,
    n_past: int,
    svd_width: Optional[int],
    svd_height: Optional[int],
    request_num_frames: Optional[int],
    frame_selection: str,
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    decode_chunk_size: int,
    num_inference_steps: int,
    seed_value: Optional[int],
):
    current_obs_idx = n_past - 1
    needed_future = goal_h - current_obs_idx
    if needed_future <= 0:
        raise ValueError(
            f"goal_h ({goal_h}) must be greater than current_obs_idx ({current_obs_idx})."
        )

    request_frames = int(request_num_frames) if request_num_frames is not None else int(needed_future)
    _, _, _, target_height, target_width = gt_obs["visual"].shape
    request_width = int(svd_width) if svd_width is not None else int(target_width)
    request_height = int(svd_height) if svd_height is not None else int(target_height)
    svd_goal_visuals = gt_obs["visual"].clone()
    svd_meta = []

    for traj_idx in range(obs_0["visual"].shape[0]):
        conditioning_image = obs_0["visual"][traj_idx, -1]
        traj_seed = None if seed_value is None else int(seed_value + traj_idx)
        generated_frames, response_meta = client.generate(
            conditioning_image,
            num_frames=request_frames,
            width=request_width,
            height=request_height,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            num_inference_steps=num_inference_steps,
            seed_value=traj_seed,
        )
        selected_frames = select_generated_frames(
            generated_frames=generated_frames,
            needed_future=needed_future,
            strategy=frame_selection,
        )
        selected_frames = resize_visual_sequence(
            selected_frames,
            height=target_height,
            width=target_width,
        )
        svd_goal_visuals[traj_idx, current_obs_idx + 1 : goal_h + 1] = selected_frames
        response_meta["traj_idx"] = int(traj_idx)
        response_meta["seed"] = traj_seed
        response_meta["requested_num_frames"] = request_frames
        response_meta["used_num_future_frames"] = needed_future
        response_meta["requested_width"] = request_width
        response_meta["requested_height"] = request_height
        response_meta["returned_to_target_width"] = target_width
        response_meta["returned_to_target_height"] = target_height
        svd_meta.append(response_meta)

    return svd_goal_visuals, svd_meta


def plan_segmented_rollout_video(
    planner,
    model,
    gt_obs,
    svd_goal_visuals,
    goal_h,
    n_past,
    subgoal_h,
    goal_proprio_source,
):
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
                "visual": svd_goal_visuals[:, segment_goal_idx : segment_goal_idx + 1].clone(),
                "proprio": get_goal_proprio(
                    gt_obs=gt_obs,
                    current_obs_idx=current_obs_idx,
                    goal_idx=segment_goal_idx,
                    source=goal_proprio_source,
                ),
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
                    "goal_visual_source": "svd_video",
                    "goal_proprio_source": goal_proprio_source,
                }
            )
            current_obs_idx = segment_goal_idx
    finally:
        planner.horizon = original_horizon

    return actions, imagined_visuals, segment_meta


def save_rollout_plot(rows, out_png):
    if not rows:
        raise ValueError("rows must contain at least one tensor")
    rows = [rotate_visual_sequence_180(row) for row in rows]
    t = rows[0].shape[1]
    grid = torch.cat(rows, dim=1)
    grid = rearrange(grid, "b t c h w -> (b t) c h w")
    vutils.save_image(
        grid,
        out_png,
        nrow=t,
        normalize=True,
        value_range=(-1, 1),
    )


def save_rollout_plot_per_traj(rows, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [rotate_visual_sequence_180(row) for row in rows]
    b, t = rows[0].shape[:2]
    for i in range(b):
        traj_grid = torch.cat([row[i] for row in rows], dim=0)
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
        description="CEM rollout visualization driven by SVD-generated future visual subgoals."
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--model-epoch", type=str, default="latest")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
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
    parser.add_argument("--goal-h", type=int, default=None)
    parser.add_argument("--subgoal-h", type=int, default=None)
    parser.add_argument("--n-past", type=int, default=None)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--svd-url", type=str, default="http://127.0.0.1:8765")
    parser.add_argument("--svd-timeout", type=float, default=120.0)
    parser.add_argument("--svd-width", type=int, default=128)
    parser.add_argument("--svd-height", type=int, default=128)
    parser.add_argument("--svd-request-num-frames", type=int, default=None)
    parser.add_argument(
        "--svd-frame-selection",
        type=str,
        default="first",
        choices=["first", "linspace"],
    )
    parser.add_argument("--svd-fps", type=int, default=7)
    parser.add_argument("--svd-motion-bucket-id", type=int, default=127)
    parser.add_argument("--svd-noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--svd-decode-chunk-size", type=int, default=8)
    parser.add_argument("--svd-num-inference-steps", type=int, default=30)
    parser.add_argument("--svd-seed", type=int, default=None)
    parser.add_argument(
        "--goal-proprio-source",
        type=str,
        default="gt",
        choices=["gt", "current"],
        help="Keep GT proprio targets or reuse the current proprio while using SVD visuals.",
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
        logging_prefix="cem_rollout_video",
        log_filename=None,
    )

    client = SVDVideoClient(server_url=args.svd_url, timeout_sec=args.svd_timeout)
    health = client.healthcheck()
    print(f"SVD server healthcheck: {health}")

    obs_0, _, gt_obs, meta = sample_segments(
        dset=dset,
        num_rollout=args.num_rollout,
        goal_h=goal_h,
        frameskip=model_cfg.frameskip,
        n_past=n_past,
        rng=rng,
    )
    gt_visuals = gt_obs["visual"]
    svd_goal_visuals, svd_requests = build_svd_goal_visuals(
        client=client,
        obs_0=obs_0,
        gt_obs=gt_obs,
        goal_h=goal_h,
        n_past=n_past,
        svd_width=args.svd_width,
        svd_height=args.svd_height,
        request_num_frames=args.svd_request_num_frames,
        frame_selection=args.svd_frame_selection,
        fps=args.svd_fps,
        motion_bucket_id=args.svd_motion_bucket_id,
        noise_aug_strength=args.svd_noise_aug_strength,
        decode_chunk_size=args.svd_decode_chunk_size,
        num_inference_steps=args.svd_num_inference_steps,
        seed_value=args.svd_seed,
    )

    with torch.no_grad():
        if use_segmented_replan:
            actions, imagined_visuals, replan_segments = plan_segmented_rollout_video(
                planner=planner,
                model=model,
                gt_obs=gt_obs,
                svd_goal_visuals=svd_goal_visuals,
                goal_h=goal_h,
                n_past=n_past,
                subgoal_h=subgoal_h,
                goal_proprio_source=args.goal_proprio_source,
            )
        else:
            planner.horizon = goal_h
            current_obs_idx = n_past - 1
            obs_g = {
                "visual": svd_goal_visuals[:, goal_h : goal_h + 1].clone(),
                "proprio": get_goal_proprio(
                    gt_obs=gt_obs,
                    current_obs_idx=current_obs_idx,
                    goal_idx=goal_h,
                    source=args.goal_proprio_source,
                ),
            }
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
        else model_dir / "cem_rollout_video_plots"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    epoch_tag = str(loaded_epoch) if loaded_epoch is not None else "unknown"
    mode_tag = f"_sg{subgoal_h}" if use_segmented_replan else ""
    out_png = out_root / f"cem_rollout_video_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.png"
    per_traj_dir = out_root / f"cem_rollout_video_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}_per_traj"
    out_actions = out_root / f"cem_actions_video_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.pt"
    out_meta = out_root / f"cem_meta_video_e{epoch_tag}_{args.split}_n{args.num_rollout}{mode_tag}.pt"

    save_rollout_plot([gt_visuals, svd_goal_visuals, imagined_visuals], str(out_png))
    save_rollout_plot_per_traj(
        rows=[gt_visuals, svd_goal_visuals, imagined_visuals],
        out_dir=per_traj_dir,
        prefix=f"cem_rollout_video_e{epoch_tag}_{args.split}{mode_tag}",
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
            "goal_visual_source": "svd_video",
            "goal_proprio_source": args.goal_proprio_source,
            "svd_url": args.svd_url,
            "svd_requests": svd_requests,
            "svd_frame_selection": args.svd_frame_selection,
            "svd_request_num_frames": args.svd_request_num_frames,
        },
        out_meta,
    )

    print(f"Saved rollout plot: {out_png}")
    print(f"Saved per-trajectory plots under: {per_traj_dir}")
    print(f"Saved optimized actions: {out_actions}")
    print(f"Saved sampling metadata: {out_meta}")


if __name__ == "__main__":
    main()
