#!/usr/bin/env python3

import abc
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plan import load_model
from planning.cem import CEMPlanner
from planning.objectives import create_objective_fn
from preprocessor import Preprocessor
from utils import seed


class DummyWandbRun:
    def log(self, *args, **kwargs):
        pass


class RobotBridge(abc.ABC):
    """Minimal interface required by the online deployment loop.

    Observation convention follows the rest of this repo:
    - visual: uint8 [H, W, C]
    - proprio: float32 [7]

    Action convention passed into execute_action is denormalized dataset-space
    Cartesian delta action:
    [dx, dy, dz, drx, dry, drz, d_gripper]
    """

    @abc.abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def execute_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileBackedRobotBridge(RobotBridge):
    """Dry-run bridge that reads observation files and logs actions to disk.

    This is useful for testing the deployment loop before wiring the real UR5e.
    An external process can update `current_image_path` / `current_proprio_path`
    between replans to emulate fresh robot observations.
    """

    def __init__(
        self,
        current_image_path: Path,
        current_proprio_path: Path,
        action_log_path: Optional[Path] = None,
        sleep_after_action: float = 0.0,
    ):
        self.current_image_path = current_image_path
        self.current_proprio_path = current_proprio_path
        self.action_log_path = action_log_path
        self.sleep_after_action = float(sleep_after_action)
        self.action_log_path_parent_ready = False

    def get_observation(self) -> Dict[str, np.ndarray]:
        visual = load_image_uint8(self.current_image_path)
        proprio = load_vector(self.current_proprio_path)
        return {
            "visual": visual,
            "proprio": proprio.astype(np.float32),
        }

    def execute_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        print(f"[FileBackedRobotBridge] execute_action: {action.tolist()}", flush=True)
        if self.action_log_path is not None:
            if not self.action_log_path_parent_ready:
                self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
                self.action_log_path_parent_ready = True
            with self.action_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"action": action.tolist(), "time": time.time()}) + "\n")
        if self.sleep_after_action > 0:
            time.sleep(self.sleep_after_action)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Training output dir containing hydra.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--model-epoch",
        type=str,
        default="latest",
        help="Checkpoint tag. Use latest or an epoch suffix such as 40.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional override for the dataset path in hydra.yaml.",
    )
    parser.add_argument(
        "--goal-image-path",
        type=Path,
        required=True,
        help="Target RGB image for visual goal conditioning.",
    )
    parser.add_argument(
        "--goal-proprio-path",
        type=Path,
        default=None,
        help="Optional goal proprio/state file (.npy, .json, .txt). Defaults to current proprio.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--objective-base", type=float, default=2.0)
    parser.add_argument(
        "--objective-mode",
        type=str,
        choices=["last", "all"],
        default="last",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Number of world-model steps to execute before re-planning.",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=10,
        help="How many replan cycles to run before stopping.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("deploy_outputs"),
        help="Directory for plan/action dumps.",
    )
    parser.add_argument(
        "--bridge",
        type=str,
        choices=["file"],
        default="file",
        help="Bridge backend. Extend this script with your real UR5e bridge.",
    )
    parser.add_argument(
        "--current-image-path",
        type=Path,
        default=None,
        help="Required for --bridge=file. Image file representing the current robot view.",
    )
    parser.add_argument(
        "--current-proprio-path",
        type=Path,
        default=None,
        help="Required for --bridge=file. Proprio/state file representing the current robot state.",
    )
    parser.add_argument(
        "--action-log-path",
        type=Path,
        default=None,
        help="Optional jsonl file for logging executed actions in file bridge mode.",
    )
    parser.add_argument(
        "--sleep-after-action",
        type=float,
        default=0.0,
        help="Only used by the file bridge to slow down the dry run.",
    )
    return parser.parse_args()


def load_image_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_vector(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        vec = np.load(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            vec = np.asarray(json.load(f), dtype=np.float32)
    else:
        vec = np.loadtxt(path, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return vec


def build_bridge(args) -> RobotBridge:
    if args.bridge == "file":
        if args.current_image_path is None or args.current_proprio_path is None:
            raise ValueError(
                "--bridge=file requires --current-image-path and --current-proprio-path."
            )
        return FileBackedRobotBridge(
            current_image_path=args.current_image_path,
            current_proprio_path=args.current_proprio_path,
            action_log_path=args.action_log_path,
            sleep_after_action=args.sleep_after_action,
        )
    raise NotImplementedError(f"Unsupported bridge type: {args.bridge}")


def load_runtime_components(
    model_dir: Path,
    model_epoch: str,
    device: torch.device,
    dataset_path_override: Optional[Path] = None,
):
    hydra_path = model_dir / "hydra.yaml"
    if not hydra_path.exists():
        raise FileNotFoundError(f"hydra.yaml not found under {model_dir}")

    model_cfg = OmegaConf.load(hydra_path)
    with open_dict(model_cfg):
        if dataset_path_override is not None:
            model_cfg.env.dataset.data_path = str(dataset_path_override)

    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    valid_traj_dset = traj_dset["valid"]

    preprocessor = Preprocessor(
        action_mean=valid_traj_dset.action_mean,
        action_std=valid_traj_dset.action_std,
        state_mean=valid_traj_dset.state_mean,
        state_std=valid_traj_dset.state_std,
        proprio_mean=valid_traj_dset.proprio_mean,
        proprio_std=valid_traj_dset.proprio_std,
        transform=valid_traj_dset.transform,
    )

    ckpt_name = "model_latest.pth" if model_epoch == "latest" else f"model_{model_epoch}.pth"
    model_ckpt = model_dir / "checkpoints" / ckpt_name
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt}")

    model = load_model(
        model_ckpt=model_ckpt,
        train_cfg=model_cfg,
        num_action_repeat=model_cfg.num_action_repeat,
        device=device,
    )
    model.eval()
    return model_cfg, valid_traj_dset, preprocessor, model


def create_planner(args, model, preprocessor, action_dim: int) -> CEMPlanner:
    objective_fn = create_objective_fn(
        alpha=args.alpha,
        base=args.objective_base,
        mode=args.objective_mode,
    )
    return CEMPlanner(
        horizon=args.horizon,
        topk=args.topk,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        opt_steps=args.opt_steps,
        eval_every=max(args.opt_steps + 1, 999999),
        wm=model,
        action_dim=action_dim,
        objective_fn=objective_fn,
        preprocessor=preprocessor,
        evaluator=None,
        wandb_run=DummyWandbRun(),
        logging_prefix="deploy",
        log_filename=None,
    )


def build_obs_batch(
    visual_hist: Sequence[np.ndarray],
    proprio_hist: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    visual = np.stack(list(visual_hist), axis=0)
    proprio = np.stack(list(proprio_hist), axis=0)
    return {
        "visual": np.expand_dims(visual, axis=0),
        "proprio": np.expand_dims(proprio, axis=0),
    }


def build_goal_obs(
    goal_image_path: Path,
    goal_proprio: np.ndarray,
) -> Dict[str, np.ndarray]:
    visual = load_image_uint8(goal_image_path)
    proprio = np.asarray(goal_proprio, dtype=np.float32)
    return {
        "visual": visual[None, None],
        "proprio": proprio[None, None],
    }


def ensure_history(
    bridge: RobotBridge,
    history_len: int,
) -> Tuple[Deque[np.ndarray], Deque[np.ndarray]]:
    visual_hist: Deque[np.ndarray] = deque(maxlen=history_len)
    proprio_hist: Deque[np.ndarray] = deque(maxlen=history_len)
    first_obs = bridge.get_observation()
    for _ in range(history_len):
        visual_hist.append(first_obs["visual"])
        proprio_hist.append(first_obs["proprio"])
    return visual_hist, proprio_hist


def plan_once(
    planner: CEMPlanner,
    model,
    preprocessor: Preprocessor,
    obs_0: Dict[str, np.ndarray],
    obs_g: Dict[str, np.ndarray],
):
    actions, _ = planner.plan(obs_0=obs_0, obs_g=obs_g, actions=None)
    actions_cpu = actions.detach().cpu()
    primitive_action_dim = int(preprocessor.action_mean.shape[0])
    if planner.action_dim % primitive_action_dim != 0:
        raise ValueError(
            f"Planner action dim {planner.action_dim} is not divisible by "
            f"primitive action dim {primitive_action_dim}."
        )
    frameskip = planner.action_dim // primitive_action_dim
    primitive_actions = actions_cpu.reshape(
        actions_cpu.shape[0],
        actions_cpu.shape[1],
        frameskip,
        primitive_action_dim,
    )
    primitive_actions = primitive_actions.reshape(actions_cpu.shape[0], -1, primitive_action_dim)
    primitive_actions = preprocessor.denormalize_actions(primitive_actions).numpy()

    with torch.no_grad():
        trans_obs_0 = preprocessor.transform_obs(obs_0)
        trans_obs_0 = {k: v.to(planner.device) for k, v in trans_obs_0.items()}
        z_obses, _ = model.rollout(obs_0=trans_obs_0, act=actions.to(planner.device))
        decoded = None
        if model.decoder is not None:
            decoded = model.decode_obs(z_obses)[0]["visual"].detach().cpu()
    return actions_cpu, primitive_actions[0], decoded


def save_cycle_artifacts(
    save_dir: Path,
    cycle_idx: int,
    planned_actions_wm: torch.Tensor,
    primitive_actions: np.ndarray,
    imagined_visuals: Optional[torch.Tensor],
):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"cycle_{cycle_idx:03d}_primitive_actions.npy", primitive_actions)
    torch.save(planned_actions_wm, save_dir / f"cycle_{cycle_idx:03d}_wm_actions.pt")
    if imagined_visuals is not None:
        torch.save(imagined_visuals, save_dir / f"cycle_{cycle_idx:03d}_imagined_visuals.pt")


def main():
    args = parse_args()
    seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_cfg, traj_dset, preprocessor, model = load_runtime_components(
        model_dir=args.model_dir,
        model_epoch=args.model_epoch,
        device=device,
        dataset_path_override=args.dataset_path,
    )

    history_len = int(args.history_len) if args.history_len is not None else int(model_cfg.num_hist)
    primitive_action_dim = int(traj_dset.action_dim)
    planner_action_dim = primitive_action_dim * int(model_cfg.frameskip)
    planner = create_planner(
        args=args,
        model=model,
        preprocessor=preprocessor,
        action_dim=planner_action_dim,
    )

    bridge = build_bridge(args)
    visual_hist, proprio_hist = ensure_history(bridge, history_len)

    current_goal_proprio = None
    if args.goal_proprio_path is not None:
        current_goal_proprio = load_vector(args.goal_proprio_path)

    try:
        for cycle_idx in range(args.max_replans):
            current_obs = bridge.get_observation()
            visual_hist.append(current_obs["visual"])
            proprio_hist.append(current_obs["proprio"])

            if current_goal_proprio is None:
                goal_proprio = current_obs["proprio"]
            else:
                goal_proprio = current_goal_proprio

            obs_0 = build_obs_batch(visual_hist, proprio_hist)
            obs_g = build_goal_obs(args.goal_image_path, goal_proprio)

            wm_actions, primitive_actions, imagined_visuals = plan_once(
                planner=planner,
                model=model,
                preprocessor=preprocessor,
                obs_0=obs_0,
                obs_g=obs_g,
            )
            save_cycle_artifacts(
                save_dir=args.save_dir,
                cycle_idx=cycle_idx,
                planned_actions_wm=wm_actions,
                primitive_actions=primitive_actions,
                imagined_visuals=imagined_visuals,
            )

            exec_steps = min(int(args.execute_steps), wm_actions.shape[1])
            exec_primitive = exec_steps * int(model_cfg.frameskip)
            print(
                f"[deploy] cycle={cycle_idx} executing {exec_steps} WM step(s) "
                f"= {exec_primitive} primitive action(s)",
                flush=True,
            )
            for action_idx in range(exec_primitive):
                bridge.execute_action(primitive_actions[action_idx])
                latest_obs = bridge.get_observation()
                visual_hist.append(latest_obs["visual"])
                proprio_hist.append(latest_obs["proprio"])
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
