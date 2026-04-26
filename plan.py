import os
import gym
import json
import copy
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed, slice_trajdict_with_t

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def data_stats_to_cpu(data_stats):
    if data_stats is None:
        return None
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in data_stats.items()
    }


def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subgoal_tag = "" if cfg_dict.get("subgoal_H") is None else f"_subgoal_H={cfg_dict['subgoal_H']}"
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}{subgoal_tag}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
    subgoal_H=None,
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
            "subgoal_H": subgoal_H,
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dict = maybe_configure_subgoal_planner_cfg(cfg_dict)
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


def maybe_configure_subgoal_planner_cfg(cfg_dict):
    subgoal_h = cfg_dict.get("subgoal_H", None)
    if subgoal_h is None:
        return cfg_dict

    subgoal_h = int(subgoal_h)
    goal_h = int(cfg_dict["goal_H"])
    if subgoal_h <= 0:
        raise ValueError(f"subgoal_H must be positive, got {subgoal_h}")
    if subgoal_h > goal_h:
        raise ValueError(
            f"subgoal_H ({subgoal_h}) must be <= goal_H ({goal_h})"
        )

    cfg_dict = copy.deepcopy(cfg_dict)
    planner_cfg = copy.deepcopy(cfg_dict["planner"])
    is_mpc = planner_cfg.get("_target_") == "planning.mpc.MPCPlanner"
    if is_mpc:
        raise ValueError(
            "Top-level subgoal_H now means rolling subgoals with env feedback and goal updates. "
            "Please use a non-MPC planner such as planner=cem or planner=gd."
        )
    if "horizon" in planner_cfg:
        planner_cfg["horizon"] = subgoal_h

    cfg_dict["planner"] = planner_cfg
    cfg_dict["subgoal_H"] = subgoal_h
    return cfg_dict


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
        data_stats=None,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.subgoal_H = cfg_dict.get("subgoal_H", None)
        if self.subgoal_H is not None:
            self.subgoal_H = int(self.subgoal_H)
        self.full_episode = bool(cfg_dict.get("full_episode", False))
        self.full_episode_traj_id = cfg_dict.get("full_episode_traj_id", None)
        self.full_episode_goal_H = cfg_dict.get("full_episode_goal_H", None)
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]
        self.target_rollout_obses = None
        self.target_rollout_states = None

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        data_stats = data_stats_to_cpu(data_stats)
        if data_stats is None:
            raise ValueError("PlanWorkspace requires data_stats loaded from the model checkpoint.")

        self.data_preprocessor = Preprocessor(
            action_mean=data_stats["action_mean"],
            action_std=data_stats["action_std"],
            state_mean=data_stats["state_mean"],
            state_std=data_stats["state_std"],
            proprio_mean=data_stats["proprio_mean"],
            proprio_std=data_stats["proprio_std"],
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # NOTE:
        # Keep planner horizon / MPC n_taken_actions from planner config.
        # Do NOT force them to goal_H here, so MPC can run receding horizon
        # with e.g. horizon=5, n_taken_actions=5 while tracking a farther goal.
        #
        # (Previous behavior forcibly overwrote to goal_H.)
        # from planning.mpc import MPCPlanner
        # if isinstance(self.planner, MPCPlanner):
        #     self.planner.sub_planner.horizon = cfg_dict["goal_H"]
        #     self.planner.n_taken_actions = cfg_dict["goal_H"]
        # else:
        #     self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            if self.full_episode:
                if self.n_evals != 1:
                    raise ValueError(
                        "full_episode=true currently requires n_evals=1 to avoid variable-length batching."
                    )
                observations, states, actions, env_info = self.sample_full_episode_from_dset(
                    traj_id=self.full_episode_traj_id
                )
                self.env.update_env(env_info)
                init_state = np.array([states[0][0]])
                full_actions = actions[0]
                # World-model actions are chunked by frameskip, so trim to a multiple.
                usable_exec_steps = (full_actions.shape[0] // self.frameskip) * self.frameskip
                if usable_exec_steps <= 0:
                    raise ValueError(
                        f"Episode too short for frameskip={self.frameskip}: action length={full_actions.shape[0]}"
                    )
                max_goal_h = usable_exec_steps // self.frameskip
                if self.full_episode_goal_H is None:
                    goal_h = max_goal_h
                else:
                    goal_h = int(self.full_episode_goal_H)
                    if goal_h <= 0:
                        raise ValueError(
                            f"full_episode_goal_H must be positive, got {goal_h}"
                        )
                    if goal_h > max_goal_h:
                        raise ValueError(
                            f"full_episode_goal_H={goal_h} exceeds available goal horizon "
                            f"{max_goal_h} for traj_id={self.full_episode_traj_id}"
                        )
                full_actions = full_actions[: self.frameskip * goal_h]
                self.goal_H = int(goal_h)
                self.cfg_dict["goal_H"] = int(goal_h)
                print(
                    f"[plan] full_episode enabled: traj_id={self.full_episode_traj_id if self.full_episode_traj_id is not None else 'random'} "
                    f"usable_exec_steps={usable_exec_steps}, max_goal_H={max_goal_h}, goal_H={goal_h}"
                )
                actions = torch.stack([full_actions], dim=0)
            else:
            # update env config from val trajs
                observations, states, actions, env_info = (
                    self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
                )
                self.env.update_env(env_info)

                # get states from val trajs
                init_state = [x[0] for x in states]
                init_state = np.array(init_state)
                actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.target_rollout_obses = rollout_obses
            self.target_rollout_states = rollout_states
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_full_episode_from_dset(self, traj_id=None):
        states = []
        actions = []
        observations = []
        env_info = []

        if traj_id is None:
            traj_id = random.randint(0, len(self.dset) - 1)
        if not (0 <= int(traj_id) < len(self.dset)):
            raise ValueError(f"full_episode_traj_id out of range: {traj_id}")

        obs, act, state, e_info = self.dset[int(traj_id)]
        state = state.numpy()
        observations.append(obs)
        states.append(state)
        actions.append(act)
        env_info.append(e_info)
        self.full_episode_traj_id = int(traj_id)
        return observations, states, actions, env_info

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]
        self.target_rollout_obses = data.get("target_rollout_obses")
        self.target_rollout_states = data.get("target_rollout_states")

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                    "subgoal_H": self.subgoal_H,
                    "target_rollout_obses": self.target_rollout_obses,
                    "target_rollout_states": self.target_rollout_states,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def _dump_log_entry(self, logs):
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")

    def _set_planner_horizon(self, horizon):
        if hasattr(self.planner, "sub_planner"):
            raise ValueError(
                "Top-level subgoal_H uses an outer rolling-subgoal loop and does not support MPC planners. "
                "Please use planner=cem or planner=gd."
            )
        if not hasattr(self.planner, "horizon"):
            raise ValueError("Planner does not expose a horizon attribute.")
        self.planner.horizon = int(horizon)

    def _get_subgoal_cond(self, goal_h):
        if self.target_rollout_obses is None or self.target_rollout_states is None:
            raise ValueError(
                "subgoal_H requires a target trajectory with intermediate env observations/states. "
                "This is supported for goal_source=dset/random_action and files that provide target_rollout_obses/states."
            )
        goal_idx = int(goal_h) * self.frameskip
        obs_g = {
            key: np.expand_dims(arr[:, goal_idx], axis=1)
            for key, arr in self.target_rollout_obses.items()
        }
        state_g = self.target_rollout_states[:, goal_idx]
        return obs_g, state_g

    def perform_subgoal_planning(self):
        if self.subgoal_H is None:
            raise ValueError("perform_subgoal_planning called without subgoal_H")

        if self.target_rollout_obses is None or self.target_rollout_states is None:
            raise ValueError(
                "subgoal_H requires intermediate target observations/states, which are unavailable for the current goal source."
            )

        initial_obs_0 = self.obs_0
        initial_state_0 = self.state_0
        final_obs_g = self.obs_g
        final_state_g = self.state_g
        current_obs_0 = self.obs_0
        current_state_0 = self.state_0
        accumulated_actions = []
        original_horizon = getattr(self.planner, "horizon", None)
        segment_start_h = 0
        segment_idx = 0

        try:
            while segment_start_h < self.goal_H:
                segment_end_h = min(segment_start_h + self.subgoal_H, self.goal_H)
                current_horizon = segment_end_h - segment_start_h
                self._set_planner_horizon(current_horizon)
                current_obs_g, current_state_g = self._get_subgoal_cond(segment_end_h)
                self.evaluator.assign_init_cond(current_obs_0, current_state_0)
                self.evaluator.assign_goal_cond(current_obs_g, current_state_g)

                if hasattr(self.planner, "logging_prefix"):
                    self.planner.logging_prefix = f"subgoal_{segment_idx}"

                if self.debug_dset_init and self.gt_actions is not None:
                    actions_init = self.gt_actions[:, segment_start_h:segment_end_h]
                else:
                    actions_init = None

                planned_actions, _ = self.planner.plan(
                    obs_0=current_obs_0,
                    obs_g=current_obs_g,
                    actions=actions_init,
                )
                taken_actions = planned_actions.detach()[:, :current_horizon]
                accumulated_actions.append(taken_actions)

                subgoal_logs, _, e_obses, e_states = self.evaluator.eval_actions(
                    taken_actions,
                    action_len=None,
                    filename=f"subgoal_{segment_idx}",
                    save_video=False,
                )
                subgoal_logs = {f"subgoal_eval/{k}": v for k, v in subgoal_logs.items()}
                subgoal_logs.update(
                    {
                        "subgoal_eval/segment_idx": segment_idx,
                        "subgoal_eval/segment_end_h": segment_end_h,
                    }
                )
                self.wandb_run.log(subgoal_logs)
                self._dump_log_entry(subgoal_logs)

                current_obs_0 = slice_trajdict_with_t(e_obses, start_idx=-1)
                current_state_0 = e_states[:, -1]
                segment_start_h = segment_end_h
                segment_idx += 1
        finally:
            if original_horizon is not None and hasattr(self.planner, "horizon"):
                self.planner.horizon = original_horizon

        actions = torch.cat(accumulated_actions, dim=1)
        self.evaluator.assign_init_cond(initial_obs_0, initial_state_0)
        self.evaluator.assign_goal_cond(final_obs_g, final_state_g)
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), None, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        self._dump_log_entry(logs)
        return logs

    def perform_planning(self):
        if self.subgoal_H is not None:
            return self.perform_subgoal_planning()

        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        self._dump_log_entry(logs)
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    if "data_stats" in payload:
        result["data_stats"] = data_stats_to_cpu(payload["data_stats"])
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    if "data_stats" not in result:
        raise ValueError(
            f"Checkpoint does not contain data_stats: {model_ckpt}. "
            "Please use a checkpoint saved by the updated trainer."
        )
    return model, result["data_stats"]


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    cfg_dict = maybe_configure_subgoal_planner_cfg(cfg_dict)
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    with open_dict(model_cfg):
        if cfg_dict.get("dataset_path") is not None:
            model_cfg.env.dataset.data_path = cfg_dict["dataset_path"]
            print(f"Planning dataset path override: {model_cfg.env.dataset.data_path}")
        if cfg_dict.get("dataset_target") is not None:
            model_cfg.env.dataset._target_ = cfg_dict["dataset_target"]
            print(f"Planning dataset target override: {model_cfg.env.dataset._target_}")

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model, data_stats = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        data_stats=data_stats,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
