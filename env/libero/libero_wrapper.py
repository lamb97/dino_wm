import json
import os
import sys
import importlib
import warnings
from pathlib import Path

import gym
import numpy as np
from scipy.spatial.transform import Rotation as R


class LiberoWrapper(gym.Env):
    """Adapter from LIBERO OffScreenRenderEnv to dino_wm planning env API."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        default_env_args=None,
        bddl_root=None,
        camera_key="agentview_image",
        action_clip=1.0,  # fallback only when controller metadata is missing
        stabilize_steps=5,
        position_gain=4.0,
        gripper_delta_scale=0.01,
        calibrate_gripper_from_raw=True,
    ):
        super().__init__()
        self.default_env_args = default_env_args or {}
        self.bddl_root = bddl_root
        if self.bddl_root is None:
            try:
                from libero.libero import get_libero_path

                self.bddl_root = get_libero_path("bddl_files")
            except Exception:
                self.bddl_root = None
        self.camera_key = camera_key
        self.action_clip = action_clip
        self.stabilize_steps = int(stabilize_steps)
        self.position_gain = float(position_gain)
        self.gripper_delta_scale = float(gripper_delta_scale)
        self.calibrate_gripper_from_raw = bool(calibrate_gripper_from_raw)

        self._env = None
        self._env_signature = None
        self._episode_sim_states = None
        self._episode_state_traj = None
        self._last_rollout_success = False
        self._latest_env_args = None
        self._bddl_resolve_cache = {}

        # Controller scaling (OSC_POSE): command space -> physical delta.
        # We invert this mapping for precise delta->command conversion.
        self._pose_input_min = -np.ones(6, dtype=np.float32)
        self._pose_input_max = np.ones(6, dtype=np.float32)
        self._pose_output_min = -np.array(
            [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32
        )
        self._pose_output_max = np.array(
            [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32
        )

        # Gripper conversion fallback assumes a simple normalized delta-width command.
        self._gripper_affine = (
            np.float32(1.0 / max(self.gripper_delta_scale, 1e-6)),
            np.float32(0.0),
        )

        # dino_wm planner expects 7D action/proprio.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})
        self.action_dim = 7

    def _debug(self, msg):
        print(f"[LiberoWrapper] {msg}", file=sys.stderr, flush=True)

    def _decode_json_maybe(self, value):
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _normalize_env_args(self, env_args):
        env_args = self._decode_json_maybe(env_args)
        if env_args is None:
            env_args = {}
        if not isinstance(env_args, dict):
            raise ValueError(f"env_args must be a dict, got {type(env_args)}")

        # LIBERO metadata may be wrapped as:
        # {"type": ..., "bddl_file": ..., "env_kwargs": {...}}
        # or directly env kwargs dict.
        raw_kwargs = env_args.get("env_kwargs", env_args)
        if not isinstance(raw_kwargs, dict):
            raise ValueError(
                f"env_args['env_kwargs'] must be dict when provided, got {type(raw_kwargs)}"
            )

        merged = dict(self.default_env_args)
        merged.update(raw_kwargs)

        # ControlEnv internally constructs `controller_configs` from `controller`.
        # Passing `controller_configs` again in kwargs causes duplicate argument errors.
        ctrl_cfg = merged.pop("controller_configs", None)
        if ctrl_cfg is not None and "controller" not in merged:
            if isinstance(ctrl_cfg, dict) and ctrl_cfg.get("type") is not None:
                merged["controller"] = ctrl_cfg["type"]

        bddl_file = (
            merged.get("bddl_file_name")
            or env_args.get("bddl_file_name")
            or env_args.get("bddl_file")
        )
        if bddl_file is None:
            raise ValueError(
                "LIBERO env_args must include `bddl_file_name` for env creation."
            )
        resolved = self._resolve_bddl_path(str(bddl_file))
        if resolved is None:
            raise ValueError(
                f"Could not resolve bddl_file_name path: {bddl_file}. "
                "Set `bddl_root` in env config or ensure LIBERO bddl files are available."
            )
        merged["bddl_file_name"] = str(resolved)
        return merged

    def _resolve_bddl_path(self, bddl_file):
        cached = self._bddl_resolve_cache.get(bddl_file)
        if cached is not None:
            p = Path(cached)
            if p.exists():
                return p

        p = Path(bddl_file)
        if p.exists():
            self._bddl_resolve_cache[bddl_file] = str(p)
            return p

        candidates = []
        search_roots = []
        if self.bddl_root is not None:
            root = Path(self.bddl_root)
            search_roots.append(root)
            candidates.append(root / bddl_file)
            marker = "bddl_files/"
            if marker in bddl_file:
                suffix = bddl_file.split(marker, 1)[1]
                candidates.append(root / suffix)

        # Common local clone location fallback.
        repo_root = Path(__file__).resolve().parents[3] / "LIBERO" / "libero" / "libero" / "bddl_files"
        search_roots.append(repo_root)
        candidates.append(repo_root / bddl_file)
        marker = "bddl_files/"
        if marker in bddl_file:
            suffix = bddl_file.split(marker, 1)[1]
            candidates.append(repo_root / suffix)

        for cand in candidates:
            if cand.exists():
                self._bddl_resolve_cache[bddl_file] = str(cand)
                return cand

        # Fallback: dataset metadata may use a benchmark folder name (e.g., libero_100)
        # that doesn't exist in local bddl_files. Match by basename under bddl_root.
        target_name = Path(bddl_file).name
        for root in search_roots:
            if root is None or not Path(root).exists():
                continue
            matches = list(Path(root).rglob(target_name))
            if len(matches) > 0:
                self._bddl_resolve_cache[bddl_file] = str(matches[0])
                return matches[0]
        return None

    def _ensure_env(self):
        if self._env is None:
            raise RuntimeError(
                "LIBERO env not initialized. Call update_env(env_info) with env_args first."
            )

    def _maybe_create_env(self, env_args):
        signature = json.dumps(env_args, sort_keys=True, default=str)
        if self._env is not None and signature == self._env_signature:
            return

        if self._env is not None:
            self._env.close()
            self._env = None

        OffScreenRenderEnv = self._import_offscreen_env()

        self._env = OffScreenRenderEnv(**env_args)
        self._env_signature = signature
        self._latest_env_args = env_args

    def _import_offscreen_env(self):
        try:
            from libero.libero.envs import OffScreenRenderEnv

            return OffScreenRenderEnv
        except ModuleNotFoundError:
            pass

        # Fallback: try common local LIBERO repo locations.
        candidates = []
        env_root = os.environ.get("LIBERO_ROOT")
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend(
            [
                Path.home() / "LIBERO",
                Path.home() / "openpi" / "third_party" / "libero",
            ]
        )

        for cand in candidates:
            if (cand / "libero").exists():
                cstr = str(cand)
                if cstr not in sys.path:
                    sys.path.insert(0, cstr)
                try:
                    mod = importlib.import_module("libero.libero.envs")
                    return getattr(mod, "OffScreenRenderEnv")
                except ModuleNotFoundError:
                    continue

        raise ModuleNotFoundError(
            "No module named 'libero'. Install LIBERO in this env or set "
            "`LIBERO_ROOT` / `PYTHONPATH` to your LIBERO repo root "
            "(e.g. /users/6/liu03222/LIBERO)."
        )

    def _quat_to_euler_xyz(self, quat_xyzw):
        # quat from robosuite obs is xyzw.
        x, y, z, w = quat_xyzw
        ysqr = y * y
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _obs_to_state(self, obs):
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        euler = self._quat_to_euler_xyz(eef_quat)
        gripper_q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        width = np.array([np.abs(gripper_q[0] - gripper_q[1])], dtype=np.float32)
        return np.concatenate([eef_pos, euler, width], axis=0).astype(np.float32)

    def _obs_to_dino(self, obs):
        visual = np.asarray(obs[self.camera_key], dtype=np.uint8)
        proprio = self._obs_to_state(obs)
        return {"visual": visual, "proprio": proprio}

    def _select_init_sim_state(self, init_state):
        if self._episode_sim_states is None:
            return None

        init_state = np.asarray(init_state, dtype=np.float32)
        if init_state.shape[-1] == self._episode_sim_states.shape[-1]:
            return init_state

        if (
            self._episode_state_traj is not None
            and init_state.shape[-1] == self._episode_state_traj.shape[-1]
        ):
            dists = np.linalg.norm(self._episode_state_traj - init_state[None, :], axis=1)
            idx = int(np.argmin(dists))
            return self._episode_sim_states[idx]

        return self._episode_sim_states[0]

    def _action_to_libero(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        dpos = action[:3] * self.position_gain
        deuler = action[3:6]
        dwidth = float(action[6])

        # Dataset rotation delta is stored as Euler xyz for the relative transform.
        # Convert that transform back to axis-angle / exponential coordinates, which
        # is what robosuite OSC_POSE expects for the orientation command.
        drotvec = R.from_euler("xyz", deuler).as_rotvec().astype(np.float32)
        pose_delta = np.concatenate([dpos, drotvec], axis=0).astype(np.float32)
        cmd_pose = self._inverse_scale_pose_delta(pose_delta)

        # Dataset gripper uses delta finger width: positive opens, negative closes.
        # Robosuite single-DoF grippers expect an open / close direction command:
        # -1 => open, +1 => close.
        if abs(dwidth) < 1e-6:
            cmd_gripper = 0.0
        else:
            cmd_gripper = float(-np.sign(dwidth))

        cmd = np.concatenate([cmd_pose, np.array([cmd_gripper], dtype=np.float32)], axis=0)
        if np.any(np.isnan(cmd)):
            raise ValueError(f"NaN in converted action. input={action}, converted={cmd}")
        return cmd

    def _to_dim_array(self, value, dim, default):
        if value is None:
            return np.asarray(default, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, dim)
        if arr.shape[0] != dim:
            return np.asarray(default, dtype=np.float32)
        return arr

    def _update_controller_scaling(self, env_info):
        env_args = env_info.get("env_args", {})
        if not isinstance(env_args, dict):
            return
        env_kwargs = env_args.get("env_kwargs", env_args)
        if not isinstance(env_kwargs, dict):
            return
        ctrl = env_kwargs.get("controller_configs", {})
        if not isinstance(ctrl, dict):
            return

        self._pose_input_max = self._to_dim_array(
            ctrl.get("input_max"), 6, self._pose_input_max
        )
        self._pose_input_min = self._to_dim_array(
            ctrl.get("input_min"), 6, self._pose_input_min
        )
        self._pose_output_max = self._to_dim_array(
            ctrl.get("output_max"), 6, self._pose_output_max
        )
        self._pose_output_min = self._to_dim_array(
            ctrl.get("output_min"), 6, self._pose_output_min
        )

    def _inverse_scale_pose_delta(self, pose_delta):
        out_min = self._pose_output_min
        out_max = self._pose_output_max
        in_min = self._pose_input_min
        in_max = self._pose_input_max

        denom = out_max - out_min
        safe = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        scaled = (pose_delta - out_min) / safe
        cmd = scaled * (in_max - in_min) + in_min
        cmd = np.clip(cmd, in_min, in_max)
        cmd = np.clip(cmd, -self.action_clip, self.action_clip)
        return cmd.astype(np.float32)

    def update_env(self, env_info):
        if env_info is None:
            return

        env_args = env_info.get("env_args")
        if env_args is not None or self._env is None:
            normalized = self._normalize_env_args(env_args if env_args is not None else {})
            self._maybe_create_env(normalized)

        self._episode_sim_states = None
        self._episode_state_traj = None
        sim_state_path = env_info.get("sim_state_path")
        if sim_state_path is not None and Path(sim_state_path).exists():
            import torch

            sim_states = torch.load(sim_state_path)
            if hasattr(sim_states, "cpu"):
                sim_states = sim_states.cpu().numpy()
            self._episode_sim_states = np.asarray(sim_states, dtype=np.float32)

        state_traj = env_info.get("state_traj")
        if state_traj is not None:
            self._episode_state_traj = np.asarray(state_traj, dtype=np.float32)

        self._update_controller_scaling(env_info)

    def eval_state(self, goal_state, cur_state):
        goal_state = np.asarray(goal_state, dtype=np.float32)
        cur_state = np.asarray(cur_state, dtype=np.float32)
        if goal_state.shape == cur_state.shape:
            state_dist = float(np.linalg.norm(goal_state - cur_state))
        else:
            state_dist = float("nan")
        return {
            "success": bool(self._last_rollout_success),
            "state_dist": state_dist,
        }

    def prepare(self, seed, init_state):
        self._debug(f"prepare start seed={seed}")
        self._ensure_env()
        self._env.seed(int(seed))
        self._debug("prepare: first env.reset()")
        self._env.reset()

        init_state_arr = np.asarray(init_state, dtype=np.float32)
        exact_sim_state = (
            self._episode_sim_states is not None
            and init_state_arr.shape[-1] == self._episode_sim_states.shape[-1]
        )
        sim_state = self._select_init_sim_state(init_state_arr)
        if sim_state is not None:
            self._debug("prepare: set_init_state()")
            obs = self._env.set_init_state(sim_state)
        else:
            self._debug("prepare: fallback env.reset()")
            obs = self._env.reset()

        # Let contact dynamics settle when restoring from approximate episode states.
        if not exact_sim_state:
            zero = np.zeros(self.action_dim, dtype=np.float32)
            self._debug(f"prepare: stabilize with {self.stabilize_steps} zero steps")
            for step_idx in range(self.stabilize_steps):
                obs, _, _, _ = self._env.step(zero)
                if step_idx == 0:
                    self._debug("prepare: first stabilize step finished")

        self._debug("prepare: building dino obs")
        dino_obs = self._obs_to_dino(obs)
        state = self._env.get_sim_state().copy().astype(np.float32)
        self._debug("prepare done")
        return dino_obs, state

    def sample_random_init_goal_states(self, seed):
        self._ensure_env()
        rs = np.random.RandomState(seed)
        # Fallback sampling for `goal_source=random_state`.
        self._env.seed(int(seed))
        self._env.reset()
        init_sim = self._env.get_sim_state().copy()
        for _ in range(int(rs.randint(5, 25))):
            a = rs.uniform(low=-0.2, high=0.2, size=(self.action_dim,)).astype(np.float32)
            self._env.step(self._action_to_libero(a))
        goal_sim = self._env.get_sim_state().copy()
        return init_sim, goal_sim

    def step_multiple(self, actions):
        self._debug(f"step_multiple start num_actions={len(actions)}")
        obses = []
        rewards = []
        dones = []
        infos = []
        for idx, action in enumerate(actions):
            obs, reward, done, info = self._env.step(self._action_to_libero(action))
            if idx == 0:
                self._debug("step_multiple: first action step finished")
            dino_obs = self._obs_to_dino(obs)
            obses.append(dino_obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            info = dict(info)
            info["state"] = self._env.get_sim_state().copy().astype(np.float32)
            info["success"] = bool(done)
            infos.append(info)

        keys = obses[0].keys()
        obs_stack = {k: np.stack([o[k] for o in obses], axis=0) for k in keys}
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        # Aggregate dict entries into np arrays when possible.
        agg_info = {}
        for k in infos[0].keys():
            vals = [it[k] for it in infos]
            try:
                agg_info[k] = np.stack(vals, axis=0)
            except Exception:
                agg_info[k] = np.array(vals, dtype=object)
        self._debug("step_multiple done")
        return obs_stack, rewards, dones, agg_info

    def rollout(self, seed, init_state, actions):
        self._debug(f"rollout start num_actions={len(actions)}")
        obs0, state0 = self.prepare(seed, init_state)
        obses, _, dones, infos = self.step_multiple(actions)

        out_obs = {}
        for k in obses.keys():
            out_obs[k] = np.concatenate([obs0[k][None], obses[k]], axis=0)
        states = np.concatenate([state0[None], infos["state"]], axis=0)
        self._last_rollout_success = bool(np.any(dones))
        self._debug("rollout done")
        return out_obs, states

    # Gym compatibility.
    def reset(self):
        self._ensure_env()
        obs = self._env.reset()
        return self._obs_to_dino(obs)

    def step(self, action):
        self._ensure_env()
        obs, reward, done, info = self._env.step(self._action_to_libero(action))
        self._debug("prepare: building dino obs")
        dino_obs = self._obs_to_dino(obs)
        info = dict(info)
        info["state"] = self._env.get_sim_state().copy().astype(np.float32)
        return dino_obs, reward, done, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
