"""
MADDPG trainer (skeleton — file 12e-1).

This file establishes the MaddpgTrainer class and all constructor-time
setup: network and optimizer creation, target network hard-sync, replay
buffer instantiation, tensorboard writer, seed management, and checkpoint
save/load helpers.

The two substantive methods — _update() and run() — are added in files
12e-2 and 12e-3 respectively.

Seed management is centralised in _seed_everything() and covers torch,
numpy, and python random. The Unity-side seed is passed via the CLI
script (file 12f) and logged to tensorboard at training start.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from training.maddpg.config import (
    MaddpgConfig,
    action_size_for_bandwidth,
    IDX_GATE_STATE,
    IDX_COOLDOWN,
    IDX_BELT_SPEED,
    IDX_DEST_MAPPING_START,
    IDX_PACKAGE_SLOTS_START,
    IDX_PACKAGE_SLOTS_END,
    IDX_OTHER_AGENTS_START_FULL,
    IDX_OTHER_AGENTS_END_FULL,
    IDX_CONGESTION_START_FULL,
    IDX_CONGESTION_END_FULL,
    IDX_CONGESTION_START_PARTIAL,
    IDX_CONGESTION_END_PARTIAL,
    FULL_OBS_SIZE,
)
from training.maddpg.critic_input import expected_critic_input_dim
from training.maddpg.networks import Actor, Critic, hard_update, soft_update
from training.maddpg.replay_buffer import ReplayBuffer


class MaddpgTrainer:
    """
    Multi-Agent DDPG trainer with centralised critics.

    One Actor and one Critic per agent (no parameter sharing at the
    network level). Target networks for both. Experience shared across
    agents via a single ReplayBuffer storing per-step (obs, actor_outputs,
    masked_actions, rewards, next_obs, dones) tuples for all agents.
    """

    def __init__(
        self,
        config: MaddpgConfig,
        obs_size: int,
        comm_bandwidth: int,
        run_id: str,
        seed: int,
        results_dir: str = "results",
        device: Optional[str] = None,
    ):
        """
        Args:
            config: MaddpgConfig bundle with all hyperparameters.
            obs_size: per-agent observation vector length (34/36/38/40).
            comm_bandwidth: 0 / 1 / 3 — determines action_size via
                action_size_for_bandwidth(). Messages (if any) are
                already part of obs_size.
            run_id: unique identifier for this training run. Used for
                tensorboard log dir and checkpoint file names.
            seed: master RNG seed. Applied to torch, numpy, python random,
                and the replay buffer's dedicated RNG.
            results_dir: parent directory under which logs/ and
                checkpoints/ subdirs are created.
            device: 'cpu' or 'cuda'. If None, auto-detects ('cuda' if
                available else 'cpu').
        """
        self._config = config
        self._obs_size = obs_size
        self._comm_bandwidth = comm_bandwidth
        self._action_size = action_size_for_bandwidth(comm_bandwidth)
        self._run_id = run_id
        self._seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # --- Seed everything before creating networks so weights are reproducible.
        self._seed_everything(seed)

        # --- Directory layout
        self._results_dir = Path(results_dir)
        self._log_dir = self._results_dir / "logs" / run_id
        self._checkpoint_dir = self._results_dir / "checkpoints" / run_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-agent networks and optimizers
        self._actors: List[Actor] = []
        self._critics: List[Critic] = []
        self._target_actors: List[Actor] = []
        self._target_critics: List[Critic] = []
        self._actor_optims: List[optim.Adam] = []
        self._critic_optims: List[optim.Adam] = []

        critic_input_dim = expected_critic_input_dim(obs_size, self._action_size)

        for _ in range(config.num_agents):
            # Online networks
            actor = Actor(obs_size, self._action_size, config).to(self._device)
            critic = Critic(critic_input_dim, config).to(self._device)

            # Target networks — start as exact copies of the online networks
            target_actor = Actor(obs_size, self._action_size, config).to(self._device)
            target_critic = Critic(critic_input_dim, config).to(self._device)
            hard_update(target_actor, actor)
            hard_update(target_critic, critic)

            # Optimizers
            actor_optim = optim.Adam(actor.parameters(), lr=config.actor_lr)
            critic_optim = optim.Adam(critic.parameters(), lr=config.critic_lr)

            self._actors.append(actor)
            self._critics.append(critic)
            self._target_actors.append(target_actor)
            self._target_critics.append(target_critic)
            self._actor_optims.append(actor_optim)
            self._critic_optims.append(critic_optim)

        # --- Replay buffer (seeded independently for deterministic sampling)
        self._buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            num_agents=config.num_agents,
            obs_size=obs_size,
            action_size=self._action_size,
            seed=seed,
        )

        # --- TensorBoard writer
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

        # --- Training step counter (environment steps, not gradient updates)
        self._env_step = 0

        # --- Log startup config for reproducibility
        self._log_startup()

    # =================================================================
    # Seed management
    # =================================================================

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """
        Set torch, numpy, and python random seeds. Unity's seed is
        passed separately at env construction time in the run loop.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # =================================================================
    # Startup logging
    # =================================================================

    def _log_startup(self) -> None:
        """Write config and seed info as tensorboard text for later inspection."""
        cfg = self._config
        config_text = (
            f"run_id: {self._run_id}  \n"
            f"seed: {self._seed}  \n"
            f"device: {self._device}  \n"
            f"obs_size: {self._obs_size}  \n"
            f"action_size: {self._action_size}  \n"
            f"comm_bandwidth: {self._comm_bandwidth}  \n"
            f"num_agents: {cfg.num_agents}  \n"
            f"behavior_name: {cfg.behavior_name}  \n"
            f"actor_lr: {cfg.actor_lr}  \n"
            f"critic_lr: {cfg.critic_lr}  \n"
            f"gamma: {cfg.gamma}  \n"
            f"tau: {cfg.tau}  \n"
            f"batch_size: {cfg.batch_size}  \n"
            f"buffer_capacity: {cfg.replay_buffer_capacity}  \n"
            f"warmup: {cfg.warmup_transitions}  \n"
            f"max_steps: {cfg.max_steps}  \n"
            f"sigma: {cfg.sigma_start} -> {cfg.sigma_end} over {cfg.sigma_decay_steps}  \n"
            f"hidden_units: {cfg.hidden_units}  \n"
            f"num_hidden_layers: {cfg.num_hidden_layers}  \n"
        )
        self._writer.add_text("config", config_text, global_step=0)
        self._writer.flush()

        print(f"[MaddpgTrainer] run_id={self._run_id} seed={self._seed} device={self._device}")
        print(f"[MaddpgTrainer] obs_size={self._obs_size} action_size={self._action_size} "
              f"comm_bandwidth={self._comm_bandwidth}")
        print(f"[MaddpgTrainer] critic_input_dim={expected_critic_input_dim(self._obs_size, self._action_size)}")
        print(f"[MaddpgTrainer] log_dir={self._log_dir}")
        print(f"[MaddpgTrainer] checkpoint_dir={self._checkpoint_dir}")

    # =================================================================
    # Checkpointing
    # =================================================================

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """
        Save a full training checkpoint: all actors, critics, target
        networks, and optimizer states.

        Args:
            tag: optional suffix. If None, uses the current env step.
        Returns:
            Path to the saved .pt file.
        """
        if tag is None:
            tag = f"step{self._env_step}"
        path = self._checkpoint_dir / f"checkpoint_{tag}.pt"

        state = {
            "env_step": self._env_step,
            "config": {
                "obs_size": self._obs_size,
                "action_size": self._action_size,
                "comm_bandwidth": self._comm_bandwidth,
                "num_agents": self._config.num_agents,
            },
            "actors": [a.state_dict() for a in self._actors],
            "critics": [c.state_dict() for c in self._critics],
            "target_actors": [a.state_dict() for a in self._target_actors],
            "target_critics": [c.state_dict() for c in self._target_critics],
            "actor_optims": [o.state_dict() for o in self._actor_optims],
            "critic_optims": [o.state_dict() for o in self._critic_optims],
        }
        torch.save(state, path)
        print(f"[MaddpgTrainer] checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load a full training checkpoint, restoring all network weights,
        optimizer states, and env_step counter.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        state = torch.load(path, map_location=self._device)

        # Sanity: make sure the checkpoint matches our current config.
        saved = state["config"]
        if saved["obs_size"] != self._obs_size:
            raise ValueError(
                f"checkpoint obs_size {saved['obs_size']} != trainer {self._obs_size}"
            )
        if saved["action_size"] != self._action_size:
            raise ValueError(
                f"checkpoint action_size {saved['action_size']} != trainer {self._action_size}"
            )
        if saved["num_agents"] != self._config.num_agents:
            raise ValueError(
                f"checkpoint num_agents {saved['num_agents']} != trainer {self._config.num_agents}"
            )

        for i in range(self._config.num_agents):
            self._actors[i].load_state_dict(state["actors"][i])
            self._critics[i].load_state_dict(state["critics"][i])
            self._target_actors[i].load_state_dict(state["target_actors"][i])
            self._target_critics[i].load_state_dict(state["target_critics"][i])
            self._actor_optims[i].load_state_dict(state["actor_optims"][i])
            self._critic_optims[i].load_state_dict(state["critic_optims"][i])

        self._env_step = state["env_step"]
        print(f"[MaddpgTrainer] checkpoint loaded from {path} at env_step={self._env_step}")

    # =================================================================
    # Update step
    # =================================================================
    @staticmethod
    def _resolve_behavior_name(configured: str, available: List[str]) -> str:
        """
        Resolve the configured behavior name against what Unity actually
        reports. ML-Agents appends '?team=N' to behavior names when agents
        are registered in a MultiAgentGroup, so the literal name from
        config.py won't match.

        Matching strategy:
          1. Exact match — preferred.
          2. Match on the prefix before '?'. If exactly one available
             behavior starts with '<configured>?', use it.
          3. Otherwise, raise with a helpful list.
        """
        if configured in available:
            return configured

        prefix = configured + "?"
        candidates = [name for name in available if name.startswith(prefix)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Behavior name '{configured}' is ambiguous — "
                f"multiple matches: {candidates}"
            )

        raise RuntimeError(
            f"Behavior '{configured}' not found. Available: {available}"
        )
    
    def _build_critic_input_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensor-native version of critic_input.build_critic_input.

        Mirrors the fixed-index slicing used in the numpy version. The
        numpy version remains the canonical reference for single-step
        rollout action selection in run(); this version is for batched
        gradient updates.

        Args:
            obs:     (batch, num_agents, obs_size)
            actions: (batch, num_agents, action_size)
        Returns:
            (batch, critic_input_dim) flat critic input, same layout as
            build_critic_input: [shared(4), local_a0, local_a1, local_a2,
            act_a0, act_a1, act_a2].
        """
        num_agents = obs.shape[1]
        obs_size = obs.shape[2]
        is_full_obs = (obs_size == FULL_OBS_SIZE)

        # Shared features extracted from agent 0 (identical across agents).
        belt_speed = obs[:, 0, IDX_BELT_SPEED : IDX_BELT_SPEED + 1]   # (B, 1)
        if is_full_obs:
            congestion = obs[:, 0, IDX_CONGESTION_START_FULL:IDX_CONGESTION_END_FULL]  # (B, 3)
        else:
            # CRITICAL: use fixed indices, not obs[:, 0, -3:] — the latter
            # would grab message floats in comm variants instead of congestion.
            congestion = obs[:, 0, IDX_CONGESTION_START_PARTIAL:IDX_CONGESTION_END_PARTIAL]
        shared = torch.cat([belt_speed, congestion], dim=1)  # (B, 4)

        # Per-agent local blocks.
        local_parts = []
        for i in range(num_agents):
            gate_cooldown = obs[:, i, IDX_GATE_STATE : IDX_COOLDOWN + 1]          # (B, 2)
            dest_mapping = obs[:, i, IDX_DEST_MAPPING_START:IDX_PACKAGE_SLOTS_START]  # (B, 3)
            packages = obs[:, i, IDX_PACKAGE_SLOTS_START:IDX_PACKAGE_SLOTS_END]   # (B, 25)
            parts = [gate_cooldown, dest_mapping, packages]

            if is_full_obs:
                peers = obs[:, i, IDX_OTHER_AGENTS_START_FULL:IDX_OTHER_AGENTS_END_FULL]
                parts.append(peers)
            else:
                # Messages (if any) live AFTER congestion in partial-obs mode.
                # Slice is empty for obs_size=34, 2 floats for 36, 6 floats for 40.
                messages = obs[:, i, IDX_CONGESTION_END_PARTIAL:]
                if messages.shape[1] > 0:
                    parts.append(messages)

            local_parts.append(torch.cat(parts, dim=1))

        # Per-agent action blocks.
        action_parts = [actions[:, i, :] for i in range(num_agents)]

        return torch.cat([shared] + local_parts + action_parts, dim=1)

    def _update(self) -> Optional[Dict[str, float]]:
        """
        One MADDPG gradient update across all agents.

        For each agent i:
          1. Target Q = r_i + gamma * (1 - done_i) * target_critic_i(
                 next_obs, target_actors(next_obs))
          2. Current Q = critic_i(obs, masked_actions_from_buffer)
          3. Critic loss = MSE(current_Q, target_Q); step critic optim.
          4. Actor loss = -mean(critic_i(obs, [masked_others, actor_i(obs_i)]))
             — replace agent i's action with its CURRENT actor output while
             keeping the other agents' buffer actions. Step actor optim.
          5. Soft-update target_actor_i and target_critic_i with tau.

        Returns a dict of per-step metrics for TensorBoard logging, or None
        if the buffer has not yet reached warmup_transitions.
        """
        if not self._buffer.is_ready(self._config.warmup_transitions):
            return None

        batch = self._buffer.sample(self._config.batch_size)

        # Move batch to device. Replay buffer stores float32; torch defaults match.
        obs = torch.from_numpy(batch.obs).to(self._device)                      # (B, N, obs_size)
        masked_actions = torch.from_numpy(batch.masked_actions).to(self._device)  # (B, N, A)
        rewards = torch.from_numpy(batch.rewards).to(self._device)              # (B, N)
        next_obs = torch.from_numpy(batch.next_obs).to(self._device)            # (B, N, obs_size)
        dones = torch.from_numpy(batch.dones).to(self._device)                  # (B, N)

        num_agents = self._config.num_agents

        # Precompute all target actor outputs for next_obs — shared across agents.
        with torch.no_grad():
            next_actions = torch.zeros_like(masked_actions)
            for i in range(num_agents):
                next_actions[:, i, :] = self._target_actors[i](next_obs[:, i, :])
            next_critic_input = self._build_critic_input_batch(next_obs, next_actions)

        critic_losses = []
        actor_losses = []
        mean_qs = []

        for i in range(num_agents):
            # ===== Critic update =====
            with torch.no_grad():
                target_q_next = self._target_critics[i](next_critic_input).squeeze(-1)  # (B,)
                target_q = (
                    rewards[:, i]
                    + self._config.gamma * (1.0 - dones[:, i]) * target_q_next
                )

            current_critic_input = self._build_critic_input_batch(obs, masked_actions)
            current_q = self._critics[i](current_critic_input).squeeze(-1)  # (B,)

            critic_loss = F.mse_loss(current_q, target_q)
            self._critic_optims[i].zero_grad()
            critic_loss.backward()
            self._critic_optims[i].step()

            critic_losses.append(critic_loss.item())
            mean_qs.append(current_q.mean().item())

            # ===== Actor update =====
            # Rebuild action tensor: agent i uses current actor output (differentiable),
            # other agents use masked actions from buffer (no grad).
            my_action = self._actors[i](obs[:, i, :])  # (B, A), requires grad
            action_slices = []
            for j in range(num_agents):
                if j == i:
                    action_slices.append(my_action)
                else:
                    action_slices.append(masked_actions[:, j, :])
            actions_for_actor = torch.stack(action_slices, dim=1)  # (B, N, A)

            actor_critic_input = self._build_critic_input_batch(obs, actions_for_actor)
            actor_loss = -self._critics[i](actor_critic_input).mean()

            self._actor_optims[i].zero_grad()
            actor_loss.backward()
            self._actor_optims[i].step()

            actor_losses.append(actor_loss.item())

            # ===== Soft update target networks =====
            soft_update(self._target_actors[i], self._actors[i], self._config.tau)
            soft_update(self._target_critics[i], self._critics[i], self._config.tau)

        return {
            "critic_loss_mean": float(np.mean(critic_losses)),
            "actor_loss_mean": float(np.mean(actor_losses)),
            "mean_q": float(np.mean(mean_qs)),
        }


    # =================================================================
    # Main training loop
    # =================================================================

    def run(
        self,
        env_path: Optional[str] = None,
        no_graphics: bool = True,
    ) -> None:
        """
        Connect to Unity and run the training loop until max_steps.

        Args:
            env_path: path to a standalone Unity build. If None, connects
                to the Unity Editor (which must be in Play mode when this
                method is called).
            no_graphics: passed to UnityEnvironment. Has no effect when
                connecting to the Editor; only applies to standalone builds.
        """
        # Lazy import so the module stays importable in unit tests
        # that don't need mlagents_envs.
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.base_env import ActionTuple

        configured_name = self._config.behavior_name
        print(f"[MaddpgTrainer] connecting to Unity (behavior='{configured_name}')...")

        env_kwargs: dict = {"seed": self._seed}
        if env_path is not None:
            env_kwargs["file_name"] = env_path
            env_kwargs["no_graphics"] = no_graphics
        env = UnityEnvironment(**env_kwargs)

        try:
            env.reset()

            available = list(env.behavior_specs.keys())
            behavior_name = self._resolve_behavior_name(configured_name, available)
            spec = env.behavior_specs[behavior_name]
            print(f"[MaddpgTrainer] resolved behavior name: '{behavior_name}'")
            actual_obs_shape = spec.observation_specs[0].shape
            expected_obs_shape = (self._obs_size,)
            if actual_obs_shape != expected_obs_shape:
                raise RuntimeError(
                    f"Observation shape mismatch: trainer expects "
                    f"{expected_obs_shape}, Unity reports {actual_obs_shape}. "
                    f"Check BehaviorParameters vector obs size in Unity."
                )
            print(f"[MaddpgTrainer] connected. obs_shape={actual_obs_shape}, "
                  f"action_spec={spec.action_spec}")

            # ---- Rolling metrics
            episode_rewards: List[float] = []
            current_episode_reward = 0.0
            illegal_action_count = 0
            total_action_count = 0
            terminal_transitions_stored = 0  # sanity counter

            # ---- Pending transition state
            prev_obs: Optional[np.ndarray] = None
            prev_actor_outputs: Optional[np.ndarray] = None
            prev_masked_actions: Optional[np.ndarray] = None

            start_time = time.time()
            last_log_step = 0

            print(f"[MaddpgTrainer] starting training, max_steps={self._config.max_steps}")

            while self._env_step < self._config.max_steps:
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # ---- Terminal agents: close the pending transition with done=True
                if len(terminal_steps) > 0 and prev_obs is not None:
                    terminal_obs = np.zeros(
                        (self._config.num_agents, self._obs_size), dtype=np.float32
                    )
                    terminal_rewards = np.zeros(self._config.num_agents, dtype=np.float32)

                    agent_ids = list(terminal_steps.agent_id)
                    for slot_idx in range(min(len(agent_ids), self._config.num_agents)):
                        agent_id = agent_ids[slot_idx]
                        terminal_obs[slot_idx] = terminal_steps[agent_id].obs[0]
                        terminal_rewards[slot_idx] = terminal_steps[agent_id].reward

                    dones = np.ones(self._config.num_agents, dtype=np.float32)
                    self._buffer.store(
                        prev_obs, prev_actor_outputs, prev_masked_actions,
                        terminal_rewards, terminal_obs, dones,
                    )
                    terminal_transitions_stored += 1

                    current_episode_reward += float(terminal_rewards[0])
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0.0

                    prev_obs = None
                    prev_actor_outputs = None
                    prev_masked_actions = None

                # ---- Decision agents: complete prior transition, select new actions
                if len(decision_steps) > 0:
                    current_obs = np.zeros(
                        (self._config.num_agents, self._obs_size), dtype=np.float32
                    )
                    current_rewards = np.zeros(self._config.num_agents, dtype=np.float32)

                    agent_ids = list(decision_steps.agent_id)
                    n_present = min(len(agent_ids), self._config.num_agents)
                    for slot_idx in range(n_present):
                        agent_id = agent_ids[slot_idx]
                        current_obs[slot_idx] = decision_steps[agent_id].obs[0]
                        current_rewards[slot_idx] = decision_steps[agent_id].reward

                    # Complete pending (non-terminal) transition
                    if prev_obs is not None:
                        dones = np.zeros(self._config.num_agents, dtype=np.float32)
                        self._buffer.store(
                            prev_obs, prev_actor_outputs, prev_masked_actions,
                            current_rewards, current_obs, dones,
                        )
                        current_episode_reward += float(current_rewards[0])

                    # Select actions: actor forward + Gaussian noise, clip to [0,1]
                    sigma = self._config.sigma_at_step(self._env_step)
                    actor_outputs = np.zeros(
                        (self._config.num_agents, self._action_size), dtype=np.float32
                    )
                    with torch.no_grad():
                        for i in range(self._config.num_agents):
                            obs_tensor = torch.from_numpy(current_obs[i]).to(self._device)
                            a = self._actors[i](obs_tensor).cpu().numpy()
                            noise = np.random.normal(0.0, sigma, size=a.shape).astype(np.float32)
                            actor_outputs[i] = np.clip(a + noise, 0.0, 1.0)

                    # Python-side masking: force gate action to 0 when gate not retracted.
                    masked_actions = actor_outputs.copy()
                    for i in range(self._config.num_agents):
                        gate_state = current_obs[i, IDX_GATE_STATE]
                        if gate_state != 0.0:
                            masked_actions[i, 0] = 0.0
                            if actor_outputs[i, 0] > 0.5:
                                illegal_action_count += 1
                        total_action_count += 1

                    # Send to Unity
                    action_tuple = ActionTuple(continuous=masked_actions.astype(np.float32))
                    env.set_actions(behavior_name, action_tuple)

                    # Remember as pending for the next iteration
                    prev_obs = current_obs
                    prev_actor_outputs = actor_outputs
                    prev_masked_actions = masked_actions

                    self._env_step += 1

                    # Gradient update (returns None until warmup reached)
                    update_metrics = self._update()

                    # Periodic logging
                    if self._env_step - last_log_step >= self._config.summary_freq:
                        elapsed = time.time() - start_time
                        sps = self._env_step / max(elapsed, 1e-6)
                        recent = episode_rewards[-20:] if episode_rewards else [0.0]
                        mean_ep_reward = float(np.mean(recent))
                        illegal_rate = (
                            illegal_action_count / max(total_action_count, 1)
                        )

                        self._writer.add_scalar(
                            "rollout/mean_episode_reward", mean_ep_reward, self._env_step
                        )
                        self._writer.add_scalar("rollout/sigma", sigma, self._env_step)
                        self._writer.add_scalar(
                            "rollout/illegal_action_rate", illegal_rate, self._env_step
                        )
                        self._writer.add_scalar(
                            "rollout/buffer_size", self._buffer.size, self._env_step
                        )
                        self._writer.add_scalar(
                            "rollout/steps_per_sec", sps, self._env_step
                        )
                        self._writer.add_scalar(
                            "rollout/terminal_transitions_cum",
                            terminal_transitions_stored, self._env_step,
                        )

                        if update_metrics is not None:
                            for k, v in update_metrics.items():
                                self._writer.add_scalar(f"train/{k}", v, self._env_step)

                        self._writer.flush()
                        print(
                            f"[step {self._env_step:>8}/{self._config.max_steps}] "
                            f"ep_rew={mean_ep_reward:+7.3f} "
                            f"sigma={sigma:.3f} "
                            f"illegal={illegal_rate:6.1%} "
                            f"buf={self._buffer.size:>6} "
                            f"terms={terminal_transitions_stored:>4} "
                            f"sps={sps:5.0f}"
                        )

                        last_log_step = self._env_step
                        illegal_action_count = 0
                        total_action_count = 0

                    # Periodic checkpoint
                    if (
                        self._env_step > 0
                        and self._env_step % self._config.checkpoint_freq == 0
                    ):
                        self.save_checkpoint()

                # Advance the Unity simulation by one physics step
                env.step()

            print(f"[MaddpgTrainer] max_steps reached. terminal transitions stored: {terminal_transitions_stored}")
            self.save_checkpoint(tag="final")

        except KeyboardInterrupt:
            print(f"[MaddpgTrainer] interrupted at step {self._env_step}, saving checkpoint...")
            self.save_checkpoint(tag=f"interrupted_step{self._env_step}")
        finally:
            try:
                env.close()
                print(f"[MaddpgTrainer] Unity env closed")
            except Exception as e:
                print(f"[MaddpgTrainer] error closing env: {e}")

    # =================================================================
    # Cleanup
    # =================================================================

    def close(self) -> None:
        """Flush and close the tensorboard writer."""
        self._writer.flush()
        self._writer.close()


# =====================================================================
# Self-test
# =====================================================================

# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MaddpgTrainer._update() self-test")
    print("=" * 60)

    cfg = MaddpgConfig(
        replay_buffer_capacity=200,
        warmup_transitions=50,
        batch_size=32,
        max_steps=10_000,
        tau=0.1,  # larger than default for visible target drift in a single step
    )

    trainer = MaddpgTrainer(
        config=cfg,
        obs_size=38,
        comm_bandwidth=0,
        run_id="update_test",
        seed=42,
    )

    # ---- Fill buffer with 100 fake transitions
    rng = np.random.default_rng(42)
    for _ in range(100):
        obs = rng.random((3, 38)).astype(np.float32)
        actor_out = rng.random((3, 1)).astype(np.float32)
        masked = actor_out.copy()
        # Simulate masking: zero out action[0] where gate state (obs[0]) > 0.5
        for a in range(3):
            if obs[a, 0] > 0.5:
                masked[a, 0] = 0.0
        rewards = (rng.standard_normal(3) * 0.1).astype(np.float32)
        next_obs = rng.random((3, 38)).astype(np.float32)
        dones = (rng.random(3) < 0.1).astype(np.bool_)
        trainer._buffer.store(obs, actor_out, masked, rewards, next_obs, dones)

    assert trainer._buffer.size == 100
    print(f"buffer filled with 100 transitions OK")

    # ---- Capture initial weights for a sample network from each category
    actor0_before = trainer._actors[0]._trunk[0].weight.data.clone()
    critic0_before = trainer._critics[0]._trunk[0].weight.data.clone()
    target_actor0_before = trainer._target_actors[0]._trunk[0].weight.data.clone()
    target_critic0_before = trainer._target_critics[0]._trunk[0].weight.data.clone()

    # ---- Run one update
    losses = trainer._update()
    assert losses is not None, "_update returned None despite buffer being past warmup"
    print(f"_update returned losses: {losses}")

    # ---- Losses are finite
    assert all(np.isfinite(v) for v in losses.values()), f"non-finite loss: {losses}"
    print(f"all losses finite OK")

    # ---- Online networks actually changed
    assert not torch.allclose(actor0_before, trainer._actors[0]._trunk[0].weight.data), \
        "actor 0 weights did not change after _update"
    assert not torch.allclose(critic0_before, trainer._critics[0]._trunk[0].weight.data), \
        "critic 0 weights did not change after _update"
    print(f"online networks updated OK")

    # ---- Target networks drifted toward online but are not equal
    target_actor0_after = trainer._target_actors[0]._trunk[0].weight.data
    target_critic0_after = trainer._target_critics[0]._trunk[0].weight.data
    assert not torch.allclose(target_actor0_before, target_actor0_after), \
        "target actor 0 did not drift"
    assert not torch.allclose(target_critic0_before, target_critic0_after), \
        "target critic 0 did not drift"
    # With tau=0.1 and only one update, target should NOT equal online.
    assert not torch.allclose(target_actor0_after, trainer._actors[0]._trunk[0].weight.data), \
        "target actor became equal to online after 1 update (tau too large?)"
    print(f"target networks soft-updated OK")

    # ---- 20 more updates: stability / no explosions
    for k in range(20):
        losses = trainer._update()
        assert losses is not None
        assert all(np.isfinite(v) for v in losses.values()), \
            f"non-finite loss at update {k}: {losses}"
    print(f"20 additional updates: all losses finite OK")

    # ---- _update returns None while buffer is below warmup
    cfg2 = MaddpgConfig(
        replay_buffer_capacity=200,
        warmup_transitions=1000,
        batch_size=32,
        max_steps=10_000,
    )
    trainer2 = MaddpgTrainer(
        config=cfg2,
        obs_size=38,
        comm_bandwidth=0,
        run_id="update_warmup_test",
        seed=99,
    )
    for _ in range(10):
        obs = rng.random((3, 38)).astype(np.float32)
        zero_a = np.zeros((3, 1), dtype=np.float32)
        r = np.zeros(3, dtype=np.float32)
        trainer2._buffer.store(
            obs, zero_a, zero_a, r, obs, np.zeros(3, dtype=np.bool_)
        )
    result = trainer2._update()
    assert result is None, f"_update should return None below warmup, got {result}"
    print(f"_update returns None below warmup OK")

    # ---- Critic input dim sanity: tensor version matches numpy expected dim
    fake_obs = torch.zeros(4, 3, 38)
    fake_act = torch.zeros(4, 3, 1)
    ci = trainer._build_critic_input_batch(fake_obs, fake_act)
    assert ci.shape == (4, 109), f"critic input batch shape {ci.shape} != (4, 109)"
    print(f"_build_critic_input_batch matches expected dim 109 OK")

    # Also test partial-obs comm3 variant (obs=40, action=4, dim=124)
    trainer3 = MaddpgTrainer(
        config=MaddpgConfig(replay_buffer_capacity=10, warmup_transitions=1, batch_size=1),
        obs_size=40,
        comm_bandwidth=3,
        run_id="update_comm3_dim_test",
        seed=7,
    )
    fake_obs40 = torch.zeros(2, 3, 40)
    fake_act4 = torch.zeros(2, 3, 4)
    ci40 = trainer3._build_critic_input_batch(fake_obs40, fake_act4)
    assert ci40.shape == (2, 124), f"critic input batch shape {ci40.shape} != (2, 124)"
    print(f"_build_critic_input_batch comm3 variant dim 124 OK")

    trainer.close()
    trainer2.close()
    trainer3.close()
    # ---- run() method signature check (does NOT actually run — needs Unity)
    import inspect
    assert hasattr(MaddpgTrainer, "run"), "run() method missing"
    sig = inspect.signature(MaddpgTrainer.run)
    params = list(sig.parameters.keys())
    assert "env_path" in params, "run() missing env_path arg"
    assert "no_graphics" in params, "run() missing no_graphics arg"
    print(f"run() method present with signature: {sig}")
    print()
    print("maddpg_trainer.py (with _update): all self-tests passed.")