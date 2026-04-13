"""
Independent DQN trainer for the Step 15b baseline.

Three independent single-agent Q-learners sharing one Unity environment.
Each agent has its own QNetwork, target network, replay buffer, and
epsilon schedule. No centralisation, no communication, no shared state.

The defining property of this baseline is INDEPENDENCE — it exists to
demonstrate that naive per-agent Q-learning underperforms cooperative
methods (MA-POCA, MATD3) on a non-stationary multi-agent task with
selfish reward.

Reads per-agent rewards from decision_steps.reward / terminal_steps.reward.
In Unity, IndependentRewardDistributor must be enabled (and the cooperative
RewardDistributor disabled) so those rewards reflect selfish per-agent
credit assignment. The trainer does not know or care about this — it
just reads whatever Unity sends.
"""

from __future__ import annotations

import logging
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

from training.dqn.config import DqnConfig, IDX_GATE_STATE
from training.dqn.q_network import QNetwork, hard_update
from training.dqn.replay_buffer import PerAgentReplayBuffer

# Suppress mlagents_envs StatsSideChannel "unknown channel" warnings.
# Same rationale as the MADDPG trainer: the StatsRecorder channel carries
# C# custom telemetry that this trainer doesn't consume. Wiring it up is
# a deferred task.
logging.getLogger("mlagents_envs.side_channel.side_channel_manager").setLevel(logging.ERROR)


class DqnTrainer:
    """
    Independent per-agent DQN.

    Each call to _update() samples a batch from each agent's buffer,
    computes the DQN target (`reward + gamma * max_a' Q_target(next_obs, a')`)
    with action masking, and steps that agent's optimiser. Target networks
    are hard-copied every target_update_freq environment steps.
    """

    def __init__(
        self,
        config: DqnConfig,
        obs_size: int,
        run_id: str,
        seed: int,
        results_dir: str = "results",
        device: Optional[str] = None,
    ):
        self._config = config
        self._obs_size = obs_size
        self._run_id = run_id
        self._seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._seed_everything(seed)

        self._results_dir = Path(results_dir)
        self._log_dir = self._results_dir / "logs" / run_id
        self._checkpoint_dir = self._results_dir / "checkpoints" / run_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-agent networks, target networks, optimizers, buffers
        self._q_nets: List[QNetwork] = []
        self._target_nets: List[QNetwork] = []
        self._optims: List[optim.Adam] = []
        self._buffers: List[PerAgentReplayBuffer] = []

        for i in range(config.num_agents):
            online = QNetwork(obs_size, config.num_actions, config).to(self._device)
            target = QNetwork(obs_size, config.num_actions, config).to(self._device)
            hard_update(target, online)
            opt = optim.Adam(online.parameters(), lr=config.learning_rate)
            # Each agent's buffer gets a distinct seed so they're independent
            # but reproducible.
            buf = PerAgentReplayBuffer(
                capacity=config.replay_buffer_capacity,
                obs_size=obs_size,
                num_actions=config.num_actions,
                seed=seed + i + 1,
            )
            self._q_nets.append(online)
            self._target_nets.append(target)
            self._optims.append(opt)
            self._buffers.append(buf)

        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        self._env_step = 0
        self._log_startup()

    # =================================================================
    # Seed management
    # =================================================================

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # =================================================================
    # Startup logging
    # =================================================================

    def _log_startup(self) -> None:
        cfg = self._config
        config_text = (
            f"run_id: {self._run_id}  \n"
            f"seed: {self._seed}  \n"
            f"device: {self._device}  \n"
            f"obs_size: {self._obs_size}  \n"
            f"num_actions: {cfg.num_actions}  \n"
            f"num_agents: {cfg.num_agents}  \n"
            f"behavior_name: {cfg.behavior_name}  \n"
            f"learning_rate: {cfg.learning_rate}  \n"
            f"gamma: {cfg.gamma}  \n"
            f"batch_size: {cfg.batch_size}  \n"
            f"buffer_capacity (per agent): {cfg.replay_buffer_capacity}  \n"
            f"warmup (per agent): {cfg.warmup_transitions}  \n"
            f"max_steps: {cfg.max_steps}  \n"
            f"epsilon: {cfg.epsilon_start} -> {cfg.epsilon_end} "
            f"over {cfg.epsilon_decay_steps}  \n"
            f"target_update_freq: {cfg.target_update_freq}  \n"
            f"hidden_units: {cfg.hidden_units}  \n"
        )
        self._writer.add_text("config", config_text, global_step=0)
        self._writer.flush()

        print(f"[DqnTrainer] run_id={self._run_id} seed={self._seed} device={self._device}")
        print(f"[DqnTrainer] obs_size={self._obs_size} num_actions={self._config.num_actions}")
        print(f"[DqnTrainer] log_dir={self._log_dir}")
        print(f"[DqnTrainer] checkpoint_dir={self._checkpoint_dir}")

    # =================================================================
    # Checkpointing
    # =================================================================

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        if tag is None:
            tag = f"step{self._env_step}"
        path = self._checkpoint_dir / f"checkpoint_{tag}.pt"
        state = {
            "env_step": self._env_step,
            "config": {
                "obs_size": self._obs_size,
                "num_agents": self._config.num_agents,
                "num_actions": self._config.num_actions,
            },
            "q_nets": [n.state_dict() for n in self._q_nets],
            "target_nets": [n.state_dict() for n in self._target_nets],
            "optims": [o.state_dict() for o in self._optims],
        }
        torch.save(state, path)
        print(f"[DqnTrainer] checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        state = torch.load(path, map_location=self._device)
        saved = state["config"]
        if saved["obs_size"] != self._obs_size:
            raise ValueError(f"obs_size mismatch: {saved['obs_size']} vs {self._obs_size}")
        if saved["num_agents"] != self._config.num_agents:
            raise ValueError(f"num_agents mismatch: {saved['num_agents']} vs {self._config.num_agents}")
        for i in range(self._config.num_agents):
            self._q_nets[i].load_state_dict(state["q_nets"][i])
            self._target_nets[i].load_state_dict(state["target_nets"][i])
            self._optims[i].load_state_dict(state["optims"][i])
        self._env_step = state["env_step"]
        print(f"[DqnTrainer] checkpoint loaded from {path} at env_step={self._env_step}")

    # =================================================================
    # Behavior name resolution (shared with MADDPG)
    # =================================================================

    @staticmethod
    def _resolve_behavior_name(configured: str, available: List[str]) -> str:
        if configured in available:
            return configured
        prefix = configured + "?"
        candidates = [name for name in available if name.startswith(prefix)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f"Behavior '{configured}' is ambiguous — matches: {candidates}")
        raise RuntimeError(f"Behavior '{configured}' not found. Available: {available}")

    # =================================================================
    # Action selection with epsilon-greedy and masking
    # =================================================================

    def _select_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """
        Epsilon-greedy action selection for one agent with action masking.

        action_mask convention: True = DISABLED (ML-Agents). Masked actions
        receive Q = -1e9 (NOT 0.0) so that when all unmasked Q-values are
        negative, the masked action is still not picked.
        """
        legal = ~action_mask  # True = legal
        # Defensive: if every action is masked (should never happen with a
        # 2-action branch where action 0 is always legal), fall back to 0.
        if not np.any(legal):
            return 0

        if random.random() < epsilon:
            # Uniform random over LEGAL actions only.
            legal_indices = np.flatnonzero(legal)
            return int(random.choice(legal_indices))

        # Greedy: argmax over masked Q-values.
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(self._device)
            q = self._q_nets[agent_idx](obs_t).cpu().numpy()  # (num_actions,)
        # Set masked Q-values to -1e9 (NOT 0.0 — the workflow is explicit about this).
        q_masked = q.copy()
        q_masked[action_mask] = -1e9
        return int(np.argmax(q_masked))

    # =================================================================
    # Per-agent gradient update
    # =================================================================

    def _update_agent(self, agent_idx: int) -> Optional[Dict[str, float]]:
        """
        One DQN gradient update for a single agent.

        Returns None if the buffer has not yet reached warmup.
        """
        buf = self._buffers[agent_idx]
        if not buf.is_ready(self._config.warmup_transitions):
            return None

        batch = buf.sample(self._config.batch_size)

        obs = torch.from_numpy(batch.obs).to(self._device)
        actions = torch.from_numpy(batch.actions).to(self._device)
        rewards = torch.from_numpy(batch.rewards).to(self._device)
        next_obs = torch.from_numpy(batch.next_obs).to(self._device)
        next_masks = torch.from_numpy(batch.next_action_masks).to(self._device)  # bool
        dones = torch.from_numpy(batch.dones).to(self._device)

        # ===== Target Q =====
        with torch.no_grad():
            next_q = self._target_nets[agent_idx](next_obs)  # (B, num_actions)
            # Mask out illegal next-step actions by setting their Q to -1e9.
            neg_inf = torch.full_like(next_q, -1e9)
            next_q_masked = torch.where(next_masks, neg_inf, next_q)
            max_next_q, _ = next_q_masked.max(dim=1)  # (B,)
            target_q = rewards + self._config.gamma * (1.0 - dones) * max_next_q

        # ===== Current Q (for the action actually taken) =====
        current_q_all = self._q_nets[agent_idx](obs)  # (B, num_actions)
        # Gather Q for the taken action.
        current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Huber loss (smooth L1) — same rationale as MATD3.
        loss = F.smooth_l1_loss(current_q, target_q)

        self._optims[agent_idx].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_nets[agent_idx].parameters(), max_norm=1.0)
        self._optims[agent_idx].step()

        return {
            "q_loss": float(loss.item()),
            "mean_q": float(current_q.mean().item()),
        }

    def _update(self) -> Optional[Dict[str, float]]:
        """Update all agents. Returns averaged metrics, or None if any is pre-warmup."""
        metrics_per_agent = []
        for i in range(self._config.num_agents):
            m = self._update_agent(i)
            if m is None:
                return None
            metrics_per_agent.append(m)

        # Hard-update target networks every target_update_freq env steps.
        if self._env_step % self._config.target_update_freq == 0:
            for i in range(self._config.num_agents):
                hard_update(self._target_nets[i], self._q_nets[i])

        return {
            "q_loss_mean": float(np.mean([m["q_loss"] for m in metrics_per_agent])),
            "mean_q": float(np.mean([m["mean_q"] for m in metrics_per_agent])),
        }

    # =================================================================
    # Main training loop
    # =================================================================

    def run(
        self,
        env_path: Optional[str] = None,
        no_graphics: bool = True,
    ) -> None:
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.base_env import ActionTuple

        configured_name = self._config.behavior_name
        print(f"[DqnTrainer] connecting to Unity (behavior='{configured_name}')...")

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
            print(f"[DqnTrainer] resolved behavior name: '{behavior_name}'")

            actual_obs_shape = spec.observation_specs[0].shape
            expected_obs_shape = (self._obs_size,)
            if actual_obs_shape != expected_obs_shape:
                raise RuntimeError(
                    f"Observation shape mismatch: trainer expects "
                    f"{expected_obs_shape}, Unity reports {actual_obs_shape}."
                )
            print(f"[DqnTrainer] connected. obs_shape={actual_obs_shape}, action_spec={spec.action_spec}")

            # ---- Rolling metrics
            episode_rewards: List[float] = []
            current_episode_reward = 0.0
            terminal_transitions_stored = 0

            # ---- Pending per-agent transition state
            # For each agent, hold onto (prev_obs, prev_action, prev_mask) until
            # the next observation arrives, at which point we can store the
            # complete (s, a, m, r, s', m', done) tuple in that agent's buffer.
            num_agents = self._config.num_agents
            prev_obs: List[Optional[np.ndarray]] = [None] * num_agents
            prev_actions: List[int] = [0] * num_agents
            prev_masks: List[Optional[np.ndarray]] = [None] * num_agents

            start_time = time.time()
            last_log_step = 0

            print(f"[DqnTrainer] starting training, max_steps={self._config.max_steps}")

            while self._env_step < self._config.max_steps:
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # ---- Terminal agents: close pending transitions with done=True
                if len(terminal_steps) > 0:
                    agent_ids = list(terminal_steps.agent_id)
                    for slot_idx in range(min(len(agent_ids), num_agents)):
                        agent_id = agent_ids[slot_idx]
                        if prev_obs[slot_idx] is None:
                            continue  # no pending transition for this agent
                        terminal_obs = terminal_steps[agent_id].obs[0]
                        terminal_reward = terminal_steps[agent_id].reward
                        # At terminal, mask is irrelevant (done=True cuts the bootstrap),
                        # but store an all-False mask so the buffer format is consistent.
                        terminal_mask = np.zeros(self._config.num_actions, dtype=bool)
                        self._buffers[slot_idx].store(
                            prev_obs[slot_idx],
                            prev_actions[slot_idx],
                            prev_masks[slot_idx],
                            terminal_reward,
                            terminal_obs,
                            terminal_mask,
                            True,  # done
                        )
                        prev_obs[slot_idx] = None
                        prev_masks[slot_idx] = None

                    # Episode-level accounting (use agent 0's perspective for the
                    # rolling mean, since rewards are selfish and per-agent in DQN).
                    terminal_transitions_stored += 1
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0.0

                # ---- Decision agents: complete prior transitions, select new actions
                if len(decision_steps) > 0:
                    agent_ids = list(decision_steps.agent_id)
                    n_present = min(len(agent_ids), num_agents)

                    # Collect all current obs / rewards / masks first.
                    current_obs = np.zeros((num_agents, self._obs_size), dtype=np.float32)
                    current_rewards = np.zeros(num_agents, dtype=np.float32)
                    current_masks = np.zeros((num_agents, self._config.num_actions), dtype=bool)

                    for slot_idx in range(n_present):
                        agent_id = agent_ids[slot_idx]
                        current_obs[slot_idx] = decision_steps[agent_id].obs[0]
                        current_rewards[slot_idx] = decision_steps[agent_id].reward
                        # action_mask is a list of arrays, one per branch. Branch 0
                        # is our only branch (size 2). True = DISABLED.
                        masks = decision_steps[agent_id].action_mask
                        if masks is not None and len(masks) > 0:
                            current_masks[slot_idx] = masks[0].astype(bool)
                        # else: all False (no mask) — already zeros, fine.

                    # Complete pending transitions (reward from this tick, not terminal).
                    for slot_idx in range(n_present):
                        if prev_obs[slot_idx] is not None:
                            self._buffers[slot_idx].store(
                                prev_obs[slot_idx],
                                prev_actions[slot_idx],
                                prev_masks[slot_idx],
                                float(current_rewards[slot_idx]),
                                current_obs[slot_idx],
                                current_masks[slot_idx],
                                False,  # not done
                            )

                    # Agent-0 reward feeds the rolling episode mean.
                    current_episode_reward += float(current_rewards[0])

                    # Select new actions per agent.
                    epsilon = self._config.epsilon_at_step(self._env_step)
                    actions = np.zeros((num_agents, 1), dtype=np.int32)
                    for slot_idx in range(n_present):
                        a = self._select_action(
                            slot_idx,
                            current_obs[slot_idx],
                            current_masks[slot_idx],
                            epsilon,
                        )
                        actions[slot_idx, 0] = a
                        # Remember as pending.
                        prev_obs[slot_idx] = current_obs[slot_idx].copy()
                        prev_actions[slot_idx] = a
                        prev_masks[slot_idx] = current_masks[slot_idx].copy()

                    # Send to Unity. Discrete actions: shape (num_agents, num_branches).
                    action_tuple = ActionTuple(discrete=actions)
                    env.set_actions(behavior_name, action_tuple)

                    self._env_step += 1

                    # Gradient update
                    update_metrics = self._update()

                    # Periodic logging
                    if self._env_step - last_log_step >= self._config.summary_freq:
                        elapsed = time.time() - start_time
                        sps = self._env_step / max(elapsed, 1e-6)
                        recent = episode_rewards[-20:] if episode_rewards else [0.0]
                        mean_ep_reward = float(np.mean(recent))

                        min_buf_size = min(buf.size for buf in self._buffers)

                        self._writer.add_scalar(
                            "rollout/mean_episode_reward", mean_ep_reward, self._env_step
                        )
                        self._writer.add_scalar("rollout/epsilon", epsilon, self._env_step)
                        self._writer.add_scalar(
                            "rollout/buffer_size_min", min_buf_size, self._env_step
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
                            f"eps={epsilon:.3f} "
                            f"buf_min={min_buf_size:>6} "
                            f"terms={terminal_transitions_stored:>4} "
                            f"sps={sps:5.0f}"
                        )

                        last_log_step = self._env_step

                    # Periodic checkpoint
                    if (
                        self._env_step > 0
                        and self._env_step % self._config.checkpoint_freq == 0
                    ):
                        self.save_checkpoint()

                env.step()

            print(f"[DqnTrainer] max_steps reached. terminal transitions stored: {terminal_transitions_stored}")
            self.save_checkpoint(tag="final")

        except KeyboardInterrupt:
            print(f"[DqnTrainer] interrupted at step {self._env_step}, saving checkpoint...")
            self.save_checkpoint(tag=f"interrupted_step{self._env_step}")
        finally:
            try:
                env.close()
                print(f"[DqnTrainer] Unity env closed")
            except Exception as e:
                print(f"[DqnTrainer] error closing env: {e}")

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DqnTrainer self-test (offline, no Unity)")
    print("=" * 60)

    cfg = DqnConfig(
        replay_buffer_capacity=200,
        warmup_transitions=50,
        batch_size=32,
        max_steps=10_000,
        target_update_freq=100,
    )

    trainer = DqnTrainer(
        config=cfg,
        obs_size=38,
        run_id="dqn_offline_test",
        seed=42,
    )

    # ---- Fill each buffer with 100 synthetic transitions
    rng = np.random.default_rng(42)
    for _ in range(100):
        for agent_idx in range(3):
            obs = rng.random(38).astype(np.float32)
            action = int(rng.integers(0, 2))
            # Simulate an occasional mask: when gate_state > 0.5, action 1 is masked.
            mask = np.zeros(2, dtype=bool)
            if obs[0] > 0.5:
                mask[1] = True
            reward = float(rng.standard_normal() * 0.1)
            next_obs = rng.random(38).astype(np.float32)
            next_mask = np.zeros(2, dtype=bool)
            if next_obs[0] > 0.5:
                next_mask[1] = True
            done = bool(rng.random() < 0.1)
            trainer._buffers[agent_idx].store(obs, action, mask, reward, next_obs, next_mask, done)

    for agent_idx in range(3):
        assert trainer._buffers[agent_idx].size == 100, \
            f"agent {agent_idx} buffer size {trainer._buffers[agent_idx].size}"
    print(f"all 3 agent buffers filled to 100 OK")

    # ---- Capture pre-update weights
    before = [trainer._q_nets[i]._trunk[0].weight.data.clone() for i in range(3)]

    # ---- Run one update
    metrics = trainer._update()
    assert metrics is not None, "_update returned None despite warmup reached"
    assert np.isfinite(metrics["q_loss_mean"]), f"non-finite q_loss_mean: {metrics}"
    assert np.isfinite(metrics["mean_q"]), f"non-finite mean_q: {metrics}"
    print(f"_update metrics: {metrics}")

    # ---- Online networks changed
    for i in range(3):
        after = trainer._q_nets[i]._trunk[0].weight.data
        assert not torch.allclose(before[i], after), f"q_net {i} weights did not change"
    print(f"all 3 q_nets updated OK")

    # ---- 50 more updates: stability check, no NaN
    for k in range(50):
        m = trainer._update()
        assert m is not None
        assert np.isfinite(m["q_loss_mean"]), f"non-finite q_loss at update {k}"
    print(f"50 additional updates: all finite OK")

    # ---- Warmup gate: trainer 2 with warmup too high
    cfg2 = DqnConfig(
        replay_buffer_capacity=200,
        warmup_transitions=1000,
        batch_size=32,
        max_steps=10_000,
    )
    trainer2 = DqnTrainer(
        config=cfg2,
        obs_size=38,
        run_id="dqn_warmup_test",
        seed=99,
    )
    for _ in range(10):
        for a in range(3):
            trainer2._buffers[a].store(
                np.zeros(38, dtype=np.float32),
                0, np.zeros(2, dtype=bool),
                0.0,
                np.zeros(38, dtype=np.float32),
                np.zeros(2, dtype=bool),
                False,
            )
    result = trainer2._update()
    assert result is None, f"expected None pre-warmup, got {result}"
    print(f"_update returns None below warmup OK")

    # ---- Action selection with masking
    obs = np.random.randn(38).astype(np.float32)
    obs[0] = 0.8  # gate not retracted — action 1 should be masked
    mask = np.array([False, True])  # action 1 disabled

    # Greedy path (epsilon=0): must return action 0 even if Q(obs, 1) > Q(obs, 0)
    action = trainer._select_action(0, obs, mask, epsilon=0.0)
    assert action == 0, f"masked action 1 was selected anyway: {action}"
    print(f"greedy action selection respects mask OK")

    # Epsilon=1 path: always random, but only over legal actions
    for _ in range(50):
        a = trainer._select_action(0, obs, mask, epsilon=1.0)
        assert a == 0, f"random selection picked masked action: {a}"
    print(f"epsilon=1 random selection respects mask OK")

    # Unmasked path: either action is possible
    mask_open = np.array([False, False])
    seen = set()
    for _ in range(100):
        a = trainer._select_action(0, obs, mask_open, epsilon=1.0)
        seen.add(a)
    assert seen == {0, 1}, f"epsilon=1 unmasked should pick both actions, got {seen}"
    print(f"epsilon=1 unmasked covers both actions OK")

    # ---- Checkpoint save/load
    path = trainer.save_checkpoint(tag="offline_test")
    assert path.exists()
    trainer.load_checkpoint(str(path))
    print(f"checkpoint save/load OK")

    # ---- run() method signature
    import inspect
    sig = inspect.signature(DqnTrainer.run)
    params = list(sig.parameters.keys())
    assert "env_path" in params
    assert "no_graphics" in params
    print(f"run() signature OK: {sig}")

    trainer.close()
    trainer2.close()
    print()
    print("dqn_trainer.py: all self-tests passed.")