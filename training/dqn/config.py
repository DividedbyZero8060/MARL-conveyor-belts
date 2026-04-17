"""
Independent DQN hyperparameters and constants.

Reuses the observation index constants from training.maddpg.config so the
two trainers stay in lockstep — any change to the observation layout lands
in both places automatically.

DQN is intentionally simpler than MADDPG:
  - Smaller network (128 hidden units vs 256)
  - Per-agent replay buffers (not joint)
  - Epsilon-greedy exploration (not Gaussian noise)
  - Hard target network updates (not Polyak)
  - No centralised critic, no communication, no per-agent coupling

The simplicity is deliberate. This baseline represents "naively applying
single-agent Q-learning to a multi-agent problem" and is expected to
underperform MADDPG/MA-POCA due to non-stationarity and selfish rewards.
"""

from dataclasses import dataclass

# Re-export observation indices from the MADDPG config so both trainers
# use the exact same layout assumptions.
from training.maddpg.config import (
    IDX_GATE_STATE,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
)


# =====================================================================
# Hyperparameters
# =====================================================================

@dataclass(frozen=True)
class DqnConfig:
    """Immutable hyperparameter bundle. Pass into DqnTrainer constructor."""

    # ---- Learning ----
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # ---- Batch and buffer ----
    batch_size: int = 64
    replay_buffer_capacity: int = 50_000    # PER AGENT, not joint
    warmup_transitions: int = 2_000          # Per agent before updates begin

    # ---- Network architecture ----
    hidden_units: int = 128
    num_hidden_layers: int = 2

    # ---- Epsilon-greedy exploration ----
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500_000

    # ---- Target network hard-update frequency (env steps) ----
    target_update_freq: int = 1_000

    # ---- Training schedule ----
    max_steps: int = 2_000_000
    update_every_n_steps: int = 4

    # ---- Logging cadence (in environment steps) ----
    summary_freq: int = 5_000
    checkpoint_freq: int = 100_000

    # ---- Multi-agent constants (fixed for this project) ----
    num_agents: int = 3
    behavior_name: str = "SortingAgent"
    num_actions: int = 2  # discrete branch 0: [do nothing, activate]

    def epsilon_at_step(self, step: int) -> float:
        """Linearly decay epsilon from epsilon_start to epsilon_end."""
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = step / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


DEFAULT_CONFIG = DqnConfig()


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    print(f"learning_rate={cfg.learning_rate}, batch={cfg.batch_size}")
    print(f"buffer (per agent)={cfg.replay_buffer_capacity}, warmup={cfg.warmup_transitions}")
    print(f"epsilon: {cfg.epsilon_start} -> {cfg.epsilon_end} over {cfg.epsilon_decay_steps}")
    print(f"target_update_freq={cfg.target_update_freq}")

    # Epsilon decay sanity
    assert abs(cfg.epsilon_at_step(0) - cfg.epsilon_start) < 1e-9
    assert abs(cfg.epsilon_at_step(cfg.epsilon_decay_steps) - cfg.epsilon_end) < 1e-9
    assert abs(cfg.epsilon_at_step(cfg.epsilon_decay_steps * 2) - cfg.epsilon_end) < 1e-9
    midpoint = cfg.epsilon_at_step(cfg.epsilon_decay_steps // 2)
    expected_mid = (cfg.epsilon_start + cfg.epsilon_end) / 2
    assert abs(midpoint - expected_mid) < 1e-6, f"midpoint {midpoint} != {expected_mid}"
    print(f"epsilon decay OK: 0->{cfg.epsilon_at_step(0):.4f}, "
          f"mid->{midpoint:.4f}, end->{cfg.epsilon_at_step(cfg.epsilon_decay_steps):.4f}")

    print(f"obs index imports OK: IDX_GATE_STATE={IDX_GATE_STATE}, "
          f"FULL_OBS_SIZE={FULL_OBS_SIZE}, PARTIAL_OBS_SIZE={PARTIAL_OBS_SIZE}")

    print("config.py: all self-tests passed.")