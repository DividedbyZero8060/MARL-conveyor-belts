"""
MADDPG hyperparameters and constants.

This module is the single source of truth for every tunable in the
MADDPG trainer. Other modules (networks, replay_buffer, critic_input,
maddpg_trainer) import from here. Do NOT hardcode any of these values
elsewhere.

Two groups of constants:

  1. HYPERPARAMETERS — learning rates, batch size, exploration schedule,
     network architecture. Tune these.

  2. OBSERVATION INDEX MIRROR — exact replicas of the constants in
     Assets/Scripts/Agents/ObsIndices.cs. Used by critic_input.py to
     slice the observation vector with fixed indices. DO NOT change
     these without updating the C# file in lockstep — desync will
     silently corrupt the centralised critic input.
"""

from dataclasses import dataclass
from typing import Tuple


# =====================================================================
# Hyperparameters
# =====================================================================

@dataclass(frozen=True)
class MaddpgConfig:
    """Immutable hyperparameter bundle. Pass into trainer constructor."""

    # ---- Learning rates ----
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    # ---- Discount and target update ----
    gamma: float = 0.99            # Discount factor
    tau: float = 0.01              # Soft target update rate

    # ---- Batch and buffer ----
    batch_size: int = 256
    replay_buffer_capacity: int = 500_000
    warmup_transitions: int = 10_000   # Min buffer fill before updates begin

    # ---- Network architecture ----
    hidden_units: int = 256
    num_hidden_layers: int = 2

    # ---- Exploration (Gaussian noise on actor output) ----
    sigma_start: float = 0.20
    sigma_end: float = 0.05
    sigma_decay_steps: int = 500_000

    # ---- Training schedule ----
    max_steps: int = 2_000_000
    update_every_n_steps: int = 1      # Per-step updates after warmup

    # ---- Logging cadence (in environment steps) ----
    summary_freq: int = 5_000
    checkpoint_freq: int = 100_000

    # ---- Multi-agent constants (fixed for this project) ----
    num_agents: int = 3
    behavior_name: str = "SortingAgent"

    def sigma_at_step(self, step: int) -> float:
        """Linearly decay exploration sigma from sigma_start to sigma_end."""
        if step >= self.sigma_decay_steps:
            return self.sigma_end
        frac = step / self.sigma_decay_steps
        return self.sigma_start + frac * (self.sigma_end - self.sigma_start)


# Default config used when no override is provided.
DEFAULT_CONFIG = MaddpgConfig()


# =====================================================================
# Observation index mirror — MUST stay in sync with ObsIndices.cs
# =====================================================================
#
# Layout (full observability, 38 floats):
#   [0]      gate_state                       (local)
#   [1]      cooldown                         (local)
#   [2]      belt_speed                       (shared)
#   [3..6)   dest_mapping one-hot             (local, THIS branch)
#   [6..31)  5 package slots × 5 floats       (local)
#   [31..35) peer features (full-obs only)    (local)
#   [35..38) congestion × 3                   (shared)
#
# Layout (partial observability, 34 floats):
#   [0..6)   same as above                    (local)
#   [6..31)  package slots                    (local)
#   [31..34) congestion × 3                   (shared)
#   [34..]   message floats appended (comm variants)
#
# CRITICAL: in comm variants the obs vector grows AFTER congestion.
# Use these fixed constants for slicing — do NOT use negative indexing
# like obs[-3:], which would grab message floats instead of congestion
# when comm_bandwidth > 0.

IDX_GATE_STATE: int = 0
IDX_COOLDOWN: int = 1
IDX_BELT_SPEED: int = 2

IDX_DEST_MAPPING_START: int = 3
IDX_DEST_MAPPING_END: int = 6        # exclusive

IDX_PACKAGE_SLOTS_START: int = 6
IDX_PACKAGE_SLOTS_END: int = 31      # exclusive
PACKAGE_SLOT_COUNT: int = 5
PACKAGE_SLOT_WIDTH: int = 5

# Full-obs only: 4-float peer feature block at indices 31..35
IDX_OTHER_AGENTS_START_FULL: int = 31
IDX_OTHER_AGENTS_END_FULL: int = 35  # exclusive

# Congestion lives at different indices depending on observability mode.
IDX_CONGESTION_START_FULL: int = 35
IDX_CONGESTION_END_FULL: int = 38    # exclusive

IDX_CONGESTION_START_PARTIAL: int = 31
IDX_CONGESTION_END_PARTIAL: int = 34 # exclusive

CONGESTION_WIDTH: int = 3

FULL_OBS_SIZE: int = 38
PARTIAL_OBS_SIZE: int = 34


# =====================================================================
# Action space sizes per comm bandwidth (Step 17 reuses these)
# =====================================================================
#
# Action layout (continuous): [gate, msg1, msg2, ...]
# Index 0 is always the gate action; remaining are message floats.

ACTION_SIZES: Tuple[int, ...] = (1, 2, 4)  # bandwidth 0, 1, 3

def action_size_for_bandwidth(bandwidth: int) -> int:
    """Return the actor output dimension for a given comm bandwidth."""
    if bandwidth == 0:
        return 1
    if bandwidth == 1:
        return 2
    if bandwidth == 3:
        return 4
    raise ValueError(
        f"Unsupported comm bandwidth {bandwidth}. Allowed: 0, 1, 3."
    )


# =====================================================================
# Self-test — run `python -m training.maddpg.config` to verify
# =====================================================================

if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    print(f"actor_lr={cfg.actor_lr}, critic_lr={cfg.critic_lr}")
    print(f"buffer={cfg.replay_buffer_capacity}, batch={cfg.batch_size}")
    print(f"sigma_start={cfg.sigma_start}, sigma_end={cfg.sigma_end}")

    # Sigma decay sanity
    assert abs(cfg.sigma_at_step(0) - cfg.sigma_start) < 1e-9
    assert abs(cfg.sigma_at_step(cfg.sigma_decay_steps) - cfg.sigma_end) < 1e-9
    assert abs(cfg.sigma_at_step(cfg.sigma_decay_steps * 2) - cfg.sigma_end) < 1e-9
    midpoint = cfg.sigma_at_step(cfg.sigma_decay_steps // 2)
    expected_mid = (cfg.sigma_start + cfg.sigma_end) / 2
    assert abs(midpoint - expected_mid) < 1e-6, f"midpoint {midpoint} != {expected_mid}"
    print(f"sigma decay OK: 0→{cfg.sigma_at_step(0):.4f}, "
          f"mid→{midpoint:.4f}, end→{cfg.sigma_at_step(cfg.sigma_decay_steps):.4f}")

    # Index sanity
    assert IDX_PACKAGE_SLOTS_END - IDX_PACKAGE_SLOTS_START == \
           PACKAGE_SLOT_COUNT * PACKAGE_SLOT_WIDTH
    assert IDX_CONGESTION_END_PARTIAL - IDX_CONGESTION_START_PARTIAL == CONGESTION_WIDTH
    assert IDX_CONGESTION_END_FULL - IDX_CONGESTION_START_FULL == CONGESTION_WIDTH
    assert FULL_OBS_SIZE == IDX_CONGESTION_END_FULL
    assert PARTIAL_OBS_SIZE == IDX_CONGESTION_END_PARTIAL
    print(f"obs sizes OK: full={FULL_OBS_SIZE}, partial={PARTIAL_OBS_SIZE}")

    # Action size sanity
    assert action_size_for_bandwidth(0) == 1
    assert action_size_for_bandwidth(1) == 2
    assert action_size_for_bandwidth(3) == 4
    try:
        action_size_for_bandwidth(2)
        assert False, "should have raised"
    except ValueError:
        pass
    print(f"action sizes OK: 0→1, 1→2, 3→4")

    print("config.py: all self-tests passed.")