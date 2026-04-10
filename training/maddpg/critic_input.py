"""
Centralised critic input assembly for MADDPG.

The critic for each agent sees a structured vector composed of:

    [shared_features, local_agent0, local_agent1, local_agent2,
     action_agent0, action_agent1, action_agent2]

SHARED features are the 4 floats every agent observes identically:
    - belt_speed (1)
    - branch congestion (3)
Extracting these from only one agent avoids feeding the same 4 numbers
into the critic three times.

LOCAL features per agent are everything else the agent sees:
    - gate_state, cooldown (2)
    - destination_mapping for THIS branch (3)
    - 5 package slots × 5 floats (25)
    - peer features (4, full-obs only)
    - messages (0, 2, or 6, comm variants only — partial-obs only)

Local size = obs_size - 4 regardless of mode (subtract belt_speed and
the 3 congestion floats).

CRITICAL — congestion indexing:
    In comm variants (partial-obs + messages), the observation grows
    AFTER congestion. The layout is:
        [..., congestion(3), message_floats(2 or 6)]
    Naive negative indexing obs[-3:] would grab message floats, NOT
    congestion, producing garbage critic input.
    This module uses FIXED indices from config.py for congestion, which
    is correct in all 4 modes: full(38), partial(34), partial+comm1(36),
    partial+comm3(40).

Supported observation sizes:
    38  -> full observability, no comm
    34  -> partial observability, no comm
    36  -> partial observability, 1-float comm channel
    40  -> partial observability, 3-float comm channel
"""

from typing import Sequence

import numpy as np

from training.maddpg.config import (
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
    CONGESTION_WIDTH,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
)


# Valid observation sizes. 38 is full-obs; 34/36/40 are partial-obs with
# 0/1/3 message floats respectively.
_VALID_OBS_SIZES = {38, 34, 36, 40}

# Size of the shared feature block (belt_speed + congestion).
_SHARED_SIZE = 1 + CONGESTION_WIDTH  # 4


def _is_full_obs(obs_size: int) -> bool:
    """Full-obs mode is exactly 38 floats; partial-obs variants are <38."""
    return obs_size == FULL_OBS_SIZE


def _extract_shared(obs: np.ndarray) -> np.ndarray:
    """
    Extract the 4-float shared feature block from a single agent's obs.
    Uses fixed congestion indices that differ between full and partial modes.
    """
    obs_size = obs.shape[0]
    belt_speed = obs[IDX_BELT_SPEED : IDX_BELT_SPEED + 1]

    if _is_full_obs(obs_size):
        congestion = obs[IDX_CONGESTION_START_FULL:IDX_CONGESTION_END_FULL]
    else:
        # Partial-obs: congestion is always at indices 31..34, regardless
        # of whether messages are appended after (comm variants).
        congestion = obs[IDX_CONGESTION_START_PARTIAL:IDX_CONGESTION_END_PARTIAL]

    return np.concatenate([belt_speed, congestion])


def _extract_local(obs: np.ndarray) -> np.ndarray:
    """
    Extract per-agent local features: everything EXCEPT belt_speed and
    congestion. In comm variants, message floats (living after congestion
    in the raw obs) are included here as local features.
    """
    obs_size = obs.shape[0]

    gate_cooldown = obs[IDX_GATE_STATE : IDX_COOLDOWN + 1]          # 2
    dest_mapping = obs[IDX_DEST_MAPPING_START:IDX_PACKAGE_SLOTS_START]  # 3
    packages = obs[IDX_PACKAGE_SLOTS_START:IDX_PACKAGE_SLOTS_END]   # 25

    parts = [gate_cooldown, dest_mapping, packages]

    if _is_full_obs(obs_size):
        # Add the 4-float peer feature block. No messages in full-obs mode.
        peers = obs[IDX_OTHER_AGENTS_START_FULL:IDX_OTHER_AGENTS_END_FULL]
        parts.append(peers)
    else:
        # Partial-obs: messages (if any) start right after congestion.
        # For obs_size=34 this slice is empty; for 36 it has 2 floats;
        # for 40 it has 6 floats.
        messages = obs[IDX_CONGESTION_END_PARTIAL:]
        if messages.size > 0:
            parts.append(messages)

    return np.concatenate(parts)


def build_critic_input(
    all_obs: Sequence[np.ndarray],
    all_actions: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Build the flat centralised critic input vector.

    Args:
        all_obs: list of 3 numpy arrays, one per agent. All must share the
            same obs_size. Supported sizes: 34, 36, 38, 40.
        all_actions: list of 3 numpy arrays, one per agent. All must share
            the same action_size.

    Returns:
        1D numpy array of length:
            shared(4) + local_per_agent * 3 + action_size * 3
        where local_per_agent = obs_size - 4.
    """
    if len(all_obs) != 3:
        raise ValueError(f"Expected 3 agent observations, got {len(all_obs)}")
    if len(all_actions) != 3:
        raise ValueError(f"Expected 3 agent actions, got {len(all_actions)}")

    obs_size = all_obs[0].shape[0]
    if obs_size not in _VALID_OBS_SIZES:
        raise ValueError(
            f"Unsupported obs_size {obs_size}. "
            f"Expected one of {sorted(_VALID_OBS_SIZES)}."
        )

    for i, obs in enumerate(all_obs):
        if obs.shape[0] != obs_size:
            raise ValueError(
                f"Agent {i} obs size {obs.shape[0]} != agent 0 obs size {obs_size}"
            )

    action_size = all_actions[0].shape[0]
    for i, act in enumerate(all_actions):
        if act.shape[0] != action_size:
            raise ValueError(
                f"Agent {i} action size {act.shape[0]} != agent 0 action size {action_size}"
            )

    # Shared block: extract from agent 0 (identical across all agents by design).
    shared = _extract_shared(all_obs[0])

    # Local blocks: one per agent.
    locals_list = [_extract_local(obs) for obs in all_obs]

    # Actions: already numpy arrays, just concatenate.
    actions_list = [np.asarray(a, dtype=np.float32) for a in all_actions]

    return np.concatenate([shared] + locals_list + actions_list).astype(np.float32)


def expected_critic_input_dim(obs_size: int, action_size: int) -> int:
    """
    Return the expected flat critic input dimension.
    Useful for sizing the Critic network's first linear layer.
    """
    if obs_size not in _VALID_OBS_SIZES:
        raise ValueError(f"Unsupported obs_size {obs_size}")
    local_per_agent = obs_size - _SHARED_SIZE
    return _SHARED_SIZE + local_per_agent * 3 + action_size * 3


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    # ---- Mode 1: Full-obs (38), action_size=1 → critic input = 109
    obs_full = [np.arange(38, dtype=np.float32) for _ in range(3)]
    act_full = [np.array([0.7], dtype=np.float32) for _ in range(3)]
    ci = build_critic_input(obs_full, act_full)
    exp = expected_critic_input_dim(38, 1)
    assert ci.shape == (109,), f"full: {ci.shape} != (109,)"
    assert exp == 109, f"full expected dim {exp} != 109"
    print(f"full-obs (38, act=1)         -> critic dim {ci.shape[0]} OK")

    # ---- Mode 2: Partial-obs no comm (34), action_size=1 → critic input = 97
    obs_p0 = [np.arange(34, dtype=np.float32) for _ in range(3)]
    act_p0 = [np.array([0.3], dtype=np.float32) for _ in range(3)]
    ci = build_critic_input(obs_p0, act_p0)
    assert ci.shape == (97,), f"partial no-comm: {ci.shape} != (97,)"
    assert expected_critic_input_dim(34, 1) == 97
    print(f"partial-obs no-comm (34,1)   -> critic dim {ci.shape[0]} OK")

    # ---- Mode 3: Partial-obs comm1 (36), action_size=2 → critic input = 106
    obs_p1 = [np.arange(36, dtype=np.float32) for _ in range(3)]
    act_p1 = [np.array([0.3, 0.9], dtype=np.float32) for _ in range(3)]
    ci = build_critic_input(obs_p1, act_p1)
    assert ci.shape == (106,), f"partial comm1: {ci.shape} != (106,)"
    assert expected_critic_input_dim(36, 2) == 106
    print(f"partial-obs comm1   (36,2)   -> critic dim {ci.shape[0]} OK")

    # ---- Mode 4: Partial-obs comm3 (40), action_size=4 → critic input = 124
    obs_p3 = [np.arange(40, dtype=np.float32) for _ in range(3)]
    act_p3 = [np.array([0.3, 0.1, 0.2, 0.4], dtype=np.float32) for _ in range(3)]
    ci = build_critic_input(obs_p3, act_p3)
    assert ci.shape == (124,), f"partial comm3: {ci.shape} != (124,)"
    assert expected_critic_input_dim(40, 4) == 124
    print(f"partial-obs comm3   (40,4)   -> critic dim {ci.shape[0]} OK")

    # ---- CRITICAL TEST: congestion is correctly extracted in comm variants.
    # Build a partial-obs comm3 (40 floats) observation where congestion
    # and message values are distinguishable, and verify the shared block
    # contains the congestion values, NOT the message values (which would
    # be the result of using obs[-3:] naively).
    obs_named = np.zeros(40, dtype=np.float32)
    obs_named[IDX_BELT_SPEED] = 0.50                                # belt_speed
    obs_named[IDX_CONGESTION_START_PARTIAL:IDX_CONGESTION_END_PARTIAL] = [0.11, 0.22, 0.33]
    # Messages at indices 34..40 — deliberately different values to detect wrong slice.
    obs_named[34:40] = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

    shared = _extract_shared(obs_named)
    assert shared.shape == (4,)
    assert abs(shared[0] - 0.50) < 1e-6, f"belt_speed wrong: {shared[0]}"
    assert abs(shared[1] - 0.11) < 1e-6, f"congestion[0] wrong: {shared[1]}"
    assert abs(shared[2] - 0.22) < 1e-6, f"congestion[1] wrong: {shared[2]}"
    assert abs(shared[3] - 0.33) < 1e-6, f"congestion[2] wrong: {shared[3]}"
    # Sanity: the wrong answer (from obs[-3:]) would be [0.94, 0.95, 0.96]
    assert shared[1] != 0.94, "BUG: grabbed message floats instead of congestion!"
    print(f"congestion extraction in comm3 variant: correct (not messages)")

    # ---- Error cases
    try:
        build_critic_input(obs_full[:2], act_full)  # 2 agents
        assert False, "should have raised"
    except ValueError:
        pass

    try:
        build_critic_input([np.zeros(34), np.zeros(36), np.zeros(34)], act_p0)  # mismatched
        assert False, "should have raised"
    except ValueError:
        pass

    try:
        build_critic_input([np.zeros(42)] * 3, act_full)  # unsupported size
        assert False, "should have raised"
    except ValueError:
        pass

    print(f"error handling: mismatched/invalid inputs correctly rejected")
    print("critic_input.py: all self-tests passed.")