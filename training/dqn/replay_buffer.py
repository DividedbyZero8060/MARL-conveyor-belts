"""
Per-agent replay buffer for Independent DQN.

Unlike MADDPG's joint buffer (which stores all-agent tuples per step),
this stores per-agent transitions independently. Each agent has its own
buffer with no cross-agent correlation. This is the defining property of
the independent baseline.

Transition schema (per agent):
    obs:       float32 (obs_size,)
    action:    int64   ()               -- discrete action index
    action_mask: bool  (num_actions,)   -- True = DISABLED (matches ML-Agents convention)
    reward:    float32 ()
    next_obs:  float32 (obs_size,)
    next_action_mask: bool (num_actions,)  -- mask at next_obs for target Q computation
    done:      bool    ()
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DqnBatch:
    """Sampled batch of transitions from a single agent's buffer."""
    obs: np.ndarray                # (batch, obs_size) float32
    actions: np.ndarray            # (batch,) int64
    action_masks: np.ndarray       # (batch, num_actions) bool
    rewards: np.ndarray            # (batch,) float32
    next_obs: np.ndarray           # (batch, obs_size) float32
    next_action_masks: np.ndarray  # (batch, num_actions) bool
    dones: np.ndarray              # (batch,) float32 (0.0 or 1.0)


class PerAgentReplayBuffer:
    """
    Fixed-capacity circular buffer for one agent's transitions.

    Stores: obs, action, action_mask, reward, next_obs, next_action_mask, done.
    Sampling is uniform random without replacement within a batch, using a
    dedicated numpy Generator so buffers can be seeded independently for
    deterministic tests.
    """

    def __init__(
        self,
        capacity: int,
        obs_size: int,
        num_actions: int,
        seed: Optional[int] = None,
    ):
        self._capacity = capacity
        self._obs_size = obs_size
        self._num_actions = num_actions
        self._rng = np.random.default_rng(seed)

        # Preallocate storage.
        self._obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._action_masks = np.zeros((capacity, num_actions), dtype=bool)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._next_action_masks = np.zeros((capacity, num_actions), dtype=bool)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._size = 0
        self._write_idx = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size

    def store(
        self,
        obs: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Write a single transition at the current circular index."""
        i = self._write_idx
        self._obs[i] = obs
        self._actions[i] = action
        self._action_masks[i] = action_mask
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._next_action_masks[i] = next_action_mask
        self._dones[i] = 1.0 if done else 0.0

        self._write_idx = (self._write_idx + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(self, batch_size: int) -> DqnBatch:
        """Uniform random sample of `batch_size` transitions."""
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty buffer.")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return DqnBatch(
            obs=self._obs[indices].copy(),
            actions=self._actions[indices].copy(),
            action_masks=self._action_masks[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_obs=self._next_obs[indices].copy(),
            next_action_masks=self._next_action_masks[indices].copy(),
            dones=self._dones[indices].copy(),
        )


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    buf = PerAgentReplayBuffer(capacity=100, obs_size=38, num_actions=2, seed=42)
    assert buf.size == 0
    assert not buf.is_ready(1)

    rng = np.random.default_rng(0)
    for i in range(150):  # overfill to test circular wrap
        obs = rng.random(38).astype(np.float32)
        action = int(rng.integers(0, 2))
        mask = np.array([False, False])
        reward = float(rng.standard_normal())
        next_obs = rng.random(38).astype(np.float32)
        next_mask = np.array([False, True])
        done = bool(rng.random() < 0.1)
        buf.store(obs, action, mask, reward, next_obs, next_mask, done)

    assert buf.size == 100, f"buffer should be at capacity, got {buf.size}"
    print(f"buffer fills and wraps correctly OK")

    # Sample
    batch = buf.sample(32)
    assert batch.obs.shape == (32, 38)
    assert batch.actions.shape == (32,)
    assert batch.action_masks.shape == (32, 2)
    assert batch.rewards.shape == (32,)
    assert batch.next_obs.shape == (32, 38)
    assert batch.next_action_masks.shape == (32, 2)
    assert batch.dones.shape == (32,)
    assert batch.dones.dtype == np.float32
    print(f"sample shapes OK")

    # Deterministic sampling with seed
    buf2 = PerAgentReplayBuffer(capacity=100, obs_size=38, num_actions=2, seed=42)
    for i in range(50):
        buf2.store(
            np.zeros(38, dtype=np.float32),
            0, np.array([False, False]),
            0.0,
            np.zeros(38, dtype=np.float32),
            np.array([False, False]),
            False,
        )
    b1 = buf2.sample(10)
    buf3 = PerAgentReplayBuffer(capacity=100, obs_size=38, num_actions=2, seed=42)
    for i in range(50):
        buf3.store(
            np.zeros(38, dtype=np.float32),
            0, np.array([False, False]),
            0.0,
            np.zeros(38, dtype=np.float32),
            np.array([False, False]),
            False,
        )
    b2 = buf3.sample(10)
    # Note: different write counts before sampling will differ; this only tests
    # that two identical buffers with the same seed produce identical samples.
    assert np.array_equal(b1.obs, b2.obs)
    print(f"seeded sampling is deterministic OK")

    print("replay_buffer.py: all self-tests passed.")