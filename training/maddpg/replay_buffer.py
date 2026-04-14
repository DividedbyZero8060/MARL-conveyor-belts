"""
MADDPG replay buffer.

Fixed-size circular buffer storing transitions as numpy arrays. Each
transition holds the full multi-agent state:

    obs             : (num_agents, obs_size)       — obs at decision time
    actor_outputs   : (num_agents, action_size)    — RAW actor(obs) + noise
    masked_actions  : (num_agents, action_size)    — what the env actually saw
    rewards         : (num_agents,)                — per-agent reward this step
    next_obs        : (num_agents, obs_size)       — obs after env step
    dones           : (num_agents,)                — True on terminal_steps

Why both actor_outputs AND masked_actions?
    The critic trains on masked_actions (the actions the environment
    received, i.e. after forcing action[0]=0.0 when the gate was not
    retracted). The raw actor_outputs are logged as an illegal_action_rate
    metric — percentage of steps where actor wanted to activate an
    illegal gate. If that rate stays >50% after 500k steps, the actor
    hasn't learned the gate-state constraint and something is wrong.

Circular overwrite:
    Once the buffer reaches capacity, new transitions overwrite the
    oldest entries at the same position. _size tops out at capacity;
    _cursor wraps around.

Warmup:
    Trainer checks buffer.size >= warmup_transitions before sampling.
    Until then, no gradient updates happen — just exploration.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TransitionBatch:
    """A mini-batch of transitions sampled from the replay buffer."""
    obs: np.ndarray            # (batch, num_agents, obs_size)
    actor_outputs: np.ndarray  # (batch, num_agents, action_size)
    masked_actions: np.ndarray # (batch, num_agents, action_size)
    rewards: np.ndarray        # (batch, num_agents)
    next_obs: np.ndarray       # (batch, num_agents, obs_size)
    dones: np.ndarray          # (batch, num_agents), bool or float


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Pre-allocates all numpy arrays at construction time — zero allocation
    during training.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_size: int,
        action_size: int,
        seed: Optional[int] = None,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if num_agents <= 0:
            raise ValueError(f"num_agents must be > 0, got {num_agents}")
        if obs_size <= 0:
            raise ValueError(f"obs_size must be > 0, got {obs_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be > 0, got {action_size}")

        self._capacity = capacity
        self._num_agents = num_agents
        self._obs_size = obs_size
        self._action_size = action_size

        # Pre-allocated storage. float32 for obs/actions/rewards keeps
        # memory tractable: 500k × 3 × 38 × 4 bytes ≈ 228 MB for obs alone.
        self._obs = np.zeros((capacity, num_agents, obs_size), dtype=np.float32)
        self._actor_outputs = np.zeros((capacity, num_agents, action_size), dtype=np.float32)
        self._masked_actions = np.zeros((capacity, num_agents, action_size), dtype=np.float32)
        self._rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self._next_obs = np.zeros((capacity, num_agents, obs_size), dtype=np.float32)
        self._dones = np.zeros((capacity, num_agents), dtype=np.float32)

        self._cursor = 0   # Next write index
        self._size = 0     # Current number of valid transitions

        # Dedicated RNG so buffer sampling is deterministic under a fixed seed,
        # independent of any global numpy RNG state.
        self._rng = np.random.default_rng(seed)

    # ----- Properties -----

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        """Current number of transitions stored (caps at capacity)."""
        return self._size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def obs_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def is_ready(self, warmup: int) -> bool:
        """True when the buffer has enough transitions to start training."""
        return self._size >= warmup

    # ----- Store -----

    def store(
        self,
        obs: np.ndarray,
        actor_outputs: np.ndarray,
        masked_actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Store a single transition. All arrays must have the shapes:

            obs, next_obs     : (num_agents, obs_size)
            actor_outputs     : (num_agents, action_size)
            masked_actions    : (num_agents, action_size)
            rewards           : (num_agents,)
            dones             : (num_agents,)

        Arrays are COPIED into the internal storage — the caller is free
        to reuse/mutate their inputs after the call returns.
        """
        self._validate_shape(obs, (self._num_agents, self._obs_size), "obs")
        self._validate_shape(actor_outputs, (self._num_agents, self._action_size), "actor_outputs")
        self._validate_shape(masked_actions, (self._num_agents, self._action_size), "masked_actions")
        self._validate_shape(rewards, (self._num_agents,), "rewards")
        self._validate_shape(next_obs, (self._num_agents, self._obs_size), "next_obs")
        self._validate_shape(dones, (self._num_agents,), "dones")

        i = self._cursor
        self._obs[i] = obs
        self._actor_outputs[i] = actor_outputs
        self._masked_actions[i] = masked_actions
        self._rewards[i] = rewards
        self._next_obs[i] = next_obs
        self._dones[i] = dones.astype(np.float32)

        self._cursor = (self._cursor + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def _validate_shape(self, arr: np.ndarray, expected: tuple, name: str) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be numpy ndarray, got {type(arr).__name__}")
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape} != expected {expected}")

    # ----- Sample -----

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions uniformly at random (with replacement).

        Returns a TransitionBatch with leading dim = batch_size.
        Raises RuntimeError if the buffer has fewer than batch_size stored.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self._size < batch_size:
            raise RuntimeError(
                f"Cannot sample batch of {batch_size}; buffer only has {self._size} transitions."
            )

        indices = self._rng.integers(low=0, high=self._size, size=batch_size)

        return TransitionBatch(
            obs=self._obs[indices],
            actor_outputs=self._actor_outputs[indices],
            masked_actions=self._masked_actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            dones=self._dones[indices],
        )


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    NUM_AGENTS = 3
    OBS_SIZE = 38
    ACTION_SIZE = 1
    CAPACITY = 100

    buf = ReplayBuffer(
        capacity=CAPACITY,
        num_agents=NUM_AGENTS,
        obs_size=OBS_SIZE,
        action_size=ACTION_SIZE,
        seed=42,
    )

    # ---- Empty buffer properties
    assert buf.size == 0
    assert not buf.is_ready(warmup=1)
    assert buf.capacity == CAPACITY
    print(f"empty buffer: size=0, not ready OK")

    # ---- Store 50 transitions
    for step in range(50):
        obs = np.full((NUM_AGENTS, OBS_SIZE), float(step), dtype=np.float32)
        ao = np.full((NUM_AGENTS, ACTION_SIZE), float(step) * 0.01, dtype=np.float32)
        ma = np.zeros((NUM_AGENTS, ACTION_SIZE), dtype=np.float32)
        r = np.full((NUM_AGENTS,), float(step) * 0.1, dtype=np.float32)
        nobs = np.full((NUM_AGENTS, OBS_SIZE), float(step) + 0.5, dtype=np.float32)
        d = np.array([step % 10 == 9] * NUM_AGENTS, dtype=np.bool_)
        buf.store(obs, ao, ma, r, nobs, d)

    assert buf.size == 50
    assert buf.is_ready(warmup=10)
    assert not buf.is_ready(warmup=51)
    print(f"after 50 stores: size=50, ready(10)=True, ready(51)=False OK")

    # ---- Sample a batch
    batch = buf.sample(batch_size=8)
    assert batch.obs.shape == (8, NUM_AGENTS, OBS_SIZE)
    assert batch.actor_outputs.shape == (8, NUM_AGENTS, ACTION_SIZE)
    assert batch.masked_actions.shape == (8, NUM_AGENTS, ACTION_SIZE)
    assert batch.rewards.shape == (8, NUM_AGENTS)
    assert batch.next_obs.shape == (8, NUM_AGENTS, OBS_SIZE)
    assert batch.dones.shape == (8, NUM_AGENTS)
    print(f"sample batch of 8: all shapes OK")

    # ---- Sample gives sane values
    # obs at step k was set to k (float). rewards to k*0.1.
    # Sanity check a single entry: for every sampled transition, all values
    # within an entry should be consistent (obs=k, reward=k*0.1).
    for b in range(8):
        step_val = batch.obs[b, 0, 0]  # first agent, first obs index
        expected_reward = step_val * 0.1
        assert abs(batch.rewards[b, 0] - expected_reward) < 1e-4, \
            f"step {step_val} reward {batch.rewards[b, 0]} != expected {expected_reward}"
    print(f"sampled transitions are internally consistent OK")

    # ---- Circular overwrite
    # Fill past capacity. Size should cap at CAPACITY, and the oldest
    # transitions should be gone, overwritten by the newest.
    for step in range(50, 200):  # 150 more stores → total 200, capacity 100
        obs = np.full((NUM_AGENTS, OBS_SIZE), float(step), dtype=np.float32)
        ao = np.zeros((NUM_AGENTS, ACTION_SIZE), dtype=np.float32)
        ma = np.zeros((NUM_AGENTS, ACTION_SIZE), dtype=np.float32)
        r = np.full((NUM_AGENTS,), float(step) * 0.1, dtype=np.float32)
        nobs = np.full((NUM_AGENTS, OBS_SIZE), float(step) + 0.5, dtype=np.float32)
        d = np.zeros((NUM_AGENTS,), dtype=np.bool_)
        buf.store(obs, ao, ma, r, nobs, d)

    assert buf.size == CAPACITY, f"size should cap at {CAPACITY}, got {buf.size}"
    print(f"after 200 stores: size capped at {buf.size} OK")

    # The oldest transition that survived should be step 100 (we stored
    # 0..199, and the buffer holds the last 100).
    # Check that no stored obs has step < 100.
    all_steps = buf._obs[: buf.size, 0, 0]
    assert all_steps.min() >= 100.0, f"oldest step in buffer is {all_steps.min()}, expected >= 100"
    assert all_steps.max() == 199.0, f"newest step should be 199, got {all_steps.max()}"
    print(f"circular overwrite: oldest={all_steps.min()}, newest={all_steps.max()} OK")

    # ---- Shape validation on store
    buf2 = ReplayBuffer(capacity=10, num_agents=3, obs_size=38, action_size=1)
    try:
        buf2.store(
            obs=np.zeros((3, 34), dtype=np.float32),  # wrong obs_size
            actor_outputs=np.zeros((3, 1), dtype=np.float32),
            masked_actions=np.zeros((3, 1), dtype=np.float32),
            rewards=np.zeros((3,), dtype=np.float32),
            next_obs=np.zeros((3, 38), dtype=np.float32),
            dones=np.zeros((3,), dtype=np.bool_),
        )
        assert False, "should have raised on wrong obs shape"
    except ValueError:
        pass
    print(f"shape validation: wrong obs shape correctly rejected OK")

    # ---- Sampling an empty buffer
    buf3 = ReplayBuffer(capacity=10, num_agents=3, obs_size=38, action_size=1)
    try:
        buf3.sample(batch_size=1)
        assert False, "should have raised on empty buffer"
    except RuntimeError:
        pass
    print(f"sampling empty buffer: correctly raises RuntimeError OK")

    # ---- Deterministic sampling under fixed seed
    bufA = ReplayBuffer(capacity=50, num_agents=3, obs_size=38, action_size=1, seed=123)
    bufB = ReplayBuffer(capacity=50, num_agents=3, obs_size=38, action_size=1, seed=123)
    for step in range(30):
        obs = np.full((3, 38), float(step), dtype=np.float32)
        zero_a = np.zeros((3, 1), dtype=np.float32)
        r = np.full((3,), float(step), dtype=np.float32)
        d = np.zeros((3,), dtype=np.bool_)
        bufA.store(obs, zero_a, zero_a, r, obs, d)
        bufB.store(obs, zero_a, zero_a, r, obs, d)

    bA = bufA.sample(batch_size=16)
    bB = bufB.sample(batch_size=16)
    assert np.array_equal(bA.obs, bB.obs), "same seed should give same sample"
    print(f"deterministic seeding: two buffers with same seed sample identically OK")

    print("replay_buffer.py: all self-tests passed.")