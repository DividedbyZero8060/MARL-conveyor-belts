"""
MADDPG actor and critic networks.

Actor (per agent):
    obs_i (obs_size) -> FC(hidden) -> ReLU -> FC(hidden) -> ReLU
                    -> FC(action_size) -> Sigmoid -> action in [0, 1]

    Sigmoid output keeps actions in [0, 1] to match:
    - C# continuous activation threshold (0.5) on index 0
    - Python action masking (force index 0 to 0.0 when gate not retracted)
    - Message floats in comm variants (Step 17), interpreted in [0, 1]

Critic (per agent, centralised):
    [shared(4), local_a0, local_a1, local_a2, act_a0, act_a1, act_a2]
        -> FC(hidden) -> ReLU -> FC(hidden) -> ReLU -> FC(1) -> Q-value

    Input is pre-assembled by critic_input.build_critic_input. The critic
    has no sigmoid/tanh on the output — Q-values are unbounded reals.

Each agent gets its OWN actor and its OWN critic (no parameter sharing
at this layer — that deviates from canonical MADDPG but is easier to
reason about and matches what the workflow specifies).

num_hidden_layers controls the depth: with default 2, there are 2 hidden
ReLU blocks between input and output (matching the workflow spec).
"""

from typing import Optional

import torch
import torch.nn as nn

from training.maddpg.config import MaddpgConfig, DEFAULT_CONFIG


def _build_mlp_trunk(input_dim: int, hidden_units: int, num_hidden_layers: int) -> nn.Sequential:
    """
    Build a stack of [Linear -> ReLU] blocks.

    num_hidden_layers = 2 produces:
        Linear(input_dim, hidden) -> ReLU ->
        Linear(hidden, hidden)    -> ReLU

    The final output layer is added separately by the caller.
    """
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    layers = []
    in_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.ReLU())
        in_dim = hidden_units
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Deterministic actor that maps a single agent's observation to a
    continuous action vector in [0, 1].

    obs_size varies by mode (34/36/38/40).
    action_size varies by comm bandwidth (1/2/4).
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: Optional[MaddpgConfig] = None,
    ):
        super().__init__()
        cfg = config if config is not None else DEFAULT_CONFIG

        self._obs_size = obs_size
        self._action_size = action_size

        self._trunk = _build_mlp_trunk(
            input_dim=obs_size,
            hidden_units=cfg.hidden_units,
            num_hidden_layers=cfg.num_hidden_layers,
        )
        self._head = nn.Linear(cfg.hidden_units, action_size)
        # Sigmoid applied in forward() so target-network soft-update
        # behaviour stays identical for weight averaging.

    @property
    def obs_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: shape (batch, obs_size) or (obs_size,)
        Returns:
            action: shape (batch, action_size) or (action_size,),
                    values in [0, 1] via sigmoid.
        """
        squeeze_batch = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_batch = True

        h = self._trunk(obs)
        logits = self._head(h)
        action = torch.sigmoid(logits)

        if squeeze_batch:
            action = action.squeeze(0)
        return action


class Critic(nn.Module):
    """
    Centralised critic producing a scalar Q-value from the flat critic
    input vector assembled by critic_input.build_critic_input.

    critic_input_dim is computed externally via
    critic_input.expected_critic_input_dim(obs_size, action_size) and
    passed in at construction time.
    """

    def __init__(
        self,
        critic_input_dim: int,
        config: Optional[MaddpgConfig] = None,
    ):
        super().__init__()
        cfg = config if config is not None else DEFAULT_CONFIG

        self._input_dim = critic_input_dim

        self._trunk = _build_mlp_trunk(
            input_dim=critic_input_dim,
            hidden_units=cfg.hidden_units,
            num_hidden_layers=cfg.num_hidden_layers,
        )
        self._head = nn.Linear(cfg.hidden_units, 1)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            critic_input: shape (batch, critic_input_dim) or (critic_input_dim,)
        Returns:
            Q-value: shape (batch, 1) or (1,). Unbounded real.
        """
        squeeze_batch = False
        if critic_input.dim() == 1:
            critic_input = critic_input.unsqueeze(0)
            squeeze_batch = True

        h = self._trunk(critic_input)
        q = self._head(h)

        if squeeze_batch:
            q = q.squeeze(0)
        return q


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Polyak averaging: target = tau * source + (1 - tau) * target.
    Applied to target actor and target critic after every training update.
    """
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.mul_(1.0 - tau)
            t_param.data.add_(tau * s_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """
    Copy source weights directly into target. Used once at initialisation
    to make target networks start identical to their online counterparts.
    """
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    import numpy as np
    from training.maddpg.critic_input import (
        build_critic_input,
        expected_critic_input_dim,
    )

    torch.manual_seed(0)

    # ---- Actor forward pass: full-obs, action_size=1
    actor = Actor(obs_size=38, action_size=1)
    obs_batch = torch.randn(4, 38)
    action = actor(obs_batch)
    assert action.shape == (4, 1), f"actor batch output shape {action.shape}"
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0), \
        f"actor output outside [0,1]: min={action.min()}, max={action.max()}"
    print(f"Actor (38->1) batch forward: shape {tuple(action.shape)}, range [{action.min():.3f}, {action.max():.3f}] OK")

    # Unbatched forward
    obs_single = torch.randn(38)
    action_single = actor(obs_single)
    assert action_single.shape == (1,), f"actor unbatched {action_single.shape}"
    print(f"Actor (38->1) single forward: shape {tuple(action_single.shape)} OK")

    # ---- Actor forward pass: partial-obs comm3, action_size=4
    actor_c3 = Actor(obs_size=40, action_size=4)
    obs_c3 = torch.randn(8, 40)
    act_c3 = actor_c3(obs_c3)
    assert act_c3.shape == (8, 4), f"actor comm3 shape {act_c3.shape}"
    assert torch.all(act_c3 >= 0.0) and torch.all(act_c3 <= 1.0)
    print(f"Actor (40->4) batch forward: shape {tuple(act_c3.shape)} OK")

    # ---- Critic forward pass: full-obs mode, dim 109
    critic_dim_full = expected_critic_input_dim(38, 1)
    assert critic_dim_full == 109
    critic_full = Critic(critic_input_dim=critic_dim_full)

    # Assemble a realistic critic input via build_critic_input, then batch it
    all_obs = [np.random.randn(38).astype(np.float32) for _ in range(3)]
    all_acts = [np.random.rand(1).astype(np.float32) for _ in range(3)]
    ci_np = build_critic_input(all_obs, all_acts)
    assert ci_np.shape == (109,)

    ci_batch = torch.from_numpy(ci_np).unsqueeze(0).repeat(16, 1)  # batch of 16
    q = critic_full(ci_batch)
    assert q.shape == (16, 1), f"critic batch q shape {q.shape}"
    print(f"Critic (dim 109) batch forward: q shape {tuple(q.shape)} OK")

    # Single (unbatched) critic forward
    q_single = critic_full(torch.from_numpy(ci_np))
    assert q_single.shape == (1,), f"critic single q shape {q_single.shape}"
    print(f"Critic (dim 109) single forward: q shape {tuple(q_single.shape)} OK")

    # ---- Critic for comm3 mode: dim 124
    critic_dim_c3 = expected_critic_input_dim(40, 4)
    assert critic_dim_c3 == 124
    critic_c3 = Critic(critic_input_dim=critic_dim_c3)
    ci_c3 = torch.randn(4, 124)
    q_c3 = critic_c3(ci_c3)
    assert q_c3.shape == (4, 1)
    print(f"Critic (dim 124) batch forward: q shape {tuple(q_c3.shape)} OK")

    # ---- Soft update sanity: target drifts toward source by factor tau
    source = Actor(obs_size=38, action_size=1)
    target = Actor(obs_size=38, action_size=1)

    # Capture target's initial first-layer weight
    target_w_before = target._trunk[0].weight.data.clone()
    source_w = source._trunk[0].weight.data.clone()

    tau = 0.1
    soft_update(target, source, tau)
    target_w_after = target._trunk[0].weight.data

    expected = tau * source_w + (1 - tau) * target_w_before
    assert torch.allclose(target_w_after, expected, atol=1e-6), \
        "soft_update did not match Polyak formula"
    print(f"soft_update (tau={tau}) matches Polyak formula OK")

    # ---- Hard update: target becomes identical to source
    hard_update(target, source)
    assert torch.allclose(target._trunk[0].weight.data, source._trunk[0].weight.data)
    print(f"hard_update copies source weights exactly OK")

    # ---- Backward pass smoke test: loss computes gradient without NaN
    actor_train = Actor(obs_size=34, action_size=1)
    critic_train = Critic(critic_input_dim=expected_critic_input_dim(34, 1))  # 97

    obs = torch.randn(8, 34, requires_grad=False)
    a = actor_train(obs)

    # Fake critic input: pretend 3 identical copies, sanity only
    fake_ci = torch.randn(8, 97, requires_grad=True)
    q = critic_train(fake_ci)
    loss = -q.mean()  # actor-loss-shaped
    loss.backward()

    for p in critic_train.parameters():
        assert p.grad is not None and not torch.any(torch.isnan(p.grad))
    print(f"backward pass: no NaN gradients OK")

    print("networks.py: all self-tests passed.")