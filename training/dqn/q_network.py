"""
DQN Q-network. Single-agent architecture; one instance per agent.

obs (obs_size) -> FC(hidden) -> ReLU -> FC(hidden) -> ReLU -> FC(num_actions)

Output is unbounded Q-values (one per discrete action). Action selection
(argmax, epsilon-greedy, masking) happens externally in the trainer.

No sigmoid/tanh on the output. No parameter sharing between agents.
"""

from typing import Optional

import torch
import torch.nn as nn

from training.dqn.config import DqnConfig, DEFAULT_CONFIG


def _build_mlp(input_dim: int, hidden_units: int, num_hidden_layers: int) -> nn.Sequential:
    """Linear -> ReLU stack. Same pattern as training.maddpg.networks."""
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    layers = []
    in_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.ReLU())
        in_dim = hidden_units
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """
    Q-value network for one agent. Maps its local observation to a vector
    of Q-values, one per discrete action.
    """

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        config: Optional[DqnConfig] = None,
    ):
        super().__init__()
        cfg = config if config is not None else DEFAULT_CONFIG

        self._obs_size = obs_size
        self._num_actions = num_actions

        self._trunk = _build_mlp(
            input_dim=obs_size,
            hidden_units=cfg.hidden_units,
            num_hidden_layers=cfg.num_hidden_layers,
        )
        self._head = nn.Linear(cfg.hidden_units, num_actions)

    @property
    def obs_size(self) -> int:
        return self._obs_size

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: shape (batch, obs_size) or (obs_size,)
        Returns:
            q_values: shape (batch, num_actions) or (num_actions,). Unbounded reals.
        """
        squeeze_batch = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_batch = True

        h = self._trunk(obs)
        q = self._head(h)

        if squeeze_batch:
            q = q.squeeze(0)
        return q


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy source weights into target. Used every target_update_freq env steps."""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # Basic forward pass
    net = QNetwork(obs_size=38, num_actions=2)
    obs = torch.randn(8, 38)
    q = net(obs)
    assert q.shape == (8, 2), f"q shape {q.shape}"
    print(f"QNetwork batch forward: shape {tuple(q.shape)} OK")

    # Unbatched
    q_single = net(torch.randn(38))
    assert q_single.shape == (2,)
    print(f"QNetwork unbatched forward: shape {tuple(q_single.shape)} OK")

    # Q-values can be any real (no clamping)
    big_obs = torch.randn(4, 38) * 1000
    q_big = net(big_obs)
    assert torch.all(torch.isfinite(q_big))
    print(f"QNetwork handles large input without NaN OK")

    # Partial obs
    net_p = QNetwork(obs_size=34, num_actions=2)
    q_p = net_p(torch.randn(4, 34))
    assert q_p.shape == (4, 2)
    print(f"QNetwork (partial obs 34) OK")

    # Hard update
    source = QNetwork(obs_size=38, num_actions=2)
    target = QNetwork(obs_size=38, num_actions=2)
    before = target._trunk[0].weight.data.clone()
    hard_update(target, source)
    after = target._trunk[0].weight.data
    assert torch.allclose(after, source._trunk[0].weight.data)
    assert not torch.allclose(after, before)
    print(f"hard_update copies source weights exactly OK")

    # Backward pass
    net.zero_grad()
    q_train = net(torch.randn(16, 38, requires_grad=False))
    loss = q_train.mean()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None and not torch.any(torch.isnan(p.grad))
    print(f"backward pass: no NaN gradients OK")

    print("q_network.py: all self-tests passed.")