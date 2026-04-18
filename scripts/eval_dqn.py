"""
DQN evaluation CLI.

Runs trained DQN Q-networks against Unity for N episodes with greedy
action selection (epsilon=0, no exploration). The C# EvaluationRunner
handles episode counting and CSV output; this script drives the policy.

Usage:
    python scripts/eval_dqn.py \
        --checkpoint=results/checkpoints/single_area_dqn_s42/checkpoint_final.pt \
        --seed=42

Scene setup (Unity side) before launch:
    - SortingAgent Behavior Type = Default (Python controls)
    - Discrete Branches = [2], Continuous Actions = 0
    - SortingAgentGroup enabled
    - IndependentRewardDistributor enabled (or RewardDistributor — doesn't
      matter for eval since EvaluationRunner reads EnvironmentManager counters)
    - EvaluationRunner GameObject: enabled
    - EvaluationRunner._useHeuristicAutomatic = false
    - EvaluationRunner._targetEpisodes = 100
    - EvaluationRunner._outputPath = results/eval_dqn_sXX.csv

The EvaluationRunner stops Play after _targetEpisodes episodes; this
script detects the disconnect and exits cleanly.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.getLogger("mlagents_envs.side_channel.side_channel_manager").setLevel(logging.ERROR)

from training.dqn.config import DqnConfig, DEFAULT_CONFIG, IDX_GATE_STATE
from training.dqn.q_network import QNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of trained DQN Q-networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint to evaluate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for Unity's RNG.")
    parser.add_argument("--max-env-steps", type=int, default=500_000,
                        help="Hard cap on env steps; evaluation normally stops earlier "
                             "when EvaluationRunner stops Play.")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def load_q_nets(
    checkpoint_path: str,
    device: torch.device,
) -> List[QNetwork]:
    """Load Q-networks from a DQN trainer checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    saved = state["config"]

    obs_size = saved["obs_size"]
    num_actions = saved["num_actions"]
    num_agents = saved["num_agents"]

    cfg = DEFAULT_CONFIG
    nets = []
    for i in range(num_agents):
        net = QNetwork(obs_size, num_actions, cfg).to(device)
        net.load_state_dict(state["q_nets"][i])
        net.eval()
        nets.append(net)

    print(f"[eval_dqn] loaded {num_agents} Q-networks from {checkpoint_path} "
          f"(env_step={state['env_step']}, obs_size={obs_size}, "
          f"num_actions={num_actions})")
    return nets


def main() -> None:
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.exception import UnityCommunicatorStoppedException

    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    behavior_name_prefix = DEFAULT_CONFIG.behavior_name
    num_agents = DEFAULT_CONFIG.num_agents

    print("=" * 60)
    print("DQN deterministic evaluation")
    print("=" * 60)
    print(f"checkpoint:      {args.checkpoint}")
    print(f"seed:            {args.seed}")
    print(f"device:          {device}")
    print("=" * 60)

    q_nets = load_q_nets(args.checkpoint, device)

    print(f"[eval_dqn] connecting to Unity (behavior prefix='{behavior_name_prefix}')...")
    env = UnityEnvironment(seed=args.seed)

    try:
        env.reset()
        available = list(env.behavior_specs.keys())
        behavior_name = None
        if behavior_name_prefix in available:
            behavior_name = behavior_name_prefix
        else:
            prefix = behavior_name_prefix + "?"
            matches = [n for n in available if n.startswith(prefix)]
            if len(matches) == 1:
                behavior_name = matches[0]
        if behavior_name is None:
            raise RuntimeError(
                f"Behavior '{behavior_name_prefix}' not found. Available: {available}"
            )

        spec = env.behavior_specs[behavior_name]
        print(f"[eval_dqn] resolved behavior: '{behavior_name}'")
        print(f"[eval_dqn] obs_shape={spec.observation_specs[0].shape}, "
              f"action_spec={spec.action_spec}")
        print(f"[eval_dqn] running evaluation — EvaluationRunner will stop Play "
              f"when target episodes reached")

        env_step = 0
        start = time.time()

        while env_step < args.max_env_steps:
            decision_steps, _terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                agent_ids = list(decision_steps.agent_id)
                n_present = min(len(agent_ids), num_agents)

                # Read observations and action masks
                current_obs = np.zeros(
                    (num_agents, q_nets[0].obs_size), dtype=np.float32
                )
                current_masks = np.zeros(
                    (num_agents, q_nets[0].num_actions), dtype=bool
                )
                for slot_idx in range(n_present):
                    agent_id = agent_ids[slot_idx]
                    current_obs[slot_idx] = decision_steps[agent_id].obs[0]
                    masks = decision_steps[agent_id].action_mask
                    if masks is not None and len(masks) > 0:
                        current_masks[slot_idx] = masks[0].astype(bool)

                # Greedy action selection: argmax of Q-values with masking
                actions = np.zeros((num_agents, 1), dtype=np.int32)
                with torch.no_grad():
                    for i in range(n_present):
                        obs_t = torch.from_numpy(current_obs[i]).to(device)
                        q = q_nets[i](obs_t).cpu().numpy()  # (num_actions,)
                        # Mask illegal actions with -1e9 (NOT 0.0)
                        q_masked = q.copy()
                        q_masked[current_masks[i]] = -1e9
                        actions[i, 0] = int(np.argmax(q_masked))

                env.set_actions(behavior_name, ActionTuple(discrete=actions))
                env_step += 1

                if env_step % 5000 == 0:
                    elapsed = time.time() - start
                    print(f"[eval_dqn] env_step={env_step} elapsed={elapsed:.1f}s")

            env.step()

        print(f"[eval_dqn] max_env_steps reached ({args.max_env_steps}) "
              f"without Play stopping")
        print(f"[eval_dqn] check the CSV output from EvaluationRunner anyway")

    except UnityCommunicatorStoppedException:
        print(f"[eval_dqn] Unity Play stopped — EvaluationRunner likely finished. "
              f"Check the CSV output configured on EvaluationRunner.")
    except KeyboardInterrupt:
        print(f"[eval_dqn] interrupted at env_step={env_step}")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[eval_dqn] error closing env: {e}")


if __name__ == "__main__":
    main()