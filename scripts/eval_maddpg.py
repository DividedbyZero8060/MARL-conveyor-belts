"""
MADDPG evaluation CLI.

Runs a trained MADDPG/MATD3 policy against Unity for N episodes with
deterministic action selection (no Gaussian exploration noise). The C#
EvaluationRunner handles episode counting and CSV output; this script
drives the policy.

Usage:
    python scripts/eval_maddpg.py --checkpoint=results/checkpoints/single_area_matd3_s42/checkpoint_final.pt --seed=42

Scene setup (Unity side) before launch:
    - SortingAgent Behavior Type = Default (NOT Inference Only — Python controls)
    - Continuous Actions = action size for the trained policy's comm bandwidth
    - EvaluationRunner GameObject: enabled
    - EvaluationRunner._useHeuristicAutomatic = false
    - EvaluationRunner._targetEpisodes = 100
    - EvaluationRunner._outputPath = results/eval_matd3_sXX.csv
    - SortingAgentGroup enabled, RewardDistributor enabled

The EvaluationRunner stops Play after _targetEpisodes episodes; this
script detects the disconnect and exits cleanly.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.getLogger("mlagents_envs.side_channel.side_channel_manager").setLevel(logging.ERROR)

from training.maddpg.config import (
    MaddpgConfig,
    DEFAULT_CONFIG,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
    IDX_GATE_STATE,
    action_size_for_bandwidth,
)
from training.maddpg.networks import Actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of a trained MADDPG policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint to evaluate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for Unity's RNG. Evaluation action selection is deterministic.")
    parser.add_argument("--partial-obs", action="store_true",
                        help="Partial observability. Must match training.")
    parser.add_argument("--comm-bandwidth", type=int, default=0, choices=[0, 1, 3],
                        help="Communication bandwidth. Must match training.")
    parser.add_argument("--max-env-steps", type=int, default=500_000,
                        help="Hard cap on env steps; evaluation normally stops earlier when "
                             "EvaluationRunner stops Play.")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def determine_obs_size(partial_obs: bool, comm_bandwidth: int) -> int:
    if not partial_obs:
        if comm_bandwidth != 0:
            raise ValueError("Communication requires --partial-obs.")
        return FULL_OBS_SIZE
    return {0: PARTIAL_OBS_SIZE, 1: PARTIAL_OBS_SIZE + 2, 3: PARTIAL_OBS_SIZE + 6}[comm_bandwidth]


def load_actors(checkpoint_path: str, obs_size: int, action_size: int,
                num_agents: int, device: torch.device) -> list:
    """Load just the actor networks from a trainer checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)

    saved = state["config"]
    if saved["obs_size"] != obs_size:
        raise ValueError(f"Checkpoint obs_size {saved['obs_size']} != expected {obs_size}")
    if saved["action_size"] != action_size:
        raise ValueError(f"Checkpoint action_size {saved['action_size']} != expected {action_size}")
    if saved["num_agents"] != num_agents:
        raise ValueError(f"Checkpoint num_agents {saved['num_agents']} != expected {num_agents}")

    cfg = DEFAULT_CONFIG
    actors = []
    for i in range(num_agents):
        a = Actor(obs_size, action_size, cfg).to(device)
        a.load_state_dict(state["actors"][i])
        a.eval()
        actors.append(a)
    print(f"[eval] loaded {num_agents} actors from {checkpoint_path} (env_step={state['env_step']})")
    return actors


def main() -> None:
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.exception import UnityCommunicatorStoppedException

    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    obs_size = determine_obs_size(args.partial_obs, args.comm_bandwidth)
    action_size = action_size_for_bandwidth(args.comm_bandwidth)
    num_agents = DEFAULT_CONFIG.num_agents
    behavior_name_prefix = DEFAULT_CONFIG.behavior_name

    print("=" * 60)
    print(f"MADDPG deterministic evaluation")
    print("=" * 60)
    print(f"checkpoint:      {args.checkpoint}")
    print(f"seed:            {args.seed}")
    print(f"obs_size:        {obs_size}")
    print(f"action_size:     {action_size}")
    print(f"comm_bandwidth:  {args.comm_bandwidth}")
    print(f"device:          {device}")
    print("=" * 60)

    actors = load_actors(args.checkpoint, obs_size, action_size, num_agents, device)

    print(f"[eval] connecting to Unity (behavior prefix='{behavior_name_prefix}')...")
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
            raise RuntimeError(f"Behavior '{behavior_name_prefix}' not found. Available: {available}")

        spec = env.behavior_specs[behavior_name]
        actual = spec.observation_specs[0].shape
        if actual != (obs_size,):
            raise RuntimeError(f"Obs shape mismatch: expected ({obs_size},), got {actual}")

        print(f"[eval] resolved behavior: '{behavior_name}'")
        print(f"[eval] obs_shape={actual}, action_spec={spec.action_spec}")
        print(f"[eval] running evaluation — EvaluationRunner will stop Play when target episodes reached")

        env_step = 0
        start = time.time()

        while env_step < args.max_env_steps:
            decision_steps, _terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                current_obs = np.zeros((num_agents, obs_size), dtype=np.float32)
                agent_ids = list(decision_steps.agent_id)
                n_present = min(len(agent_ids), num_agents)
                for slot_idx in range(n_present):
                    current_obs[slot_idx] = decision_steps[agent_ids[slot_idx]].obs[0]

                # Deterministic: actor forward pass, no Gaussian noise
                actions = np.zeros((num_agents, action_size), dtype=np.float32)
                with torch.no_grad():
                    for i in range(num_agents):
                        obs_t = torch.from_numpy(current_obs[i]).to(device)
                        a = actors[i](obs_t).cpu().numpy()
                        actions[i] = np.clip(a, 0.0, 1.0)

                # Gate masking (same rule as training)
                for i in range(num_agents):
                    if current_obs[i, IDX_GATE_STATE] != 0.0:
                        actions[i, 0] = 0.0

                env.set_actions(behavior_name, ActionTuple(continuous=actions.astype(np.float32)))
                env_step += 1

                if env_step % 5000 == 0:
                    elapsed = time.time() - start
                    print(f"[eval] env_step={env_step} elapsed={elapsed:.1f}s")

            env.step()

        print(f"[eval] max_env_steps reached ({args.max_env_steps}) without Play stopping")
        print(f"[eval] check the CSV output from EvaluationRunner anyway")

    except UnityCommunicatorStoppedException:
        print(f"[eval] Unity Play stopped — EvaluationRunner likely finished. "
              f"Check the CSV output configured on EvaluationRunner.")
    except KeyboardInterrupt:
        print(f"[eval] interrupted at env_step={env_step}")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[eval] error closing env: {e}")


if __name__ == "__main__":
    main()