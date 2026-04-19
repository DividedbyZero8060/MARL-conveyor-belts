"""
Evaluate a trained MADDPG policy by running N deterministic episodes
and reading the sort-accuracy metric directly from the Unity-side
StatsRecorder (via DebugOverlay → StatsSideChannel).

Usage:
    python scripts/evaluate_maddpg.py \
        --checkpoint=results/checkpoints/ablation_nocomm_level2/checkpoint_step500000.pt \
        --partial-obs --comm-bandwidth=0 --episodes=100

The script:
  - Loads the saved actor networks from the checkpoint.
  - Connects to a Unity Editor in Play mode (or a standalone build).
  - Runs N episodes with sigma=0 (no exploration noise) and the action-mask
    rule applied identically to training (force action[0]=0 when gate not retracted).
  - Listens on the StatsSideChannel for `Environment/SortAccuracy` (and
    related metrics) emitted by C# DebugOverlay at episode end.
  - Aggregates and prints summary statistics + a CSV row for the thesis table.

Important: the Unity scene must match the bandwidth this checkpoint was
trained on (Level2_nocomm.unity for bandwidth=0, etc). Action and obs
dims are inferred from the checkpoint and verified against Unity.

Note on comparability: 100 deterministic episodes give relatively tight
estimates of mean accuracy (stderr ~1-2% at the 50% accuracy level).
For tighter intervals, increase --episodes.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Make the project root importable when invoked from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress mlagents_envs verbose logging
logging.getLogger("mlagents_envs").setLevel(logging.WARNING)

import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

from training.maddpg.config import (
    MaddpgConfig,
    DEFAULT_CONFIG,
    action_size_for_bandwidth,
    IDX_GATE_STATE,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
)
from training.maddpg.networks import Actor

# ============================================================
# StatsSideChannel — receives metrics from Unity StatsRecorder
# ============================================================
#
# ML-Agents' StatsSideChannel UUID. This is hardcoded in the C# side
# (Unity.MLAgents.SideChannels.StatsSideChannel) — must match exactly.
STATS_SIDE_CHANNEL_UUID = uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520")


class StatsSideChannelReader(SideChannel):
    """
    Receives stat messages from C# StatsRecorder.Add() calls.

    Message format (from Unity ML-Agents source):
      string  metric_name
      float   value
      int32   aggregation_method   (0=Average, 1=MostRecent, 2=Sum, 3=Histogram)
    """

    def __init__(self):
        super().__init__(STATS_SIDE_CHANNEL_UUID)
        # Per-metric accumulator: list of all values received this episode
        # (or across episodes — we slice by episode boundaries externally).
        self._messages: Dict[str, List[float]] = defaultdict(list)

    def on_message_received(self, msg: IncomingMessage) -> None:
        try:
            metric_name = msg.read_string()
            value = msg.read_float32()
            _ = msg.read_int32()  # aggregation method, ignored
            self._messages[metric_name].append(float(value))
        except Exception as e:
            print(f"[StatsSideChannelReader] failed to parse message: {e}")

    def consume_all(self) -> Dict[str, List[float]]:
        """Return all messages received so far and clear the buffer."""
        result = dict(self._messages)
        self._messages = defaultdict(list)
        return result

    def queue_message_to_send(self, msg: OutgoingMessage) -> None:
        # We never send messages to Unity from this channel.
        pass


# ============================================================
# Helpers
# ============================================================

def determine_obs_size(partial_obs: bool, comm_bandwidth: int) -> int:
    """Same logic as train_maddpg.py."""
    if not partial_obs:
        if comm_bandwidth != 0:
            raise ValueError("Comm requires --partial-obs.")
        return FULL_OBS_SIZE
    base = PARTIAL_OBS_SIZE
    if comm_bandwidth == 0:
        return base
    if comm_bandwidth == 1:
        return base + 2
    if comm_bandwidth == 3:
        return base + 6
    raise ValueError(f"Unsupported comm_bandwidth: {comm_bandwidth}")


def load_actors(
        checkpoint_path: Path,
        obs_size: int,
        action_size: int,
        num_agents: int,
        device: torch.device,
        config: MaddpgConfig,
) -> List[Actor]:
    """
    Load actor networks from a MaddpgTrainer checkpoint.

    Checkpoint structure (from MaddpgTrainer.save_checkpoint):
      - 'actors': list of state_dicts, one per agent
      - 'critics', 'critics2', 'target_actors', etc: not needed for evaluation
      - 'env_step', 'config', 'update_count': metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "actors" not in checkpoint:
        raise KeyError(
            f"Checkpoint missing 'actors' key. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    actor_state_dicts = checkpoint["actors"]
    if len(actor_state_dicts) != num_agents:
        raise ValueError(
            f"Checkpoint has {len(actor_state_dicts)} actor state_dicts, "
            f"expected {num_agents}."
        )

    # Print useful metadata for sanity
    saved_step = checkpoint.get("env_step", "unknown")
    print(f"[evaluate_maddpg] checkpoint env_step: {saved_step}")

    actors: List[Actor] = []
    for i in range(num_agents):
        actor = Actor(obs_size, action_size, config).to(device)
        actor.load_state_dict(actor_state_dicts[i])
        actor.eval()
        actors.append(actor)
    return actors


def run_evaluation(
        actors: List[Actor],
        env: UnityEnvironment,
        behavior_name: str,
        stats_channel: StatsSideChannelReader,
        num_agents: int,
        obs_size: int,
        action_size: int,
        target_episodes: int,
        device: torch.device,
        print_every: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Run target_episodes deterministic episodes. Returns dict mapping metric
    name to numpy array of per-episode values.
    """
    completed_episodes = 0
    per_episode_metrics: Dict[str, List[float]] = defaultdict(list)
    cumulative_reward = 0.0
    per_episode_rewards: List[float] = []

    print(f"[evaluate_maddpg] starting evaluation, target {target_episodes} episodes")

    while completed_episodes < target_episodes:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Drain side-channel messages (these arrive between env.step calls)
        new_metrics = stats_channel.consume_all()

        # On terminal step: episode ended. Slice in any metrics that arrived
        # since last episode boundary, taking the most recent value per metric
        # (StatsRecorder typically emits once per episode end, so most recent = this episode).
        if len(terminal_steps) > 0:
            completed_episodes += 1

            for metric_name, values in new_metrics.items():
                if values:
                    # Take last value as this episode's value.
                    per_episode_metrics[metric_name].append(values[-1])

            # Track per-episode reward sum (one agent's stream is enough)
            agent_id = list(terminal_steps.agent_id)[0]
            cumulative_reward += float(terminal_steps[agent_id].reward)
            per_episode_rewards.append(cumulative_reward)
            cumulative_reward = 0.0

            if completed_episodes % print_every == 0:
                last_acc = per_episode_metrics.get("Environment/SortAccuracy", [])
                last_rew = per_episode_rewards
                acc_str = f"{np.mean(last_acc[-print_every:]):.3f}" if last_acc else "n/a"
                rew_str = f"{np.mean(last_rew[-print_every:]):+.3f}" if last_rew else "n/a"
                print(
                    f"  episodes {completed_episodes:>4}/{target_episodes}  "
                    f"recent_acc={acc_str}  recent_rew={rew_str}"
                )

        # Decision step: select deterministic actions
        if len(decision_steps) > 0:
            current_obs = np.zeros((num_agents, obs_size), dtype=np.float32)
            agent_ids = list(decision_steps.agent_id)
            for slot_idx in range(min(len(agent_ids), num_agents)):
                aid = agent_ids[slot_idx]
                current_obs[slot_idx] = decision_steps[aid].obs[0]
                cumulative_reward += float(decision_steps[aid].reward) / num_agents

            # Deterministic actor forward (no noise, no clipping needed)
            actions = np.zeros((num_agents, action_size), dtype=np.float32)
            with torch.no_grad():
                for i in range(num_agents):
                    obs_tensor = torch.from_numpy(current_obs[i]).to(device)
                    actions[i] = actors[i](obs_tensor).cpu().numpy()

            # Action masking (same rule as training): zero gate action when gate not retracted
            for i in range(num_agents):
                gate_state = current_obs[i, IDX_GATE_STATE]
                if gate_state != 0.0:
                    actions[i, 0] = 0.0

            env.set_actions(behavior_name, ActionTuple(continuous=actions))

        env.step()

    # Convert to numpy arrays
    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in per_episode_metrics.items()
    }, np.asarray(per_episode_rewards, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MADDPG checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--partial-obs", action="store_true",
        help="Use partial observability. Must match the training run.",
    )
    parser.add_argument(
        "--comm-bandwidth", type=int, default=0, choices=[0, 1, 3],
        help="Communication bandwidth. Must match the training run.",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of deterministic episodes to evaluate.",
    )
    parser.add_argument(
        "--env-path", type=str, default=None,
        help="Standalone Unity build path. If omitted, connects to Editor.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=[None, "cpu", "cuda"],
    )
    parser.add_argument(
        "--csv-out", type=str, default=None,
        help="Optional CSV path for per-episode metrics. If omitted, prints summary only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    obs_size = determine_obs_size(args.partial_obs, args.comm_bandwidth)
    action_size = action_size_for_bandwidth(args.comm_bandwidth)
    num_agents = DEFAULT_CONFIG.num_agents

    print("=" * 60)
    print(f"MADDPG evaluation: {args.checkpoint}")
    print("=" * 60)
    print(f"partial_obs:    {args.partial_obs}")
    print(f"comm_bandwidth: {args.comm_bandwidth}")
    print(f"obs_size:       {obs_size}")
    print(f"action_size:    {action_size}")
    print(f"episodes:       {args.episodes}")
    print(f"device:         {device}")
    print("=" * 60)

    # Load actors from checkpoint
    actors = load_actors(
        Path(args.checkpoint), obs_size, action_size,
        num_agents, device, DEFAULT_CONFIG,
    )
    print(f"[evaluate_maddpg] loaded {num_agents} actor networks.")

    # Set up Unity env with stats side channel
    stats_channel = StatsSideChannelReader()
    env_kwargs = {"side_channels": [stats_channel]}
    if args.env_path is not None:
        env_kwargs["file_name"] = args.env_path
        env_kwargs["no_graphics"] = True
    env = UnityEnvironment(**env_kwargs)

    try:
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]
        spec = env.behavior_specs[behavior_name]
        actual_obs_shape = spec.observation_specs[0].shape
        if actual_obs_shape != (obs_size,):
            raise RuntimeError(
                f"Obs shape mismatch: expected ({obs_size},), Unity reports {actual_obs_shape}. "
                f"Wrong scene loaded for this checkpoint's bandwidth?"
            )
        print(f"[evaluate_maddpg] connected, behavior='{behavior_name}', obs_shape={actual_obs_shape}")

        per_episode_metrics, per_episode_rewards = run_evaluation(
            actors, env, behavior_name, stats_channel,
            num_agents, obs_size, action_size,
            args.episodes, device,
        )
    finally:
        env.close()

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 60)
    print(f"Evaluation complete: {len(per_episode_rewards)} episodes")
    print("=" * 60)

    # Per-episode reward (always available)
    rewards = per_episode_rewards
    print(f"Mean episode reward:    {np.mean(rewards):+.4f} ± {np.std(rewards):.4f}")
    print(f"Median episode reward:  {np.median(rewards):+.4f}")
    print(f"Min / Max:              {np.min(rewards):+.4f} / {np.max(rewards):+.4f}")
    print()

    # SortAccuracy from side channel (the thesis number)
    if "Environment/SortAccuracy" in per_episode_metrics:
        acc = per_episode_metrics["Environment/SortAccuracy"]
        print(f"Mean SortAccuracy:      {np.mean(acc):.4f} ± {np.std(acc):.4f}")
        print(f"Median SortAccuracy:    {np.median(acc):.4f}")
        print(f"Min / Max SortAccuracy: {np.min(acc):.4f} / {np.max(acc):.4f}")
    else:
        print("WARNING: Environment/SortAccuracy not received via side channel.")
        print("  Possible causes:")
        print("  - DebugOverlay not in scene")
        print("  - DebugOverlay not calling Academy.Instance.StatsRecorder.Add()")
        print("  - Metric name string differs (check exact key in C# code)")
        print(f"  Received metric names: {list(per_episode_metrics.keys())}")
    print()

    # Other metrics that may have come through
    for name in sorted(per_episode_metrics.keys()):
        if name == "Environment/SortAccuracy":
            continue
        values = per_episode_metrics[name]
        if len(values) > 0:
            print(f"  {name}: mean={np.mean(values):+.4f}, n={len(values)}")

    # CSV output
    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_keys = sorted(per_episode_metrics.keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"] + all_keys)
            for ep in range(len(rewards)):
                row = [ep + 1, rewards[ep]]
                for k in all_keys:
                    vals = per_episode_metrics[k]
                    row.append(vals[ep] if ep < len(vals) else "")
                writer.writerow(row)
        print(f"[evaluate_maddpg] wrote per-episode CSV: {out_path}")


if __name__ == "__main__":
    main()