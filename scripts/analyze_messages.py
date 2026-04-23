"""
Message semantic analysis for the communication ablation (Step 19).

Loads a trained MATD3 comm-variant checkpoint, runs N frozen-policy
episodes, and computes the semantic content of the emitted messages:

    1. MI(message_vector, package_type) per sender agent
    2. PCA on 3-float messages (comm3): variance explained by PC1
    3. Message distribution entropy
    4. Distribution plots colored by package type

Output:
    results/msg_analysis/<run_id>_messages.csv       raw (message, type) tuples
    results/msg_analysis/<run_id>_summary.txt        MI/PCA/entropy summary
    results/msg_analysis/<run_id>_dist_agent<N>.png  distribution plots

Usage:
    python scripts/analyze_messages.py \\
        --checkpoint=results/checkpoints/ablation_comm3_level2/checkpoint_step500000.pt \\
        --partial-obs --comm-bandwidth=3 --episodes=500 \\
        --run-id=ablation_comm3_level2

Interpretation:
  - MI > 0.3: messages strongly encode package type. Learned protocol.
  - 0.05 < MI < 0.3: weak structure. Partial protocol.
  - MI < 0.05: messages are noise. Channel ignored.

  - PC1 variance > 80% (for comm3): only 1 effective dimension. Bandwidth 3
    was wasteful; bandwidth 1 would have been equivalent.
  - PC1 variance ~33%: messages use all 3 dimensions independently.

  - Message entropy < uniform entropy: structured signal.
  - Message entropy ≈ uniform entropy: random.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.getLogger("mlagents_envs").setLevel(logging.ERROR)
logging.getLogger("mlagents_envs.side_channel.side_channel_manager").setLevel(logging.ERROR)

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from training.maddpg.config import (
    MaddpgConfig,
    DEFAULT_CONFIG,
    action_size_for_bandwidth,
    IDX_GATE_STATE,
    IDX_PACKAGE_SLOTS_START,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
)
from training.maddpg.networks import Actor


# =====================================================================
# Package type decoding from observation
# =====================================================================
#
# Each package slot has 5 floats in the observation vector:
#   [present, distance, dest_one_hot[0], dest_one_hot[1], dest_one_hot[2]]
#
# Slot 0 starts at IDX_PACKAGE_SLOTS_START = 6.
# We read slot 0's destination one-hot at offsets +2, +3, +4.
# Returns type 0/1/2 (DestA/DestB/DestC) or -1 if slot is empty.

def decode_slot0_package_type(obs: np.ndarray) -> int:
    """
    Decode the package type currently in slot 0 of this agent's observation.
    Returns 0/1/2 for DestA/DestB/DestC, or -1 if slot 0 is empty.
    """
    slot0_start = IDX_PACKAGE_SLOTS_START
    present = obs[slot0_start + 0]
    if present < 0.5:
        return -1
    dest_one_hot = obs[slot0_start + 2 : slot0_start + 5]
    # argmax only valid if one-hot is well-formed; allow for noise
    if dest_one_hot.sum() < 0.5:
        return -1
    return int(np.argmax(dest_one_hot))


# =====================================================================
# Helpers
# =====================================================================

def determine_obs_size(partial_obs: bool, comm_bandwidth: int) -> int:
    """Match train_maddpg.py's logic."""
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
    """Load actor networks from MaddpgTrainer checkpoint (uses 'actors' list)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "actors" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'actors' key. Keys: {list(checkpoint.keys())}")

    actor_state_dicts = checkpoint["actors"]
    if len(actor_state_dicts) != num_agents:
        raise ValueError(
            f"Checkpoint has {len(actor_state_dicts)} actors, expected {num_agents}."
        )

    saved_step = checkpoint.get("env_step", "unknown")
    print(f"[analyze_messages] checkpoint env_step: {saved_step}")

    actors: List[Actor] = []
    for i in range(num_agents):
        actor = Actor(obs_size, action_size, config).to(device)
        actor.load_state_dict(actor_state_dicts[i])
        actor.eval()
        actors.append(actor)
    return actors


# =====================================================================
# Metric computations
# =====================================================================

def compute_mi_per_agent(
        messages_per_agent: List[np.ndarray],
        types_per_agent: List[np.ndarray],
        n_neighbors: int = 3,
) -> Dict[int, float]:
    """
    Compute joint MI(message_vector, package_type) for each agent.

    Uses sklearn's mutual_info_classif on features projected to 1-D via
    PCA, giving a single MI estimate that respects feature dependence.
    Bounded above by log(n_classes). For 3-class type labels: max ≈ 1.099 nats.
    """
    if not _HAS_SKLEARN:
        print("[analyze_messages] sklearn not installed; skipping MI computation.")
        return {}

    mi_results: Dict[int, float] = {}
    for agent_idx, (messages, types) in enumerate(
            zip(messages_per_agent, types_per_agent)
    ):
        if len(messages) == 0:
            mi_results[agent_idx] = 0.0
            continue

        mask = types >= 0
        if mask.sum() < 10:
            mi_results[agent_idx] = 0.0
            continue

        X = messages[mask]
        y = types[mask]

        # Joint MI estimate: project to 1-D first (PCA top component)
        # to avoid the sum-of-marginals over-counting bug.
        if X.shape[1] > 1:
            pca = PCA(n_components=1)
            X_1d = pca.fit_transform(X)
        else:
            X_1d = X

        # Now a single-feature MI; returns array of length 1
        mi = mutual_info_classif(X_1d, y, n_neighbors=n_neighbors, random_state=0)
        mi_results[agent_idx] = float(mi[0])

    return mi_results


def compute_pca_per_agent(
    messages_per_agent: List[np.ndarray],
    bandwidth: int,
) -> Dict[int, Dict[str, float]]:
    """
    For comm3 agents, run PCA on 3D messages and return variance explained
    by each component. For comm1, trivially return {PC1: 1.0}.
    """
    pca_results: Dict[int, Dict[str, float]] = {}

    for agent_idx, messages in enumerate(messages_per_agent):
        if bandwidth == 1 or len(messages) < bandwidth:
            pca_results[agent_idx] = {"PC1": 1.0}
            continue

        if not _HAS_SKLEARN:
            pca_results[agent_idx] = {}
            continue

        if bandwidth == 3:
            pca = PCA(n_components=3)
            pca.fit(messages)
            var = pca.explained_variance_ratio_
            pca_results[agent_idx] = {
                "PC1": float(var[0]),
                "PC2": float(var[1]),
                "PC3": float(var[2]),
            }

    return pca_results


def compute_entropy_per_agent(
    messages_per_agent: List[np.ndarray],
    n_bins: int = 20,
) -> Dict[int, float]:
    """
    Approximate differential entropy of each agent's message distribution
    via histogram discretization per dimension, then sum.

    Uniform entropy over [0, 1] is 0 nats; structured distributions
    should have negative entropy (concentrated mass).

    We use a normalized "entropy ratio" instead: H(messages) / H_uniform,
    where H_uniform is the uniform histogram entropy. Ratio close to 1
    means uniform (random); ratio << 1 means structured.
    """
    entropy_results: Dict[int, float] = {}

    for agent_idx, messages in enumerate(messages_per_agent):
        if len(messages) < n_bins:
            entropy_results[agent_idx] = 1.0
            continue

        total_entropy = 0.0
        uniform_entropy = np.log(n_bins)  # bits in nats

        for dim in range(messages.shape[1]):
            hist, _ = np.histogram(
                messages[:, dim], bins=n_bins, range=(0, 1), density=False
            )
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            dim_entropy = -np.sum(probs * np.log(probs))
            total_entropy += dim_entropy

        avg_entropy_per_dim = total_entropy / messages.shape[1]
        entropy_results[agent_idx] = float(avg_entropy_per_dim / uniform_entropy)

    return entropy_results


# =====================================================================
# Main collection loop
# =====================================================================

def collect_message_type_pairs(
    actors: List[Actor],
    env: UnityEnvironment,
    behavior_name: str,
    num_agents: int,
    obs_size: int,
    action_size: int,
    comm_bandwidth: int,
    target_episodes: int,
    device: torch.device,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run target_episodes frozen-policy episodes, collect (message, type) pairs.
    Returns two lists (one per agent):
        messages_per_agent[i]: (N, bandwidth) array of emitted messages
        types_per_agent[i]:    (N,) array of package types in that agent's slot 0
    """
    messages_collected: List[List[np.ndarray]] = [[] for _ in range(num_agents)]
    types_collected: List[List[int]] = [[] for _ in range(num_agents)]

    completed_episodes = 0

    print(f"[analyze_messages] collecting message-type pairs for {target_episodes} episodes...")

    while completed_episodes < target_episodes:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            completed_episodes += 1
            if completed_episodes % 50 == 0:
                total_samples = sum(len(ms) for ms in messages_collected)
                print(f"  episodes {completed_episodes:>4}/{target_episodes} "
                      f"samples={total_samples}")

        if len(decision_steps) > 0:
            current_obs = np.zeros((num_agents, obs_size), dtype=np.float32)
            agent_ids = list(decision_steps.agent_id)
            n_present = min(len(agent_ids), num_agents)
            for slot_idx in range(n_present):
                aid = agent_ids[slot_idx]
                current_obs[slot_idx] = decision_steps[aid].obs[0]

            # Actor forward pass (deterministic, no noise for eval)
            actions = np.zeros((num_agents, action_size), dtype=np.float32)
            with torch.no_grad():
                for i in range(num_agents):
                    obs_tensor = torch.from_numpy(current_obs[i]).to(device)
                    actions[i] = actors[i](obs_tensor).cpu().numpy()

            # Extract message floats (action indices 1..1+bandwidth)
            # and collect per-agent (message, type) pairs.
            for i in range(num_agents):
                message = actions[i, 1:1 + comm_bandwidth].copy()
                pkg_type = decode_slot0_package_type(current_obs[i])
                messages_collected[i].append(message)
                types_collected[i].append(pkg_type)

            # Still apply gate mask to keep env dynamics realistic
            for i in range(num_agents):
                gate_state = current_obs[i, IDX_GATE_STATE]
                if gate_state != 0.0:
                    actions[i, 0] = 0.0

            env.set_actions(behavior_name, ActionTuple(continuous=actions))

        env.step()

    # Convert to numpy arrays
    messages_per_agent = [np.asarray(ms, dtype=np.float32) for ms in messages_collected]
    types_per_agent = [np.asarray(ts, dtype=np.int32) for ts in types_collected]

    return messages_per_agent, types_per_agent


# =====================================================================
# Plotting
# =====================================================================

def plot_message_distributions(
    messages_per_agent: List[np.ndarray],
    types_per_agent: List[np.ndarray],
    bandwidth: int,
    output_dir: Path,
    run_id: str,
) -> None:
    """
    For each agent, plot the message distribution colored by package type.
    For bandwidth=1: histogram of message value, stacked by type.
    For bandwidth=3: scatter plot of (msg[0], msg[1]), colored by type.
    """
    if not _HAS_MATPLOTLIB:
        print("[analyze_messages] matplotlib not installed; skipping plots.")
        return

    num_agents = len(messages_per_agent)

    for agent_idx in range(num_agents):
        messages = messages_per_agent[agent_idx]
        types = types_per_agent[agent_idx]

        mask = types >= 0
        if mask.sum() < 10:
            continue

        messages = messages[mask]
        types = types[mask]

        fig, ax = plt.subplots(figsize=(8, 6))

        if bandwidth == 1:
            for t, label in zip([0, 1, 2], ["DestA", "DestB", "DestC"]):
                type_mask = types == t
                if type_mask.sum() == 0:
                    continue
                ax.hist(
                    messages[type_mask, 0],
                    bins=20, range=(0, 1),
                    alpha=0.5, label=label,
                )
            ax.set_xlabel("Message value")
            ax.set_ylabel("Count")
            ax.set_title(f"Agent {agent_idx} message distribution (comm1, {run_id})")
            ax.legend()

        elif bandwidth == 3:
            for t, label in zip([0, 1, 2], ["DestA", "DestB", "DestC"]):
                type_mask = types == t
                if type_mask.sum() == 0:
                    continue
                ax.scatter(
                    messages[type_mask, 0],
                    messages[type_mask, 1],
                    alpha=0.3, s=8,
                    label=label,
                )
            ax.set_xlabel("message[0]")
            ax.set_ylabel("message[1]")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Agent {agent_idx} messages dims 0-1 (comm3, {run_id})")
            ax.legend()

        fig.tight_layout()
        out_path = output_dir / f"{run_id}_agent{agent_idx}_dist.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  wrote {out_path}")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse learned communication messages from a comm-variant MATD3 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--partial-obs", action="store_true",
                   help="Must match the training run.")
    p.add_argument("--comm-bandwidth", type=int, required=True, choices=[1, 3],
                   help="Must match the training run. bandwidth=0 has nothing to analyse.")
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of deterministic episodes to run for collection.")
    p.add_argument("--run-id", type=str, required=True,
                   help="Label for output files. Typically matches the checkpoint's run_id.")
    p.add_argument("--env-path", type=str, default=None,
                   help="Standalone Unity build. If omitted, connects to Editor.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    p.add_argument("--output-dir", type=str, default="results/msg_analysis",
                   help="Output directory for CSV / plots / summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    obs_size = determine_obs_size(args.partial_obs, args.comm_bandwidth)
    action_size = action_size_for_bandwidth(args.comm_bandwidth)
    num_agents = DEFAULT_CONFIG.num_agents
    bandwidth = args.comm_bandwidth

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Message analysis: {args.checkpoint}")
    print("=" * 60)
    print(f"comm_bandwidth:  {bandwidth}")
    print(f"obs_size:        {obs_size}")
    print(f"action_size:     {action_size}")
    print(f"episodes:        {args.episodes}")
    print(f"output_dir:      {output_dir}")
    print("=" * 60)

    # Load actors
    actors = load_actors(
        Path(args.checkpoint), obs_size, action_size,
        num_agents, device, DEFAULT_CONFIG,
    )

    # Connect to Unity
    env = UnityEnvironment(
        file_name=args.env_path,
        no_graphics=(args.env_path is not None),
    ) if args.env_path else UnityEnvironment()

    try:
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]
        spec = env.behavior_specs[behavior_name]
        actual_obs_shape = spec.observation_specs[0].shape
        if actual_obs_shape != (obs_size,):
            raise RuntimeError(
                f"Obs shape mismatch: expected ({obs_size},), Unity reports {actual_obs_shape}. "
                f"Wrong scene loaded for this bandwidth?"
            )
        print(f"[analyze_messages] connected, obs_shape={actual_obs_shape}")

        messages_per_agent, types_per_agent = collect_message_type_pairs(
            actors, env, behavior_name,
            num_agents, obs_size, action_size, bandwidth,
            args.episodes, device,
        )
    finally:
        env.close()

    # =================================================================
    # Metrics
    # =================================================================
    print()
    print("=" * 60)
    print("Message analysis results")
    print("=" * 60)

    # Sample counts per agent
    for i in range(num_agents):
        n_total = len(messages_per_agent[i])
        n_valid = (types_per_agent[i] >= 0).sum()
        print(f"Agent {i}: {n_total} total samples, {n_valid} with non-empty slot 0")

    print()
    print("--- MI(message_vector, package_type) ---")
    mi = compute_mi_per_agent(messages_per_agent, types_per_agent)
    for i in range(num_agents):
        val = mi.get(i, 0.0)
        # Max MI for 3-class labels is log(3) ≈ 1.099 nats
        # Scale thresholds against this ceiling.
        tag = "STRONG" if val > 0.5 else ("WEAK" if val > 0.1 else "NOISE")
        print(f"  Agent {i}: MI = {val:.4f} nats  [{tag}]  (ceiling: 1.099)")

    print()
    print("--- PCA on messages (comm3 only) ---")
    pca = compute_pca_per_agent(messages_per_agent, bandwidth)
    for i in range(num_agents):
        if bandwidth == 3 and i in pca and "PC3" in pca[i]:
            print(f"  Agent {i}: PC1={pca[i]['PC1']:.3f}  "
                  f"PC2={pca[i]['PC2']:.3f}  PC3={pca[i]['PC3']:.3f}")
            if pca[i]["PC1"] > 0.80:
                print(f"           → effective dim ≈ 1, bandwidth 3 mostly wasted")

    print()
    print("--- Message entropy ratio (1.0 = uniform/random, <1 = structured) ---")
    ent = compute_entropy_per_agent(messages_per_agent)
    for i in range(num_agents):
        val = ent.get(i, 1.0)
        tag = "STRUCTURED" if val < 0.90 else "UNIFORM"
        print(f"  Agent {i}: entropy ratio = {val:.4f}  [{tag}]")

    # =================================================================
    # Save raw data and summary
    # =================================================================
    print()
    print("--- Saving outputs ---")

    # Raw CSV per agent
    import csv
    for i in range(num_agents):
        out_path = output_dir / f"{args.run_id}_agent{i}_messages.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + [f"msg_{j}" for j in range(bandwidth)] + ["package_type"]
            writer.writerow(header)
            for step in range(len(messages_per_agent[i])):
                row = [step] + list(messages_per_agent[i][step]) + [int(types_per_agent[i][step])]
                writer.writerow(row)
        print(f"  wrote {out_path}")

    # Summary text
    summary_path = output_dir / f"{args.run_id}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Message analysis: {args.run_id}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Bandwidth: {bandwidth}\n")
        f.write(f"Episodes: {args.episodes}\n\n")
        f.write("MI(message, package_type):\n")
        for i in range(num_agents):
            f.write(f"  Agent {i}: {mi.get(i, 0.0):.4f} nats\n")
        f.write("\nPCA (comm3):\n")
        for i in range(num_agents):
            if bandwidth == 3 and i in pca and "PC3" in pca[i]:
                f.write(f"  Agent {i}: PC1={pca[i]['PC1']:.3f}  "
                        f"PC2={pca[i]['PC2']:.3f}  PC3={pca[i]['PC3']:.3f}\n")
        f.write("\nEntropy ratio (1.0 = uniform):\n")
        for i in range(num_agents):
            f.write(f"  Agent {i}: {ent.get(i, 1.0):.4f}\n")
    print(f"  wrote {summary_path}")

    # Plots
    plot_message_distributions(
        messages_per_agent, types_per_agent,
        bandwidth, output_dir, args.run_id,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()