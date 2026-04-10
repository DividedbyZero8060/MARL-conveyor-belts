"""
MADDPG training CLI entry point.

Usage:
    python scripts/train_maddpg.py --run-id=my_run --seed=42 --max-steps=2000000

Common invocations:
    # Smoke test (quick verification, Editor mode)
    python scripts/train_maddpg.py --run-id=smoke --max-steps=2000 \\
        --warmup=500 --buffer=10000 --summary-freq=200

    # Full single-area run, full observability, no comm
    python scripts/train_maddpg.py --run-id=maddpg_v1 --seed=42

    # Partial observability + comm bandwidth 1
    python scripts/train_maddpg.py --run-id=ablation_comm1_s42 \\
        --seed=42 --partial-obs --comm-bandwidth=1

    # Partial observability + comm bandwidth 3
    python scripts/train_maddpg.py --run-id=ablation_comm3_s256 \\
        --seed=256 --partial-obs --comm-bandwidth=3

The script connects to the Unity Editor by default. Pass --env-path to
connect to a standalone build instead.

Note on Unity scene state: before launching this script, ensure that all
three SortingAgent BehaviorParameters are set to:
    Behavior Type    = Default
    Continuous Actions = action_size_for_bandwidth(comm_bandwidth)
    Discrete Branches  = 0
and that EnvironmentManager.PartialObservability matches the --partial-obs
flag passed here.
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when invoked from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from training.maddpg.config import (
    MaddpgConfig,
    DEFAULT_CONFIG,
    FULL_OBS_SIZE,
    PARTIAL_OBS_SIZE,
)
from training.maddpg.maddpg_trainer import MaddpgTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MADDPG trainer for the cooperative sorting task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Identification
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="Unique identifier for this run. Used for log/checkpoint dirs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (torch, numpy, random, replay buffer, Unity).",
    )

    # ---- Observation / action mode
    parser.add_argument(
        "--partial-obs", action="store_true",
        help="Use partial observability (34 base floats). Default is full obs (38).",
    )
    parser.add_argument(
        "--comm-bandwidth", type=int, default=0, choices=[0, 1, 3],
        help="Communication channel size. 0=no comm, 1=1 float, 3=3 floats. "
             "Comm variants require --partial-obs.",
    )

    # ---- Training schedule
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_CONFIG.max_steps,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_CONFIG.warmup_transitions,
        help="Min replay buffer fill before gradient updates begin.",
    )
    parser.add_argument(
        "--buffer", type=int, default=DEFAULT_CONFIG.replay_buffer_capacity,
        help="Replay buffer capacity.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG.batch_size,
        help="Mini-batch size for gradient updates.",
    )
    parser.add_argument(
        "--summary-freq", type=int, default=DEFAULT_CONFIG.summary_freq,
        help="Steps between TensorBoard log writes.",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=DEFAULT_CONFIG.checkpoint_freq,
        help="Steps between checkpoint saves.",
    )

    # ---- Unity connection
    parser.add_argument(
        "--env-path", type=str, default=None,
        help="Path to a standalone Unity build. If omitted, connects to the Editor.",
    )
    parser.add_argument(
        "--no-graphics", action="store_true", default=True,
        help="Run standalone build headless. Ignored when connecting to Editor.",
    )
    parser.add_argument(
        "--graphics", dest="no_graphics", action="store_false",
        help="Force standalone build to render graphics. Slower; for debugging.",
    )

    # ---- Output
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Parent dir for logs/ and checkpoints/ subdirs.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=[None, "cpu", "cuda"],
        help="Torch device. If omitted, auto-detects (cuda if available).",
    )

    return parser.parse_args()


def determine_obs_size(partial_obs: bool, comm_bandwidth: int) -> int:
    """
    Compute the per-agent observation size from the mode flags.

    Layouts (must match the C# CollectObservations layout in
    Assets/Scripts/Agents/SortingAgent.Observations.cs):

        full obs, no comm           -> 38
        partial obs, no comm        -> 34
        partial obs, comm bandwidth=1 -> 34 + 2 = 36
        partial obs, comm bandwidth=3 -> 34 + 6 = 40

    Comm in full-obs mode is not supported by the workflow — full obs
    already exposes peer state, making explicit messaging redundant.
    """
    if not partial_obs:
        if comm_bandwidth != 0:
            raise ValueError(
                "Communication channels are only supported in partial-obs mode. "
                "Pass --partial-obs together with --comm-bandwidth > 0."
            )
        return FULL_OBS_SIZE

    base = PARTIAL_OBS_SIZE
    if comm_bandwidth == 0:
        return base
    if comm_bandwidth == 1:
        return base + 2  # 36
    if comm_bandwidth == 3:
        return base + 6  # 40
    raise ValueError(f"Unsupported comm_bandwidth: {comm_bandwidth}")


def main() -> None:
    args = parse_args()

    obs_size = determine_obs_size(args.partial_obs, args.comm_bandwidth)

    cfg = MaddpgConfig(
        replay_buffer_capacity=args.buffer,
        warmup_transitions=args.warmup,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        summary_freq=args.summary_freq,
        checkpoint_freq=args.checkpoint_freq,
    )

    print("=" * 60)
    print(f"MADDPG training run: {args.run_id}")
    print("=" * 60)
    print(f"seed:            {args.seed}")
    print(f"partial_obs:     {args.partial_obs}")
    print(f"comm_bandwidth:  {args.comm_bandwidth}")
    print(f"obs_size:        {obs_size}")
    print(f"max_steps:       {args.max_steps}")
    print(f"warmup:          {args.warmup}")
    print(f"buffer capacity: {args.buffer}")
    print(f"env_path:        {args.env_path or '(Unity Editor)'}")
    print(f"results_dir:     {args.results_dir}")
    print("=" * 60)

    trainer = MaddpgTrainer(
        config=cfg,
        obs_size=obs_size,
        comm_bandwidth=args.comm_bandwidth,
        run_id=args.run_id,
        seed=args.seed,
        results_dir=args.results_dir,
        device=args.device,
    )

    try:
        trainer.run(env_path=args.env_path, no_graphics=args.no_graphics)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()