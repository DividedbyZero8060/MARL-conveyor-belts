"""
CLI entry point for Independent DQN training.

Mirrors scripts/train_maddpg.py in structure and arguments.

Usage:
    python scripts/train_dqn.py --run-id=dqn_optionc_s42 --seed=42 \
        --max-steps=100000 --warmup=2000 --buffer=50000 \
        --batch-size=64 --summary-freq=2000 --checkpoint-freq=50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `training` package is importable
# regardless of which directory the script is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.dqn.config import DqnConfig, FULL_OBS_SIZE, PARTIAL_OBS_SIZE
from training.dqn.dqn_trainer import DqnTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent DQN trainer CLI")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Unique run identifier; used for log/checkpoint directories.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for torch, numpy, random, Unity env, and buffer RNG.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps in the DqnConfig. Default uses the config value.")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Override warmup_transitions (per-agent) in the DqnConfig.")
    parser.add_argument("--buffer", type=int, default=None,
                        help="Override replay_buffer_capacity (per-agent) in the DqnConfig.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size in the DqnConfig.")
    parser.add_argument("--summary-freq", type=int, default=None,
                        help="Override summary_freq in the DqnConfig.")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Override checkpoint_freq in the DqnConfig.")
    parser.add_argument("--partial-obs", action="store_true",
                        help="If set, use partial-obs size (34). Default is full-obs (38).")
    parser.add_argument("--env-path", type=str, default=None,
                        help="Path to a standalone Unity build. If omitted, connects to the Editor.")
    parser.add_argument("--no-graphics", action="store_true",
                        help="Pass --no-graphics to the Unity environment (headless; build only).")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Parent directory for logs/ and checkpoints/ subdirs.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DqnConfig:
    """Build a DqnConfig with any CLI overrides applied."""
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.warmup is not None:
        overrides["warmup_transitions"] = args.warmup
    if args.buffer is not None:
        overrides["replay_buffer_capacity"] = args.buffer
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.summary_freq is not None:
        overrides["summary_freq"] = args.summary_freq
    if args.checkpoint_freq is not None:
        overrides["checkpoint_freq"] = args.checkpoint_freq

    # Use dataclasses.replace since DqnConfig is frozen.
    import dataclasses
    cfg = DqnConfig()
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    obs_size = PARTIAL_OBS_SIZE if args.partial_obs else FULL_OBS_SIZE

    print(f"[train_dqn] run_id={args.run_id} seed={args.seed}")
    print(f"[train_dqn] obs_size={obs_size} (partial_obs={args.partial_obs})")
    print(f"[train_dqn] max_steps={cfg.max_steps} warmup={cfg.warmup_transitions} "
          f"buffer={cfg.replay_buffer_capacity} batch={cfg.batch_size}")

    trainer = DqnTrainer(
        config=cfg,
        obs_size=obs_size,
        run_id=args.run_id,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    try:
        trainer.run(env_path=args.env_path, no_graphics=args.no_graphics)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()