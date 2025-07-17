#!/usr/bin/env python3
"""
run_custom_sequences.py

Convenience launcher for Sample Factory HASARD/VizDoom sequences.

This script lets you define and iterate over multiple task sequences—varying
order, length, and repetitions—without re‑typing long command lines. It wraps
`sample_factory/doom/train_vizdoom.py` and forwards all the usual flags plus
per‑sequence Weights & Biases bookkeeping.

Example usage
-------------
Run every sequence defined in the script (good for an overnight sweep)::

    python run_hasard_sequences.py --with_wandb --wandb_user YOUR_WANDB_NAME

Run just the movement‑focused curriculum on GPU 0::

    CUDA_VISIBLE_DEVICES=0 python run_hasard_sequences.py \
        --device cuda:0 --only movement_then_all --with_wandb \
        --wandb_user YOUR_WANDB_NAME

Add or modify sequences at the bottom of this file—no other changes needed.
"""

import argparse
import subprocess
import sys
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def build_sequences() -> Dict[str, List[str]]:
    """Return a dict of <sequence_name> -> [env1, env2, ...]"""

    ARMS_DEALER = ["arms_dealer"]
    CHAINSAW = ["chainsaw"]
    FLOOR_IS_LAVA = ["floor_is_lava"]
    HEALTH_GATHERING = ["health_gathering"]
    HIDE_AND_SEEK = ["hide_and_seek"]
    PITFALL = ["pitfall"]
    RAISE_THE_ROOF = ["raise_the_roof"]
    RUN_AND_GUN = ["run_and_gun"]
    
    ALL_TASKS = [
        "arms_dealer",
        "chainsaw",
        "floor_is_lava",
        "health_gathering",
        "hide_and_seek",
        "pitfall",
        "raise_the_roof",
        "run_and_gun", 
    ]
    
    TARGETING_TASKS = [
        "health_gathering",
        "arms_dealer"
    ]
    
    KILL_TASKS = [
        "chainsaw",
        "run_and_gun",
    ]
    
    SURVIVE_TASKS = [
        "floor_is_lava",
        "hide_and_seek",
        "raise_the_roof",
    ]

    sequences = {
        "all_tasks_once": ALL_TASKS,
        "movement_then_all": PITFALL + ALL_TASKS,
        "just_targeting": TARGETING_TASKS,
        "kill_then_survive": KILL_TASKS + SURVIVE_TASKS,
        "survive_then_kill": SURVIVE_TASKS + KILL_TASKS,
        "movement_then_kill": PITFALL + KILL_TASKS,
        "movement_then_survive": PITFALL + SURVIVE_TASKS,
        "kill_then_movement": KILL_TASKS + PITFALL,
        "survive_then_movement": SURVIVE_TASKS + PITFALL,
        "targeting_then_kill": TARGETING_TASKS + KILL_TASKS,
        "targeting_then_survive": TARGETING_TASKS + SURVIVE_TASKS,
        "targeting_then_movement": TARGETING_TASKS + PITFALL,
    }

    return sequences


# -----------------------------------------------------------------------------
# Command‑line assembly & launch
# -----------------------------------------------------------------------------

def run_sequence(name: str, envs: List[str], args: argparse.Namespace) -> None:
    envs_arg = " ".join(envs)
    
    cmd = (
        f"python sample_factory/doom/train_vizdoom.py "
        f"--algo {args.algo} "
        f"--train_for_env_steps {args.train_steps:_} "
        f"--env_steps_per_env {args.env_steps_per_env:_} "
        f"--seed {args.seed} "
        f"--with_wandb {str(args.with_wandb)} "
        f"--log_heatmap {str(args.log_heatmap)} "
        f"--device {args.device} "
        f"--wandb_user {args.wandb_user} "
        f"--wandb_project {args.wandb_project} "
        f"--wandb_tags {name} "
        f"--envs {envs_arg}"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Launching sequence '{name}' …")
    print("  ", cmd, "\n", flush=True)

    subprocess.run(shlex.split(cmd), check=True)


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch‑runner for HASARD sequences")

    p.add_argument("--algo", default="PPO", help="RL algorithm to use (default: PPO)")
    p.add_argument("--train_steps", type=int, default=7_000_000,
                   help="Total env frames across the whole sequence")
    p.add_argument("--env_steps_per_env", type=int, default=1_000_000,
                   help="Max frames per individual scenario before moving on to the next one")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--with_wandb", action="store_true", default=False)
    p.add_argument("--log_heatmap", action="store_true", default=False)
    p.add_argument("--device", default="cpu", help="cpu, gpu")

    p.add_argument("--wandb_user", required=True, help="Your W&B username")
    p.add_argument("--wandb_project", default="doomination_sequence_eval",
                   help="Base project name; sequence name is appended")

    p.add_argument("--only", nargs="*", metavar="SEQ",
                   help="Run only these sequences (default: all defined)")
    
    # some arguments to make sure it runs on my cpu
    p.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    p.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    p.add_argument("--worker_num_splits", type=int, default=2,
                   help="Number of splits per worker (default: 1, no splitting)")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    sequences = build_sequences()

    # Determine which sequences to run
    targets = args.only if args.only else sequences.keys()

    for seq_name in targets:
        if seq_name not in sequences:
            print(f"\n[!] Unknown sequence '{seq_name}'. Available: {list(sequences)}", file=sys.stderr)
            sys.exit(1)

        envs = sequences[seq_name]
        num_envs = len(envs)
        
        # Dynamically compute per-env step limit
        args.env_steps_per_env = args.train_steps // num_envs

        run_sequence(seq_name, envs, args)


if __name__ == "__main__":
    main()
