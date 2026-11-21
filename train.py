"""
Training driver: PPO (primary) and DQN (secondary) from Stable-Baselines3.

Protocol per proposal Section 3.2:
  * offline training on the simulator (trace-driven, never on the testbed),
  * the cost weight lambda is swept (default 0.005 .. 0.1 per instance-second),
  * every configuration is trained under >= 5 random seeds,
  * training functions and evaluation functions/days are disjoint.

Usage examples:
  python3 train.py --algo ppo --lam 0.005 --seeds 0 1 2 3 4 --steps 40000
  python3 train.py --algo dqn --lam 0.02  --seeds 0 1 2  --steps 30000
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from envs.serverless_env import PreWarmEnv
from envs.workloads import (DEFAULT_PROFILE, FunctionProfile,
                            load_function_profile, train_arrival_fn)


def train_one(algo: str, lam: float, seed: int, steps: int, outdir: str,
              profile: FunctionProfile = DEFAULT_PROFILE) -> str:
    from stable_baselines3 import PPO, DQN
    env = PreWarmEnv(profile, lam=lam, arrival_fn=train_arrival_fn, T=3600, seed=seed)

    if algo == "ppo":
        model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64]),
                    n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.995)
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, seed=seed, verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64]),
                    learning_rate=1e-3, buffer_size=50_000, batch_size=64,
                    learning_starts=2_000, train_freq=4, target_update_interval=500,
                    gamma=0.995, exploration_final_eps=0.05)
    else:
        raise ValueError(algo)

    model.learn(total_timesteps=steps, progress_bar=False)
    tag = f"{algo}_lam{lam}_seed{seed}"
    path = os.path.join(outdir, tag)
    model.save(path if path.endswith(".zip") else path + ".zip")
    print(f"saved {path}.zip", flush=True)
    return path + ".zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--outdir", default="../models")
    ap.add_argument("--calibration", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "aws_calibration.csv"))
    ap.add_argument("--runtime", default="python3.12")
    ap.add_argument("--workload", default="cpu_loop")
    ap.add_argument("--memory-grid-mb", type=int, nargs="+")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    profile = load_function_profile(args.calibration, args.runtime, args.workload,
                                    args.memory_grid_mb)
    for s in args.seeds:
        train_one(args.algo, args.lam, s, args.steps, args.outdir, profile)


if __name__ == "__main__":
    main()
