#!/usr/bin/env python3
"""Trains a chunk-selection RL agent per domain x algorithm.

Unlike the old pipeline (one model per dataset ROW, ~1000 models for 250
rows x 4 algorithms, with no generalization to new questions), here a single
agent per (domain, algorithm) is trained over all rows of that domain - it
generalizes to new queries within the domain and is orders of magnitude
cheaper.

Usage:
    python train.py --dataset path/to/dataset.parquet --output-dir models/
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from codecarbon import EmissionsTracker
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from config import ALGO_ORDER, DEFAULT_MODELS_DIR
from env import ChunkSelectionEnv


class ChunkCountCallback(BaseCallback):
    """Logs to tensorboard how many chunks each episode selected
    (rollout/ep_chunks_selected) - the curve that shows whether the per-chunk
    cost (config.CHUNK_COST_BASE/GROWTH) is actually reducing selection over
    the course of training."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info and "n_selected" in ep_info:
                self.logger.record("rollout/ep_chunks_selected", ep_info["n_selected"])
        return True


ALGORITHMS = {
    "ppo": (PPO, "MultiInputPolicy", False),
    "recurrent_ppo": (RecurrentPPO, "MultiInputLstmPolicy", False),
    "ddpg": (DDPG, "MultiInputPolicy", True),
    "sac": (SAC, "MultiInputPolicy", True),
}

REQUIRED_COLUMNS = {"domain", "query", "page_results_text"}


def make_env_fn(rows: pd.DataFrame, continuous: bool, seed: int):
    def _init():
        env = ChunkSelectionEnv(rows, continuous_action=continuous, row_order="random", seed=seed)
        # info_keywords=("n_selected",): without this, Monitor drops the
        # top-level "n_selected" key that the env puts in info (see env.py
        # step()) - that's how ChunkCountCallback can read
        # ep_info["n_selected"].
        return Monitor(env, info_keywords=("n_selected",))
    return _init


def train_one(domain: str, algo_name: str, rows: pd.DataFrame, timesteps: int,
              output_dir: Path, seed: int) -> tuple[Path, dict]:
    algo_cls, policy, continuous = ALGORITHMS[algo_name]
    env = DummyVecEnv([make_env_fn(rows, continuous, seed)])

    out_dir = output_dir / domain / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = dict(
        policy=policy, env=env, verbose=0, seed=seed,
        tensorboard_log=str(out_dir / "tensorboard"),
    )
    if algo_name in ("ppo", "recurrent_ppo"):
        model_kwargs["n_steps"] = 128
        # SB3's default ent_coef=0.0 (no entropy bonus) makes PPO/RecurrentPPO
        # converge too early to an almost-deterministic policy - with the
        # new per-chunk cost reward, this locks the agent into always
        # "select everything" or "select nothing" before it explores the
        # middle ground (observed empirically). A bit of entropy keeps
        # exploration alive longer.
        model_kwargs["ent_coef"] = 0.01
    if algo_name == "ddpg":
        # same reasoning as above: DDPG without exploration noise
        # (action_noise=None is SB3's default) tends to converge too early
        # to the same kind of degenerate solution.
        model_kwargs["action_noise"] = NormalActionNoise(mean=[0.0], sigma=[0.2])

    model = algo_cls(**model_kwargs)

    # sustainability: emissions (CodeCarbon) and training time, per (domain, algorithm)
    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    start = time.time()
    model.learn(total_timesteps=timesteps, tb_log_name=algo_name, callback=ChunkCountCallback())
    train_time_s = time.time() - start
    tracker.stop()
    emissions_data = tracker.final_emissions_data

    model_path = out_dir / "model.zip"
    model.save(model_path)
    env.close()

    sustainability = {
        "domain": domain,
        "algo": algo_name,
        "timesteps": timesteps,
        "train_time_s": train_time_s,
        "emissions_kg_co2": emissions_data.emissions,
        "energy_kwh": emissions_data.energy_consumed,
        "cpu_power_w": emissions_data.cpu_power,
        "gpu_power_w": emissions_data.gpu_power,
    }
    return model_path, sustainability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="parquet with columns domain, query, answer, page_results_text")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--algorithms", default=",".join(ALGO_ORDER))
    parser.add_argument("--domains", default=None, help="comma-separated list; default = all domains in the dataset")
    parser.add_argument("--timesteps", type=int, default=40000, help="training timesteps per (domain, algorithm)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", help="skip (domain, algorithm) whose model.zip already exists")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {a} (options: {sorted(ALGORITHMS)})")

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else sorted(df["domain"].unique())
    output_dir = Path(args.output_dir)

    print(f"Dataset: {args.dataset} ({len(df)} rows, {len(domains)} domains)")
    print(f"Algorithms: {algos}")
    print(f"Timesteps per (domain, algorithm): {args.timesteps}")
    print(f"Output: {output_dir}\n")

    sustainability_rows = []
    for domain in domains:
        rows = df[df["domain"] == domain]
        if rows.empty:
            print(f"[warning] domain '{domain}' not found in dataset, skipping.")
            continue
        for algo_name in algos:
            model_path = output_dir / domain / algo_name / "model.zip"
            if args.skip_existing and model_path.exists():
                print(f"[skip] {domain}/{algo_name} already exists at {model_path}")
                continue
            print(f"== training {algo_name} / domain={domain} ({len(rows)} rows) ==")
            path, sustainability = train_one(domain, algo_name, rows, args.timesteps, output_dir, args.seed)
            sustainability_rows.append(sustainability)
            print(f"   saved to {path} ({sustainability['train_time_s']:.1f}s, "
                  f"{sustainability['emissions_kg_co2'] * 1000:.4f} g CO2eq)")

    if sustainability_rows:
        sustainability_path = output_dir / "sustainability_training.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        sustainability_df = pd.DataFrame(sustainability_rows)
        if sustainability_path.exists():
            sustainability_df = pd.concat(
                [pd.read_csv(sustainability_path), sustainability_df], ignore_index=True
            ).drop_duplicates(subset=["domain", "algo"], keep="last")
        sustainability_df.to_csv(sustainability_path, index=False)
        print(f"\nTraining sustainability data saved to {sustainability_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
