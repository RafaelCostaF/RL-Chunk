import os
import pandas as pd
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC
from sb3_contrib.ppo_recurrent import RecurrentPPO
from RL_environment import FixedChunkEnvGranularReward
import torch.multiprocessing as mp
import torch
import time
from codecarbon import EmissionsTracker

from multiprocessing import Semaphore

from stable_baselines3.common.callbacks import BaseCallback

class EpisodicLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            reward = self.locals["infos"][0]["episode"]["r"]
            length = self.locals["infos"][0]["episode"]["l"]
            print(f"🎯 Episode finished: reward={reward:.2f}, length={length}")
        return True


all_algorithms = {
    "ppo": (PPO, "MultiInputPolicy", False),
    "recurrent_ppo": (RecurrentPPO, "MultiInputLstmPolicy", False),
    "ddpg": (DDPG, "MultiInputPolicy", True),
    "sac": (SAC, "MultiInputPolicy", True),
}
# def train_single_model(gpu_id, row_idx, sampled_file, base_dir):
#     import torch
#     import pandas as pd
#     from stable_baselines3 import PPO, DDPG, SAC
#     from sb3_contrib.ppo_recurrent import RecurrentPPO
#     from stable_baselines3.common.monitor import Monitor
#     from FixedChunkEnv import FixedChunkEnv
#     from codecarbon import EmissionsTracker
#     import os
#     import time

#     device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#     torch.cuda.set_device(device)
def train_single_model(gpu_id, row_idx, sampled_file, base_dir):
    import torch
    import pandas as pd
    from stable_baselines3 import PPO, DDPG, SAC
    from sb3_contrib.ppo_recurrent import RecurrentPPO
    from stable_baselines3.common.monitor import Monitor
    from FixedChunkEnv import FixedChunkEnv
    from codecarbon import EmissionsTracker
    import os
    import time

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.5, device)  # ✅ limit GPU to 50%
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))  # ✅ limit CPU threads

    # ✅ Read only one row from file
    df = pd.read_parquet(sampled_file, engine="pyarrow")
    row = df.iloc[row_idx]

    all_algorithms = {
        "ppo": (PPO, "MultiInputPolicy", False),
        "recurrent_ppo": (RecurrentPPO, "MultiInputLstmPolicy", False),
        "ddpg": (DDPG, "MultiInputPolicy", True),
        "sac": (SAC, "MultiInputPolicy", True),
    }

    results = []  # Collect metrics for this row

    for algo_name, (AlgoClass, policy, use_continuous_action) in all_algorithms.items():
        model_dir = os.path.join(base_dir, f"row_{row_idx}", algo_name)
        model_path = os.path.join(model_dir, f"{algo_name}_final_model")

        if os.path.exists(model_path + ".zip"):
            print(f"✅ Skipping existing model at {model_path}")
            continue

        os.makedirs(model_dir, exist_ok=True)
        tensorboard_log_dir = os.path.join(model_dir, "tensorboard")

        env = Monitor(FixedChunkEnv(row, continuous_action=use_continuous_action))
        model_kwargs = {
            "policy": policy,
            "env": env,
            "verbose": 0,
            "tensorboard_log": tensorboard_log_dir,
            "device": device
        }

        if algo_name in ["ppo", "recurrent_ppo"]:
            model_kwargs["n_steps"] = 128

        # ✅ Track emissions and time
        tracker = EmissionsTracker(
            output_dir=model_dir,
            log_level="error",
            save_to_file=True,  # ✅ This enables file saving
            tracking_mode="process"  # ✅ Optional but safer with multiprocessing
        )

        tracker.start()
        start_time = time.time()

        model = AlgoClass(**model_kwargs)
        model.learn(total_timesteps=50000, tb_log_name=f"{algo_name}_row_{row_idx}", callback=EpisodicLogger())

        duration_minutes = (time.time() - start_time) / 60
        emissions_data = tracker.stop()

        model.save(model_path)


sema = Semaphore(2)  # Global semaphore

# def safe_train(gpu_id, row_idx, sampled_file, base_dir):
#     with sema:
#         train_single_model(gpu_id, row_idx, sampled_file, base_dir)

def safe_train(gpu_id, row_idx, sampled_file, base_dir, max_retries=3):
    with sema:
        for attempt in range(max_retries):
            try:
                print(f"🚀 Starting training for row {row_idx}, attempt {attempt + 1}")
                train_single_model(gpu_id, row_idx, sampled_file, base_dir)
                print(f"✅ Success for row {row_idx}")
                break
            except Exception as e:
                print(f"⚠️ Error on row {row_idx} (attempt {attempt + 1}): {e}")
                time.sleep(5)
        else:
            print(f"❌ Failed to train model for row {row_idx} after {max_retries} attempts")


def train_rows_on_gpu(gpu_id, index_chunks, sampled_file, base_dir):
    import torch.multiprocessing as mp
    procs = []
    for idx in index_chunks[gpu_id]:
        p = mp.Process(target=safe_train, args=(gpu_id, idx, sampled_file, base_dir))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()



def split_indices(n_rows, n_chunks):
    # Evenly split indices into chunks
    chunk_size = (n_rows + n_chunks - 1) // n_chunks
    return [list(range(i * chunk_size, min((i + 1) * chunk_size, n_rows))) for i in range(n_chunks)]

if __name__ == "__main__":
    sampled_file = "sampled_50_per_domain.parquet"
    base_dir = "./train_per_row_final_only_granular_reward"

    df = pd.read_parquet(sampled_file)
    total_rows = len(df)
    n_gpus = torch.cuda.device_count()

    print(f"🧠 Total rows: {total_rows}")
    print(f"🎮 Available GPUs: {n_gpus}")

    index_chunks = split_indices(total_rows, n_gpus)
    mp.set_start_method("spawn", force=True)
    mp.spawn(train_rows_on_gpu, args=(index_chunks, sampled_file, base_dir), nprocs=n_gpus)
