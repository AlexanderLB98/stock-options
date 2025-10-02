"""
Example usage of the EvaluationCallback during training.

This script demonstrates how to integrate the evaluation callback
into a PPO training loop.
"""
import os 
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from stock_options.environments import TradingEnv
from stock_options.utils.data import load_random_data
from stock_options.evaluate_models.evaluation_callback import create_evaluation_callback

import logging

import torch
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_env(seed, csv_path, max_options, n_months, go_short):
    def _init():
        df = load_random_data(csv_path, seed)
        env = TradingEnv(df, window_size=10, max_options=max_options, n_months=n_months, go_short=go_short, flatten_observations=True)
        return env
    return _init

def train_with_evaluation(max_options = 4, n_months = 2, name=None):
    """Example training script with evaluation callback."""
    
    # Load training data
    # csv_path = "data/stock_data_2025_09_10.csv"
    csv_path = "data/train/train_data_all.csv"
    seed = 123  # Different seed for training
    df = load_random_data(csv_path, seed)
    
    max_options = 4
    n_months = 2
    model_name_suffix = f"short_selling_max_options_{max_options}_n_months_{n_months}"
    if name:
        model_name_suffix = f"{name}_{model_name_suffix}"

    # Create training environment
    # env = TradingEnv(df, window_size=10, max_options=max_options, n_months=n_months, go_short=True)

    # VecEnvs for parallel training:
    # num_envs = 10  # Number of parallel environments
    num_envs = os.cpu_count()  # Use all available CPU cores
    print(f"Using {num_envs} parallel environments")

    # env = DummyVecEnv([
    #     make_env(seed + i, csv_path, max_options, n_months, True) 
    #     for i in range(num_envs)
    # ])
    env = SubprocVecEnv([
        make_env(seed + i, csv_path, max_options, n_months, True) 
        for i in range(num_envs)
    ])
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./recurrent_ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")

    # Create callbacks with dynamic path generation
    eval_callback = create_evaluation_callback(
        n_eval_steps=10000,  
        num_eval_runs=20,    
        save_path="results", 
        model_name_suffix=model_name_suffix,  
        window_size=10,      
        n_months=n_months,
        max_options=max_options,
        go_short=True,
        flatten_observations=True,
        csv_path="data/test/test_data_all.csv"  
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_trading"
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    model.learn(
        total_timesteps=1000000,
        callback=callback,
        tb_log_name="recurrent_PPO_with_evaluation"
    )
    
    # Save final model
    model.save("models/recurrent_ppo_trading_final")
    logger.info("Training completed!")


if __name__ == "__main__":
    train_with_evaluation(max_options = 4, n_months = 2, name="multiProcessing_training")
