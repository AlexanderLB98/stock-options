"""
Example usage of the EvaluationCallback during training.

This script demonstrates how to integrate the evaluation callback
into a PPO training loop.

run tensorboard with:
    tensorboard --logdir recurrent_ppo_trading_tensorboard
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

def make_env(csv_path, max_options, n_months, go_short):
    def _init():
        df = load_random_data(csv_path)
        env = TradingEnv(df, window_size=10, max_options=max_options, n_months=n_months, go_short=go_short, flatten_observations=True)
        return env
    return _init

def train_with_evaluation(max_options = 4, n_months = 2, name=None, total_timesteps=1000000):
    """Example training script with evaluation callback."""
    
    # Load training data
    # csv_path = "data/stock_data_2025_09_10.csv"
    csv_path = "data/train/train_data_all.csv"
    
    model_name_suffix = f"short_selling_max_options_{max_options}_n_months_{n_months}"
    if name:
        model_name_suffix = f"{name}_{model_name_suffix}"

    # VecEnvs for parallel training:
    num_envs = os.cpu_count()  # Use all available CPU cores
    print(f"Using {num_envs} parallel environments")

    env = SubprocVecEnv([
        make_env(csv_path, max_options, n_months, True) 
        for i in range(num_envs)
    ])
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./recurrent_ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")

    # Create callbacks with dynamic path generation
    eval_callback = create_evaluation_callback(
        n_eval_steps=250000,  
        num_eval_runs=250,    
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
        save_freq=250000,
        save_path="./models/checkpoints/",
        name_prefix="recurrent_ppo_trading"
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="recurrent_PPO_with_evaluation"
    )
    
    # Save final model
    model.save("models/recurrent_ppo_trading_final")
    logger.info("Training completed!")

def test_model():
    # Load test data
    csv_path = "data/test/test_data_all.csv"
    df = load_random_data(csv_path)

    # Create environment (flattened obs for compatibility)
    env = TradingEnv(df, window_size=10, max_options=4, n_months=2, go_short=True, flatten_observations=True)

    # Load trained model
    model = RecurrentPPO.load("models/recurrent_ppo_trading_final_0")

    obs, info = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Step reward: {reward}, Cash: {info['portfolio'].cash}, Portfolio Value: {info['portfolio'].portfolio_value}")
        # Action is always the same
        
    print("Final portfolio value:", info["portfolio"].portfolio_value)
    print("Total portfolio diff:", info["portfolio"].total_value_diff)

if __name__ == "__main__":
    train_with_evaluation(max_options = 2, n_months = 2, name="2025-10-25-experiment")
    test_model()
