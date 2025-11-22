"""
Example usage of the EvaluationCallback during training.

This script demonstrates how to integrate the evaluation callback
into a PPO training loop.

run tensorboard with:
    tensorboard --logdir recurrent_ppo_trading_tensorboard
"""
import os 
import argparse
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

def make_env(csv_path):
    def _init():
        df = load_random_data(csv_path)
        env = TradingEnv(df, window_size=10, flatten_observations=True)
        return env
    return _init

def train_with_evaluation(name=None, total_timesteps=1000000, n_eval_steps=100000):
    """Example training script with evaluation callback."""
    
    # Load training data
    # csv_path = "data/stock_data_2025_09_10.csv"
    csv_path = "data/train/train_data_all.csv"
    
    # VecEnvs for parallel training:
    num_envs = os.cpu_count()  # Use all available CPU cores
    print(f"Using {num_envs} parallel environments")

    env = SubprocVecEnv([
        make_env(csv_path) 
        for i in range(num_envs)
    ])
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./recurrent_ppo_trading_tensorboard/", device="cuda" if torch.cuda.is_available() else "cpu")

    # Create callbacks with dynamic path generation
    eval_callback = create_evaluation_callback(
        n_eval_steps=n_eval_steps,  
        num_eval_runs=250,    
        save_path="results", 
        name=name,  
        window_size=10,      
        flatten_observations=True,  # Always use flattened for SB3 compatibility
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
    env = TradingEnv(df, window_size=10, flatten_observations=True)

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

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Train PPO model with evaluation callback")
    parser.add_argument("--run_name", type=str, default="default_experiment", 
                        help="Name for the training run (default: default_experiment)")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total training timesteps (default: 1000000)")
    parser.add_argument("--n_eval_steps", type=int, default=100000,
                        help="Number of evaluation steps (default: 100000)")
    parser.add_argument("--test_only", action="store_true",
                        help="Only run model testing, skip training")
    
    args = parser.parse_args()
    
    if not args.test_only:
        print(f"Starting training with run name: {args.run_name}")
        train_with_evaluation(name=args.run_name, total_timesteps=args.total_timesteps, n_eval_steps=args.n_eval_steps)

    print("Running model test...")
    test_model()

if __name__ == "__main__":
    main()
