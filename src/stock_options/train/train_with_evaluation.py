"""
Example usage of the EvaluationCallback during training.

This script demonstrates how to integrate the evaluation callback
into a PPO training loop.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from stock_options.environments import TradingEnv
from stock_options.utils.data import load_random_data
from stock_options.evaluate_models.evaluation_callback import create_evaluation_callback

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def train_with_evaluation():
    """Example training script with evaluation callback."""
    
    # Load training data
    csv_path = "data/stock_data_2025_09_10.csv"
    seed = 123  # Different seed for training
    df = load_random_data(csv_path, seed)
    
    max_options = 10
    n_months = 3
    model_name_suffix = f"short_selling_max_options_{max_options}_n_months_{n_months}"

    # Create training environment
    env = TradingEnv(df, window_size=10, max_options=max_options, n_months=n_months, go_short=True)

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cpu")
    
    # Create callbacks with dynamic path generation
    eval_callback = create_evaluation_callback(
        n_eval_steps=10000,  
        num_eval_runs=20,    
        save_path="results", 
        model_name_suffix=model_name_suffix,  
        window_size=10,      
        n_months=n_months,
        max_options=max_options,
        go_short=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_trading"
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    model.learn(
        total_timesteps=10000000,
        callback=callback,
        tb_log_name="PPO_with_evaluation"
    )
    
    # Save final model
    model.save("models/ppo_trading_final")
    logger.info("Training completed!")


if __name__ == "__main__":
    train_with_evaluation()
