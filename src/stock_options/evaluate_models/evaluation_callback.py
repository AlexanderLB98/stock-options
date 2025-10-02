"""
Custom evaluation callback for Stable Baselines3 training.

This callback runs evaluation episodes during training and saves violin plots
showing the distribution of portfolio performance.

Usage Example:
```python
from stable_baselines3 import PPO
from stock_options.environments import TradingEnv
from stock_options.utils.data import load_random_data
from stock_options.evaluate_models.evaluation_callback import create_evaluation_callback

# Create environment and model
df = load_random_data("data/stock_data_2025_09_10.csv", seed=42)
env = TradingEnv(df, window_size=10, n_months=1)
model = PPO("MultiInputPolicy", env, verbose=1)

# Create evaluation callback with dynamic path
eval_callback = create_evaluation_callback(
    n_eval_steps=10000,  # Evaluate every 10,000 training steps
    num_eval_runs=20,    # Run 20 evaluation episodes each time
    save_path="results", # Base directory (auto-creates model-specific subfolder)
    model_name_suffix="experiment1"  # Optional suffix for unique experiments
)
# This will create: results/evaluations_ppo_experiment1/

# Train with callback
model.learn(total_timesteps=100000, callback=eval_callback)
```

The callback will automatically:
1. Create evaluation episodes using a fixed seed for reproducibility
2. Generate violin plots showing performance distribution
3. Save plots with descriptive names including model name and training step
4. Log evaluation statistics
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from stock_options.environments import TradingEnv, flatten_obs
from stock_options.utils.data import load_random_data

import warnings
warnings.filterwarnings("error")

import logging
logger = logging.getLogger(__name__)


class EvaluationCallback(BaseCallback):
    """
    Custom callback that evaluates the model every n_eval_steps and saves violin plots.
    """

    def __init__(self, 
                 n_eval_steps: int = 10000,
                 num_eval_runs: int = 10,
                 eval_env_kwargs: dict = None,
                 save_path: str = None,
                 model_name_suffix: str = "",
                 flatten_observations: bool = False,
                 verbose: int = 1):
        """
        Initialize the evaluation callback.
        
        Args:
            n_eval_steps: Number of training steps between evaluations
            num_eval_runs: Number of evaluation episodes to run
            eval_env_kwargs: Kwargs for creating the evaluation environment
            save_path: Directory to save the plots (if None, will be auto-generated)
            model_name_suffix: Optional suffix to add to the model name for path generation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.n_eval_steps = n_eval_steps
        self.num_eval_runs = num_eval_runs
        self.eval_env_kwargs = eval_env_kwargs or {}
        self.base_save_path = save_path  # Store the base path, actual path will be set later
        self.model_name_suffix = model_name_suffix
        self.last_eval_step = 0
        self.save_path = None  # Will be set when model is available
        self.flatten_observations = flatten_observations
        
        # Initialize evaluation environment
        self._init_eval_env()

    def _generate_save_path(self):
        """Generate dynamic save path based on model type and optional suffix."""
        if self.base_save_path:
            base_dir = self.base_save_path
        else:
            base_dir = "results"
        
        # Get model name from the algorithm class name
        model_name = self.model.__class__.__name__.lower()
        
        # Add suffix if provided
        if self.model_name_suffix:
            model_name = f"{model_name}_{self.model_name_suffix}"
        
        # Create the full path
        self.save_path = os.path.join(base_dir, f"evaluations_{model_name}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        if self.verbose >= 1:
            logger.info(f"Evaluation results will be saved to: {self.save_path}")

    def _on_training_start(self) -> None:
        """Called when training starts. Set up the save path now that model is available."""
        self._generate_save_path()

    def _init_eval_env(self):
        """Initialize the evaluation environment."""
        csv_path = "data/test_data.csv"
        seed = 42  # Use a fixed seed for reproducible evaluation
        df_test = load_random_data(csv_path, seed)
        
        # Use provided kwargs or defaults
        env_kwargs = {
            'window_size': 10,
            'n_months': 1,
            'mode': "test",
            'flatten_observations': self.flatten_observations,  # Add flattened observations for RecurrentPPO compatibility
            **self.eval_env_kwargs
        }

        self.eval_env = TradingEnv(df_test, **env_kwargs)
        logger.info(f"Evaluation environment initialized with kwargs: {env_kwargs}")

    def _on_step(self) -> bool:
        """
        Called after each training step.
        
        Returns:
            bool: Whether to continue training
        """
        # Check if it's time to evaluate
        if self.num_timesteps - self.last_eval_step >= self.n_eval_steps:
            try:
                self._evaluate_model()
                self.last_eval_step = self.num_timesteps
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")

        return True

    def _evaluate_model(self):
        """Run evaluation episodes and save results."""
        logger.info(f"Starting evaluation at step {self.num_timesteps}")
        
        run_rewards = []
        
        for i in range(self.num_eval_runs):
            if self.verbose >= 1:
                logger.info(f"Evaluation run {i+1}/{self.num_eval_runs}")
            
            # Run a single evaluation episode
            episode_reward = self._run_single_episode()
            run_rewards.append(episode_reward)
        
        # Save violin plot
        self._save_violin_plot(run_rewards)
        
        # Log statistics
        avg_reward = sum(run_rewards) / len(run_rewards)
        min_reward = min(run_rewards)
        max_reward = max(run_rewards)
        
        logger.info(f"Evaluation complete at step {self.num_timesteps}")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Min reward: {min_reward:.4f}")
        logger.info(f"Max reward: {max_reward:.4f}")

    def _run_single_episode(self):
        """Run a single evaluation episode and return the final portfolio difference."""
        done, truncated = False, False
        # Use a different seed for each run to get variability
        seed = np.random.randint(0, 1e6)
        observation, info = self.eval_env.reset(seed=seed)
        while not done and not truncated:
            action, _state = self.model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = self.eval_env.step(action)
        return info["portfolio"].total_value_diff

    def _save_violin_plot(self, rewards):
        """Save a violin plot of the evaluation results."""
        plt.figure(figsize=(10, 6))
        plt.violinplot(rewards, showmeans=True, showmedians=True)
        
        # Get model name from the algorithm class name
        model_name = self.model.__class__.__name__
        
        plt.title(f"Portfolio Performance Evaluation - {model_name} at {self.num_timesteps} steps")
        plt.ylabel("Total Portfolio Value Difference")
        plt.xlabel(f"{model_name} Agent")
        plt.xticks([1], [f"{model_name}"])
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics as text
        avg_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        stats_text = f"Runs: {len(rewards)}\nAvg: {avg_reward:.2f}\nMin: {min_reward:.2f}\nMax: {max_reward:.2f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the plot
        filename = f"{model_name.lower()}_evaluation_step_{self.num_timesteps}.png"
        filepath = os.path.join(self.save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        logger.info(f"Evaluation plot saved to: {filepath}")


# Convenience function to create the callback
def create_evaluation_callback(n_eval_steps: int = 10000, 
                              num_eval_runs: int = 10,
                              save_path: str = None,
                              model_name_suffix: str = "",
                              **env_kwargs) -> EvaluationCallback:
    """
    Create an evaluation callback with the specified parameters.
    
    Args:
        n_eval_steps: Number of training steps between evaluations
        num_eval_runs: Number of evaluation episodes to run
        save_path: Base directory to save the plots (if None, uses "results")
        model_name_suffix: Optional suffix to add to the model name for path generation
        **env_kwargs: Additional kwargs for the evaluation environment
    
    Returns:
        EvaluationCallback: The configured callback
        
    Example:
        # Creates path: results/evaluations_ppo_short_selling/
        callback = create_evaluation_callback(
            n_eval_steps=10000,
            model_name_suffix="short_selling"
        )
        
        # Creates path: my_results/evaluations_dqn_experiment1/
        callback = create_evaluation_callback(
            save_path="my_results",
            model_name_suffix="experiment1"
        )
    """
    return EvaluationCallback(
        n_eval_steps=n_eval_steps,
        num_eval_runs=num_eval_runs,
        eval_env_kwargs=env_kwargs,
        save_path=save_path,
        model_name_suffix=model_name_suffix
    )
