from stable_baselines3 import PPO

from stock_options.environments import TradingEnv, flatten_obs
from stock_options.utils.data import load_data, flatten_obs

import warnings
warnings.filterwarnings("error")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)

    logger.info("Initializing environment for basic test...")
    env = TradingEnv(df, window_size=10, n_months=1, flatten_observations=True) # Use a basic reward

    # --- Initial State Check ---
    observation, info = env.reset()
    logger.info("Environment reset. Performing initial state assertions.")
    # Assertions on the very first state after reset
    expected_flat_obs_shape = 4 + 5 + 2*env.window_size + 4*(env.n_options + env.max_options)
    assert len(flatten_obs(observation)) == expected_flat_obs_shape, f"Initial flattened observation shape mismatch: {len(flatten_obs(observation))} vs {expected_flat_obs_shape}"
    assert info["current_step"] == env.window_size, f"Initial step should be {env.window_size}, got {info.current_step}"
    assert info["portfolio"].cash == env.initial_cash, f"Initial cash should be {env.initial_cash}, got {info["portfolio"].cash}"
    assert info["portfolio"].portfolio_value == env.initial_cash, f"Initial portfolio value should be {env.initial_cash}, got {info["portfolio"].portfolio_value}"
    assert len(info["options_available"]) <= env.n_options, f"Initial reset: Too many available options: {len(info["options_available"])} > {env.n_options}"
    logger.info("Initial state assertions passed.")
    logger.debug(f"Initial observation: {observation}")
    logger.debug(f"Initial info: {info}")


    logger.info("-" * 20)
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Action space sample: {env.action_space.sample()}")
    logger.info("-" * 20)

    # --- Episode Loop Test ---
    done, truncated = False, False
    episode_rewards = []
    
    # Re-reset just to ensure a clean start for the loop if initial inspection modified state (it shouldn't, but good practice)
    observation, info = env.reset() 

    logging.disable(logging.INFO)   # Disable logging for the episode loop to reduce verbosity
    # Create the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cuda")
    # Train the model
    total_timesteps=1_000_000
    model.learn(total_timesteps=total_timesteps)
    # Save the model
    model.save(f"models/ppo_trading_model_{total_timesteps}")
