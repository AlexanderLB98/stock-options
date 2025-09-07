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
    env = TradingEnv(df, window_size=10, n_months=1) # Use a basic reward

    # Check env for SB3 compatibility
    from stable_baselines3.common.env_checker import check_env
    # A chuparla el check_env, me obliga a hacer que info sea un dict en lugar de State
    # Bueno parece que para usar SB3 es necesario, asi que lo hago
    check_env(env, warn=True, skip_render_check=True)

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
    logger.info("Starting episode loop test...")
    
    current_test_step = 0

    model_path = "models/ppo_trading_model_1000000_0"
    model = PPO.load(model_path, device="cuda", env=env)

    while not done and not truncated:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)

        flat_obs = flatten_obs(observation)
        # Verify flattened observation shape in loop
        assert len(flat_obs) == expected_flat_obs_shape, f"Step {info.current_step}: Flattened observation shape mismatch: {len(flat_obs)} vs {expected_flat_obs_shape}"

        # High-level checks for critical external state
        assert info["portfolio"].portfolio_value >= 0, f"Step {info.current_step}: Portfolio value became negative: {info.portfolio.portfolio_value}"
        assert info["portfolio"].cash >= 0, f"Step {info["current_step"]}: Cash became negative: {info["portfolio"].cash}"
        assert len(info["options_available"]) <= env.n_options, f"Step {info["current_step"]}: Too many available options: {len(info["options_available"])} > {env.n_options}"
        
        logger.info(f"Step {info["current_step"]}: Reward={reward:.4f}, Portfolio={info["portfolio"].portfolio_value:.2f}, Cash={info["portfolio"].cash:.2f}, Available Options={len(info["options_available"])}")
        logger.debug(f"Obs: {observation}") # Use debug for full observation, info

        current_test_step += 1
        logger.info("-" * 10) # Shorter separator for steps

    logger.info("Episode loop test finished.")
    logger.info(f"Total steps taken in test: {current_test_step}")
    logger.info(f"Total episode reward: {sum(episode_rewards):.4f}")
    logger.info(f"Final portfolio value: {info["portfolio"].portfolio_value:.2f}")
    logger.info(f"Final portfolio diff: {info["portfolio"].total_value_diff:.2f}")

    # Final assert after loop
    assert done or truncated, "Episode loop terminated unexpectedly."