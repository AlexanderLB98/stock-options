from typing import Optional
import gymnasium as gym
from gymnasium import spaces

import polars as pl
import numpy as np
import datetime
import glob
from pathlib import Path    

from collections import Counter
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium.wrappers import FlattenObservation

from stock_options.options import Option
from stock_options.stateManagement import initialize_state, State
from stock_options.utils.portfolio import Portfolio, TargetPortfolio
from stock_options.utils.history import History
from stock_options.utils.optionsPortfolio import OptionsPortfolio

from stock_options.blackScholes import gen_option_for_date
# from src.options import define_action_space, Option
from stock_options.options import define_action_space

# from datetime import datetime, date

import tempfile, os
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



class TradingEnv(gym.Env):
    """
    A trading environment for OpenAI gymnasium to trade stocks with options.
    
    This environment follows the standard `gymnasium.Env` interface. At each step, the agent
    receives an observation of the market and its portfolio and must take an action
    related to buying or selling options. The goal is to maximize the portfolio's value TBD.

    Attributes:
        df (pl.DataFrame): The DataFrame containing all historical market data.
        action_space (gym.spaces.Space): The space of actions the agent can take.
        observation_space (gym.spaces.Dict): The structure of the observation space.
        state: A state object (e.g., a dataclass) containing the full internal state of the environment.
        portfolio (OptionsPortfolio): The object that manages cash and owned options.
    
        self.n_options : int : The number of available options at each step. Depends on n_strikes and n_months.

    inputs:
    - df : polars.DataFrame : A dataframe containing the historical price data and features. It must contain a 'close' column and feature columns (with 'feature' in their name).
    - reward_function : function : A function that takes the historical info (History object) and returns a reward (float). Default is basic_reward_function.
    - portfolio_initial_value : float : The initial value of the portfolio. Default is 1000.
    - max_options : int : The maximum number of options that can be owned at once. Default is 2.
    - n_strikes : int : The number of strikes above and below the spot price. Default is 2.
    - n_months : int : The number of months to consider for options. Default is 1.
    - strike_step_pct : float : The step percentage for strikes. Default is 0.1 (10%).

    On every step, the environment:
        - Performs the action (buy/sell/hold options)
        - update the state (current step, date, price, available options)
        - update porfolio value
        - get reward
        - log/history
        - get observation
        - get info
    """

    def __init__(self,
                df : pl.DataFrame,
                reward_function = None,
                portfolio_initial_value = 1000,
                window_size = 0,
                max_options = 2,
                n_strikes = 2,
                n_months = 1,
                strike_step_pct = 0.1
                ):
        self.reward_function = reward_function
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.df = df
        self.window_size = window_size

        # Features
        self._features = [col for col in df.columns if "feature" in col]
        self._nb_features = len(self._features)

        # Options
        self._initialize_options_parameters(max_options, n_strikes, n_months, strike_step_pct)
        self.portfolio = OptionsPortfolio(initial_cash=self.portfolio_initial_value, max_options=self.max_options)

        # self.action_space = spaces.Discrete(len(positions))
        self.action_space = define_action_space(self)
        # self._initialize_state()
        self._initialize_observation_space()
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode. Initialize the state and generate options for the first date.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        
        self.state = self._initialize_state()
        self.state.current_date = self.df[self.state.current_step, "date"]
        self.state.current_price = self.df[self.state.current_step, "close"]
        
        # ---Checks added for debugging---
        assert self.state.cash == self.portfolio_initial_value, "Initial cash mismatch after reset."
        assert self.state.portfolio_value == self.portfolio_initial_value, "Initial portfolio value mismatch after reset."
        logger.debug(f"Reset: State initialized. Cash: {self.state.cash}, Portfolio Value: {self.state.portfolio_value}")
        # --- End of check ---

        # Get list of available options for the current date
        self.state.options_available = self.update_options()
        observation = self._get_obs()
        logger.info(f"Initial observation: {observation}")
        info = self._get_info()
        logger.info(f"Initial info: {info}")
        return observation, info

    def render(self):
        pass

    def step(self, action):
        """Take a step in the environment.

        steps:
        - perform the action (buy/sell/hold options)
        - get reward
        - update the state (current step, date, price, available options)
        - update porfolio value
        - log/history
        - get observation
        - get info

        Args:
            action: The action to take (e.g., buy/sell/hold for each option)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        terminated = False
        truncated = False
        
        # --- Check action validity ---
        if not self.action_space.contains(action):
            logger.error(f"Action {action} is not within the defined action space: {self.action_space}")
            raise ValueError(f"Action {action} is invalid.")
        logger.debug(f"Step {self.state.current_step}: Action received: {action}")
        # --- End of check ---

        self.perform_action(action)

        self.state.current_step += 1
        logger.info(f"Current step: {self.state.current_step}")

        self.state.current_date = self.df[self.state.current_step, "date"]
        self.state.current_price = self.df[self.state.current_step, "close"]

        self.state.portfolio_value = self.get_portfolio_value()
        # Get list of available options for the current date
        self.state.options_available = self.update_options()
        
        observation = self._get_obs()
        info = self._get_info()

        reward = self.get_reward()

        if self.state.current_step >= len(self.df) - 1:
            truncated = True
        
        return observation, reward, terminated, truncated, info

    def perform_action(self, actions):
        """ Perform the given action in the environment. Buy/Sell/Do nothing with options.
        ACtion could be a discrete list with len of available options, 0 to do nothing, 1 to buy n option and -1 to sell n option.
        """
        
        # Loop over all the options to perform the action
        for i, action in enumerate(actions):
            
            logger.info(f"action[{i}] = {action}")
            if i < self.n_options and i < len(self.state.options_available):
                # Also checks if there are available options
                logger.info("Action on available options: buy or do nothing")
            
                option = self.state.options_available[i]
                if action == 1:
                    logger.info(f"Buying option {option}")
                    # self.portfolio.buy_option(option)
                elif action == 0:
                    logger.info("Doing nothing")
            elif i >= self.n_options:
                logger.info("Action on owned options: sell or do nothing")
                n_owned = i - self.n_options
                logger.info(f"n_owned = {n_owned}")
                if action == 1 and n_owned < len(self.state.owned_options):
                    logger.info(f"Selling owned option {self.state.owned_options[n_owned]}")
                    # self.portfolio.sell_option(self.state.owned_options[n_owned])

    
    def get_reward(self):
        """ Calculate the reward for the current step
        returns: float : The reward for the current step
        """
        reward = 0.0
        # --- check Reward ---
        assert isinstance(reward, (float)), f"Reward must be a float, got {type(reward)}"
        logger.info(f"Step {self.state.current_step}: Reward={reward}")
        # --- End of check ---
        
        return reward

    def get_portfolio_value(self):
        """ Calculates and return the current portfolio value. """
        # return self.state.portfolio_value # PLACEHOLDER
        return self.portfolio.get_current_total_value(self.state.current_price, self.state.current_date)

    def update_options(self) -> list[Option]:
        """ Update the available options based on the current date and price. 
        
        1. Generate options for the current date (calls and puts)
        2. Add them to the existing options list
        3. Remove expired options
        4. Keep only unique options (by type, strike, expiry date)
        5. Sort options by date generated (newest first)
        6. Select the top N options to be available. Not more than self.max_options can be in that list.

        Currently, this list is not always full.

        Returns:
            list[Option]: A list of available options for the current date.
        """

        # Option implementation
        options_call = gen_option_for_date(
                                        current_date=self.state.current_date,
                                        option_type='call',
                                        spot_price=self.state.current_price,
                                        num_strikes=self.n_strikes,
                                        strike_step_pct=self.strike_step_pct,  
                                        n_months=self.n_months
                                    )
        options_put = gen_option_for_date(
                                        current_date=self.state.current_date,
                                        option_type='put',
                                        spot_price=self.state.current_price,
                                        num_strikes=self.n_strikes,
                                        strike_step_pct=self.strike_step_pct,  
                                        n_months=self.n_months
                                    )

        self.options = self.options + options_call + options_put
        self.options = sorted(self.options, key=lambda opt: opt.date_generated, reverse=True)
        self.options = [opt for opt in self.options if opt.expiry_date >= self.state.current_date]

        seen = set()
        unique_options = []
        for opt in self.options:
            key = (opt.option_type, opt.strike, opt.expiry_date)
            if key not in seen:
                unique_options.append(opt)
                seen.add(key)

        # self.state.options_available = unique_options[:self.n_options]
        options_available = unique_options[:self.n_options]
        logger.info(f"Current step: {self.state.current_step}")
        logger.info(f"Current date: {self.state.current_date}")
        logger.info(f"Number of generated options: {len(self.options)}")
        logger.info(f"Numero de opciones disponibles unicas: {len(self.state.options_available)}")
        assert len(options_available) <= self.n_options, "More available options than n_options limit!"
        return options_available
    
    def update_history(self):
        pass

    def _initialize_options_parameters(self, max_options, n_strikes, n_months, strike_step_pct):
        """ 
        Initialize the parameters related to options trading.
        - max_options : int : max options you can own at once
        - n_strikes : int : number of strikes above and below the spot price
        - n_months : int : number of months to consider for options
        - strike_step_pct : float : step percentage for strikes
        - n_options : int : Number of available options. 2 for call and put options
        """

        self.options = []
        self.max_options = max_options  # max options you can own at once
        self.owned_options = [] 
        self.n_owned_options = len(self.owned_options)  # number of owned options
        self.n_strikes = n_strikes  # number of strikes above and below the spot price
        self.n_months = n_months
        self.strike_step_pct = strike_step_pct  # step percentage for strikes
        self.n_options = (self.n_strikes * 2 + 1) * self.n_months * 2  # Number of available options. 2 for call and put options
 
    def _initialize_state(self) -> State:
        """ Initialize the state of the environment using the dataclass.
         The state includes:
         - current_step: int : The current step in the episode. Its defined from the window_size, so the first step is window_size.
         - done: bool : Whether the episode is done.
         - truncated: bool : Whether the episode is truncated.
         - current_date: pd.Timestamp : The current date in the episode.
         - current_price: float : The current price of the stock.
         - cash: float : The current cash available.
         - portfolio_value: float : The current total value of the portfolio.
         - owned_options: list : The list of currently owned options.
         - options_available: list : The list of currently available options.
         - history: History : The history of the episode.

        returns:
            state: The initialized state object.
        """
        state = initialize_state(current_step = self.window_size, initial_cash = self.portfolio_initial_value)
        assert state.current_step == self.window_size, "Initial step mismatch."
        assert state.cash == self.portfolio_initial_value, "Initial cash mismatch."
        logger.debug(f"State initialized: current_step={state.current_step}, cash={state.cash}")
        return state

    def _get_obs(self):
        """"
        The observation is a subset from the state with the information from:
        - Current cash
        - Current portfolio value
        - features from current date
        - N last closes (N = window_size)
        - options available: for each option (self.n_options), 4 features: type (call/put), strike, premium, days_to_expiry
        - owned options: for each option, 4 features: type (call/put), strike, premium, days_to_expiry

        Shape expected: (75,) with this configuration.
            2 (Cash, portfolio_value)
            5 (open, high, low, close, volume) 
            10 (last_closes) + 10 (last_volumes)         (window_size=10) 
            40 (10 options * 4 features)                 (self.n_options=10)
            8 (2 owned options * 4 features)             (self.max_options=2)
                = 75

            Dynamically, the shape is:
                2 + 5 + 2*window_size  + (self.n_options * 4) + (self.max_options * 4)   
                2 + 5 + 2*window_size  + 4*(self.n_options  + self.max_options)   
                
            Returns: dict: The observation dictionary.
        """""
        N = self.window_size  # window size for last closes
        M = self.n_options # available options
        K = self.max_options # max possible owned options
   
        # Expected shape: 5 + 2*window_size  + 4*(self.n_options  + self.max_options)   
        expected_shape = 2 + 5 + 2*N + 4*(M + K)
        logger.info(f"Expected observation shape: {expected_shape}")

        # --- 0. Current state for cash and portfolio value ---
        portfolio = {
            "cash": float(self.state.cash),
            "portfolio_value": float(self.state.portfolio_value)
        }

        # --- 1. Current state for today ---
        row = self.df[self.state.current_step]
        today = {
            "open": float(row["open"][0]),
            "close": float(row["close"][0]),
            "low": float(row["low"][0]),
            "high": float(row["high"][0]),
            "volume": float(row["volume"][0])
        }

        # --- 2. Last N closes ---
        start = max(0, self.state.current_step - N + 1)
        
        closes = self.df[start:self.state.current_step + 1, "close"].to_numpy().flatten()
        closes = np.pad(closes, (N - len(closes), 0), 'constant', constant_values=0)
        assert closes.shape == (N,), f"last_closes shape mismatch: {closes.shape} vs {(N,)}"
        
        # Last N volumes
        volumes = self.df[start:self.state.current_step + 1, "volume"].to_numpy().flatten()
        volumes = np.pad(closes, (N - len(closes), 0), 'constant', constant_values=0)
        assert volumes.shape == (N,), f"last_volumes shape mismatch: {volumes.shape} vs {(N,)}"
        
        # --- 3. Available options ---
        options_list = self.state.options_available  # Now a list of Option objects
        type_map = {"call": 0, "put": 1}
        available_options = []
        for i in range(M):
            if i < len(options_list):
                opt = options_list[i]
                available_options.append([
                    type_map.get(opt.option_type, -1),
                    opt.strike,
                    opt.premium,
                    opt.days_to_expire
                ])
            else:
                available_options.append([0, 0.0, 0.0, 0.0])
        available_options = np.array(available_options, dtype=np.float32)
        assert available_options.shape == (M, 4), f"available_options shape mismatch: {available_options.shape} vs {(M, 4)}"

        # --- 4. Owned options ---
        owned_options = []
        for i in range(K):
            if i < len(self.state.owned_options):
                opt = self.state.owned_options[i]
                owned_options.append([
                    type_map.get(opt.type, -1),
                    opt.strike,
                    opt.premium,
                    opt.days_to_expiry
                ])
            else:
                owned_options.append([0, 0.0, 0.0, 0.0])
        owned_options = np.array(owned_options, dtype=np.float32)
        assert owned_options.shape == (K, 4), f"owned_options shape mismatch: {owned_options.shape} vs {(K, 4)}"

        obs = {
            "portfolio": portfolio,
            "today": today,
            "last_closes": closes,
            "last_volumes": volumes,
            "available_options": available_options,
            "owned_options": owned_options
        }

        # Make sure the observation matches the expected shape
        assert len(flatten_obs(obs)) == expected_shape, f"Observation shape mismatch: got {len(flatten_obs(obs))}, expected {expected_shape}"
        
        return obs
    
    def _initialize_observation_space(self):
        """ Initialize the observation space based on features and options.
         The observation space includes:
         - The features (self._nb_features): open, high, low, close, volume
         - Include windows of past closes.
         - For each option, 4 features: type (call/put), strike, premium, days_to_expiry

         For example:
         - Features: 5 (open, high, low, close, volume)
         - last_closes: window_size (e.g., 10)
         - last_volumes: window_size (e.g., 10)
         - Options: n_options * 4 (e.g., 10 options * 4 features = 40)
         - Owned options: max_options * 4 (e.g., 2 options * 4 features = 8)
         - Total: 5 + 10 + 10 + 40 + 8 = 73
         """
        N = self.window_size  # window size for last closes
        M = self.n_options # available options
        K = self.max_options # max possible owned options

        logger.info(f"Observation space initialized with N={N}, M={M} (n_options), K={K} (max_options)")
        
        self.observation_space = Dict({
            "today": Dict({
                "open": Box(-np.inf, np.inf, shape=()),
                "close": Box(-np.inf, np.inf, shape=()),
                "low": Box(-np.inf, np.inf, shape=()),
                "high": Box(-np.inf, np.inf, shape=()),
                "volume": Box(-np.inf, np.inf, shape=()),
            }),
            "last_closes": Box(-np.inf, np.inf, shape=(N,)),
            "last_volumes": Box(-np.inf, np.inf, shape=(N,)),
            "available_options": Box(-np.inf, np.inf, shape=(M, 4)),
            "owned_options": Box(-np.inf, np.inf, shape=(K, 4)),
        })
        logger.debug(f"Full observation space definition: {self.observation_space}")

    def _get_info(self):
        return self.state
  

def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(len(df), "rows loaded from", csv_path)
    df = df[-1000:]

    # Create the features using Polars expressions
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("feature_close"),
        (pl.col("open") / pl.col("close")).alias("feature_open"),
        (pl.col("high") / pl.col("close")).alias("feature_high"),
        (pl.col("low") / pl.col("close")).alias("feature_low"),
    ])
    return df


def flatten_obs(obs: dict)-> np.ndarray:
    """ Flatten the observation dictionary into a 1D numpy array. """
    flat = []
    for v in obs.values():
        if isinstance(v, dict):
            # Flatten nested dicts (e.g., "today")
            flat.extend(list(v.values()))
        elif isinstance(v, np.ndarray):
            flat.extend(v.flatten())
        else:
            flat.append(v)
    return np.array(flat, dtype=np.float32)


if __name__ == "__main__":
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)

    logger.info("Initializing environment for basic test...")
    env = TradingEnv(df, window_size=10, n_months=1) # Use a basic reward

    # --- Initial State Check ---
    observation, info = env.reset()
    logger.info("Environment reset. Performing initial state assertions.")
    # Assertions on the very first state after reset
    expected_flat_obs_shape = 2 + 5 + 2*env.window_size + 4*(env.n_options + env.max_options)
    assert len(flatten_obs(observation)) == expected_flat_obs_shape, f"Initial flattened observation shape mismatch: {len(flatten_obs(observation))} vs {expected_flat_obs_shape}"
    assert info.current_step == env.window_size, f"Initial step should be {env.window_size}, got {info.current_step}"
    assert info.cash == env.portfolio_initial_value, f"Initial cash should be {env.portfolio_initial_value}, got {info.cash}"
    assert info.portfolio_value == env.portfolio_initial_value, f"Initial portfolio value should be {env.portfolio_initial_value}, got {info.portfolio_value}"
    assert len(info.options_available) <= env.n_options, f"Initial reset: Too many available options: {len(info.options_available)} > {env.n_options}"
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

    while not done and not truncated:
        action = env.action_space.sample() 
        observation, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)

        flat_obs = flatten_obs(observation)
        # Verify flattened observation shape in loop
        assert len(flat_obs) == expected_flat_obs_shape, f"Step {info.current_step}: Flattened observation shape mismatch: {len(flat_obs)} vs {expected_flat_obs_shape}"

        # High-level checks for critical external state
        assert info.portfolio_value >= 0, f"Step {info.current_step}: Portfolio value became negative: {info.portfolio_value}"
        assert info.cash >= 0, f"Step {info.current_step}: Cash became negative: {info.cash}"
        assert len(info.options_available) <= env.n_options, f"Step {info.current_step}: Too many available options: {len(info.options_available)} > {env.n_options}"
        
        logger.info(f"Step {info.current_step}: Reward={reward:.4f}, Portfolio={info.portfolio_value:.2f}, Cash={info.cash:.2f}, Available Options={len(info.options_available)}")
        logger.debug(f"Obs: {observation}") # Use debug for full observation, info

        current_test_step += 1
        logger.info("-" * 10) # Shorter separator for steps

    logger.info("Episode loop test finished.")
    logger.info(f"Total steps taken in test: {current_test_step}")
    logger.info(f"Total episode reward: {sum(episode_rewards):.4f}")
    logger.info(f"Final portfolio value: {info.portfolio_value:.2f}")

    # Final assert after loop
    assert done or truncated, "Episode loop terminated unexpectedly."