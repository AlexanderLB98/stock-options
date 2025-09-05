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

from gym_trading_env.stateManagement import initialize_state
from gym_trading_env.utils.portfolio import Portfolio, TargetPortfolio
from gym_trading_env.utils.history import History
from gym_trading_env.utils.optionsPortfolio import OptionsPortfolio

from gym_trading_env.blackScholes import gen_option_for_date
# from src.options import define_action_space, Option
from gym_trading_env.options import define_action_space, Option

# from datetime import datetime, date

import tempfile, os
import warnings
warnings.filterwarnings("error")

def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


class TradingEnv(gym.Env):
    """
    A trading environment for OpenAI gymnasium to trade stocks with options.

    inputs:
    - df : pandas.DataFrame : A dataframe containing the historical price data and features. It must contain a 'close' column and feature columns (with 'feature' in their name).
    - reward_function : function : A function that takes the historical info (History object) and returns a reward (float). Default is basic_reward_function.
    - portfolio_initial_value : float : The initial value of the portfolio. Default is 1000.
    - max_options : int : The maximum number of options that can be owned at once. Default is 2.
    - n_strikes : int : The number of strikes above and below the spot price. Default is 2.
    - n_months : int : The number of months to consider for options. Default is 1.
    - strike_step_pct : float : The step percentage for strikes. Default is 0.1 (10%).
    
    """
    def __init__(self,
                df : pl.DataFrame,
                reward_function = basic_reward_function,
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

        # self.action_space = spaces.Discrete(len(positions))
        self.action_space = define_action_space(self)
        # self._initialize_state()
        self._initialize_observation_space()
        self.reset()

    def _get_info(self):
        return self.state
  
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        
        self._initialize_state()
        self.state.current_date = self.df[0, "date"]
        self.state.current_price = self.df[0, "close"]
        # Get list of available options for the current date
        self.update_options()
        observation = self._get_obs()
        info = self._get_info()
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
        self.perform_action(action)

        self.state.current_step += 1
        self.state.current_date = self.df[self.state.current_step, "date"]
        self.state.current_price = self.df[self.state.current_step, "close"]
        # Get list of available options for the current date
        self.update_options()
        
        observation = self._get_obs()
        info = self._get_info()





        reward = 0    
        terminated = False
        truncated = False
        if self.state.current_step >= len(self.df) - 1:
            truncated = True
        
        return obs, reward, terminated, truncated, info

    def perform_action(self, action):
        """ Perform the given action in the environment. Buy/Sell/Do nothing with options.
        ACtion could be a discrete list with len of available options, 0 to do nothing, 1 to buy n option and -1 to sell n option.
        """
        pass

    def get_reward(self):
        """ Calculate the reward for the current step"""
        pass

    def get_portfolio_value(self):
        """ Calculates and return the current portfolio value. """
        pass

    def update_options(self):
        """ Update the available options based on the current date and price. """
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

        call_df = pl.DataFrame(options_call)
        put_df = pl.DataFrame(options_put)
        # options_df = pl.DataFrame(options)
        options = pl.concat([self.options, call_df, put_df], how="vertical")
        options = options.sort("current_date", descending=True)  # Sort by current_date descending
        # Delete expired options
        options = options.filter(pl.col("expiry_date") >= self.state.current_date)
        options = options.slice(0, self.n_options)  # Limit to n_options
        self.state.options_available = options.unique(subset=["type", "strike", "expiry_date"], maintain_order=True)


    def update_history(self):
        pass

    def _initialize_options_parameters(self, max_options, n_strikes, n_months, strike_step_pct):
        # self.options = pl.DataFrame()
        self.options = pl.DataFrame()
        self.max_options = max_options  # max options you can own at once
        self.owned_options = [] 
        self.n_owned_options = len(self.owned_options)  # number of owned options
        self.n_strikes = n_strikes  # number of strikes above and below the spot price
        self.n_months = n_months
        self.strike_step_pct = strike_step_pct  # step percentage for strikes
        self.n_options = (self.n_strikes * 2 + 1) * self.n_months * 2  # 2 for call and put options
 
    def _initialize_state(self):
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
        """
        self.state = initialize_state(current_step = self.window_size, initial_cash = self.portfolio_initial_value)


    def _get_obs(self):
        """"
        The observation is a subset from the state with the information from:
        - features from current date
        - N last closes (N = window_size)
        - options available: for each option, 4 features: type (call/put), strike, premium, days_to_expiry
        - owned options: for each option, 4 features: type (call/put), strike, premium, days_to_expiry
        """""
        N = self.window_size  # window size for last closes
        M = self.n_options # available options
        K = self.max_options # max possible owned options
   
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

        # Last N volumes
        volumes = self.df[start:self.state.current_step + 1, "volume"].to_numpy().flatten()
        volumes = np.pad(closes, (N - len(closes), 0), 'constant', constant_values=0)

        # --- 3. Available options ---
        options_df = self.state.options_available
        type_map = {"call": 0, "put": 1}
        available_options = []
        options_dicts = options_df.to_dicts()
        for i in range(M):
            if i < len(options_dicts):
                opt = options_dicts[i]
                available_options.append([
                    type_map.get(opt["type"], -1),
                    opt["strike"],
                    opt["premium"],
                    opt["days_to_expiry"]
                ])
            else:
                available_options.append([0, 0.0, 0.0, 0.0])
        available_options = np.array(available_options, dtype=np.float32)

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

        obs = {
            "today": today,
            "last_closes": closes,
            "last_volumes": volumes,
            "available_options": available_options,
            "owned_options": owned_options
        }
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
        # self.observation_space = spaces.Box(
        #     -np.inf,
        #     np.inf,
        #     shape = [self._nb_features + self.n_options * 4, ]  # 4 features for each option (type, strike, premium, days_to_expiry)
        # )

    

def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pl.read_csv(csv_path, try_parse_dates=True)
    print(len(df), "rows loaded from", csv_path)
    df = df[-1000:]

    # Create the features using Polars expressions
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("feature_close"),
        (pl.col("open") / pl.col("close")).alias("feature_open"),
        (pl.col("high") / pl.col("close")).alias("feature_high"),
        (pl.col("low") / pl.col("close")).alias("feature_low"),
    ])
    return df


if __name__ == "__main__":
    csv_path = "data/PHIA.csv"

    df = load_data(csv_path)

    env = TradingEnv(df, window_size=10)
    # env = FlattenObservation(env)
    
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)


    print(20*"-")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation space: {len(obs)}")
    print(f"Action space shape: {env.action_space}")
    print(f"Action space sample: {env.action_space.sample()}")


    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        action = env.action_space.sample() 
        observation, reward, done, truncated, info = env.step(action)
        print(f"Observation: {observation}")
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        print(20*"-")

        print(20*"-")
        print(f"Observation space shape: {env.observation_space.shape}")
        print(f"Observation space: {len(obs)}")
        print(f"Action space shape: {env.action_space}")
        print(f"Action space sample: {env.action_space.sample()}")
        print(20*"-")

