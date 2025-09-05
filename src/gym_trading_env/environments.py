from typing import Optional
import gymnasium as gym
from gymnasium import spaces

import polars as pl
import numpy as np
import datetime
import glob
from pathlib import Path    

from collections import Counter

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
                max_options = 2,
                n_strikes = 2,
                n_months = 1,
                strike_step_pct = 0.1
                ):
        self.reward_function = reward_function
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.df = df
 
        # Features
        self._features = [col for col in df.columns if "feature" in col]
        self._nb_features = len(self._features)

        # Options
        self._initialize_options_parameters(max_options, n_strikes, n_months, strike_step_pct)

        # self.action_space = spaces.Discrete(len(positions))
        self.action_space = define_action_space(self)
        # self._initialize_state()
        self.reset()
        self._initialize_observation_space()

    def _get_obs(self):
        pass

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
        pass

    def perform_action(self, action):
        """ Perform the given action in the environment. Buy/Sell/Do nothing with options."""
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
        self.MAX_OPTIONS = max_options  # max options you can own at once
        self.owned_options = [] 
        self.n_owned_options = len(self.owned_options)  # number of owned options
        self.n_strikes = n_strikes  # number of strikes above and below the spot price
        self.n_months = n_months
        self.strike_step_pct = strike_step_pct  # step percentage for strikes
        self.n_options = (self.n_strikes * 2 + 1) * self.n_months * 2  # 2 for call and put options
 
    def _initialize_state(self):
        """ Initialize the state of the environment using the dataclass.
         The state includes:
         - current_step: int : The current step in the episode.
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
        self.state = initialize_state(initial_cash = self.portfolio_initial_value)

    def _initialize_observation_space(self):
        """ Initialize the observation space based on features and options.
         The observation space includes:
         - The features (self._nb_features): open, high, low, close, volume
         - TBI: Include windows of past features.
         - For each option, 4 features: type (call/put), strike, premium, days_to_expiry
         """
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features + self.n_options * 4, ]  # 4 features for each option (type, strike, premium, days_to_expiry)
        )

    

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
    env = TradingEnv(df)

    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)