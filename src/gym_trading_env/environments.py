import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import polars as pl
pl.Config.set_tbl_rows(50)
import numpy as np
import datetime
import glob
from pathlib import Path    

from collections import Counter
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

RENDER = False
# MAX_OPTIONS = 10  # max options you can own at once

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
                df : pd.DataFrame,
                reward_function = basic_reward_function,
                portfolio_initial_value = 1000,
                max_options = 2,
                n_strikes = 2,
                n_months = 1,
                strike_step_pct = 0.1
                ):
        self.reward_function = reward_function
        self.portfolio_initial_value = float(portfolio_initial_value)
        self._set_df(df)
 
        self.options = pl.DataFrame()
        self.MAX_OPTIONS = max_options  # max options you can own at once
        self.owned_options = [] 
        self.n_owned_options = len(self.owned_options)  # number of owned options
        self.n_strikes = n_strikes  # number of strikes above and below the spot price
        self.n_months = n_months
        self.strike_step_pct = strike_step_pct  # step percentage for strikes
        self.n_options = (self.n_strikes * 2 + 1) * self.n_months * 2  # 2 for call and put options
       
        # self.action_space = spaces.Discrete(len(positions))
        self.action_space = define_action_space(self)

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features + self.n_options * 4, ]  # 4 features for each option (type, strike, premium, days_to_expiry)
        )

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    
    def reset(self, seed = None, options=None, **kwargs):
        pass


    def render(self):
        pass

    def step(self, action):
        pass

def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col= "date")
    print(len(df), "rows loaded from", csv_path)
    df = df[-1000:]

    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change()
     
    # Create the feature : open[t] / close[t]
    df["feature_open"] = df["open"]/df["close"]
     
    # Create the feature : high[t] / close[t]
    df["feature_high"] = df["high"]/df["close"]
     
    # Create the feature : low[t] / close[t]
    df["feature_low"] = df["low"]/df["close"]


    return df
if __name__ == "__main__":
    csv_path = "data/PHIA.csv"

    df = load_data(csv_path)
    env = TradingEnv(df)

    obs, info = env.reset()
    print("Initial observation:", obs)
