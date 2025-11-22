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
from stock_options.utils.history import History
from stock_options.utils.data import load_data, flatten_obs, load_random_data
from stock_options.optionsPortfolio import OptionsPortfolio

from stock_options.blackScholes import gen_option_for_date
# from src.options import define_action_space, Option
from stock_options.options import define_action_space, define_action_space_with_sell

# from datetime import datetime, date

import tempfile, os
import warnings
warnings.filterwarnings("error")

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)



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
    - initial_cash : float : The initial value of the portfolio. Default is 1000.
    - window_size : int : The size of the observation window (number of past days to include). Default is 0 (no window).
    - max_options : int : The maximum number of options that can be owned at once. Default is 2.
    - n_strikes : int : The number of strikes above and below the spot price. Default is 2.
    - n_months : int : The number of months to consider for options. Default is 1.
    - strike_step_pct : float : The step percentage for strikes. Default is 0.1 (10%).
    - go_short : bool : Whether to allow short selling of options. Default is False.

    If go_short is True, the agent can sell options it does not own, effectively going short on them. This adds complexity to the 
    environment and may require additional handling of margin and risk. What happens is that, the option is added to the owned_options list
    as a short position.

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
                initial_cash = 1000,
                window_size = 0,
                max_options = 2,
                n_strikes = 2,
                n_months = 1,
                strike_step_pct = 0.1,
                go_short: bool = False,
                mode: str = 'train',
                flatten_observations: bool = False
                ):
        self.initial_cash = float(initial_cash)
        self.input_df = df
        self.window_size = window_size
        self.flatten_observations = flatten_observations

        # Features
        self._features = [col for col in df.columns if "feature" in col]
        self._nb_features = len(self._features)

        # Training or testing mode
        self.mode = mode

        # Options
        self._initialize_options_parameters(max_options, n_strikes, n_months, strike_step_pct)
        
        if go_short:
            logger.info("Short selling of options is ENABLED.")
            self.action_space = define_action_space_with_sell(self)
        else:
            self.action_space = define_action_space(self)
        
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
        # Use default seed if None is provided (e.g., by check_env)
        effective_seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        print(f"Resetting environment with seed {effective_seed}")
        # if self.mode == 'train':
        #     self.df = load_random_data("data/train_data.csv", seed=effective_seed)
        # elif self.mode == 'test':
        #     self.df = load_random_data("data/test_data.csv", seed=effective_seed)
        # else:
        #     raise ValueError("mode must be 'train' or 'test'")
        
        self.df = self._get_random_df(effective_seed)
        
        self.state = self._initialize_state()
        self.state.current_date = self.df[self.state.current_step, "date"]
        self.state.current_price = self.df[self.state.current_step, "close"]
        
        # ---Checks added for debugging---
        assert self.state.portfolio.cash == self.initial_cash, "Initial cash mismatch after reset."
        assert self.state.portfolio.portfolio_value == self.initial_cash, "Initial portfolio value mismatch after reset."
        logger.debug(f"Reset: State initialized. Cash: {self.state.portfolio.cash}, Portfolio Value: {self.state.portfolio.portfolio_value}")
        # --- End of check ---

        # Get list of available options for the current date
        self.state.options_available = self.update_options()
        observation = self._get_obs()
        logger.info(f"Initial observation: {observation}")
        info = self._get_info()
        logger.info(f"Initial info: {info}")
        return observation, info

    def _get_random_df(self, seed):
        """
        Get a random subset from the input_df based on the seed. Generates a random company code and initial date.
        """
        df = self.input_df
        
        company_codes = df["company_code"].unique().to_list()
        random_company_code = company_codes[seed % len(company_codes)]
        df = df.filter(pl.col("company_code") == random_company_code)

        # Selects random initial date within the selected company code data
        dates = df["date"].unique().sort().to_list()
        # Ensure we have enough dates and don't go out of bounds
        # Reserve at least 1000 rows for the environment to work with
        min_required_rows = 1000
        max_start_index = max(0, len(dates) - min_required_rows) if len(dates) > min_required_rows else 0
        
        if max_start_index <= 0:
            # If we don't have enough dates, just use the first date to get maximum data
            random_initial_date = dates[0]
        else:
            random_initial_date = dates[seed % max(1, max_start_index)]
        
        df = df.filter(pl.col("date") >= random_initial_date)
        df = df.sort("date")
        logger.info(f"Selected initial date: {random_initial_date}")
        logger.info(f"DataFrame after date filter has {len(df)} rows")

        return df

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

        self.state.portfolio.portfolio_value = self.get_portfolio_value()

        # Get list of available options for the current date
        self.state.options_available = self.update_options()
        
        observation = self._get_obs()
        info = self._get_info()

        reward = self.get_reward()

        if self.state.current_step >= len(self.df) - 1:
            truncated = True
            print(f"Reached the end of the data. Portfolio value: {self.state.portfolio.portfolio_value}")

        if self.state.portfolio.portfolio_value <= 0:
            print("Portfolio value has dropped to zero or below. Terminating episode.")
            terminated = True
            truncated = True

        return observation, reward, terminated, truncated, info

    def perform_action(self, actions):
        """ Perform the given action in the environment. Buy/Sell/Do nothing with options.
        ACtion could be a discrete list with len of available options, 0 to do nothing, 1 to buy n option and -1 to sell n option.

        The `actions` parameter is an array where elements are partitioned:
        - First `self.n_options` elements: Actions for options available in the market (0=No-op, 1=Buy).
        - Next `self.max_options` elements: Actions for options currently held in the portfolio (0=No-op, 1=Sell).

        """
        # logger.info(f"Actions received: {actions}")
        assert len(actions) == self.n_options + self.max_options, f"Action length {len(actions)} does not match expected {self.n_options + self.max_options}"
        self._perform_for_available_options(actions[:self.n_options]) # Give actions for available options (first n_options elements)
        self._perform_for_owned_options(actions[-self.max_options:]) # Give actions for owned options (last max_options elements)

    def _perform_for_available_options(self, actions):
        """ Perform actions related to available options (buy or do nothing).
            Actions for available options are in the first self.n_options elements of the actions array.

            If action is 1, will instantiate the option. If no option in that index, just skip
        """
        # logger.info(f"Performing actions: {actions}")
        for i, action in enumerate(actions):
            # logger.info(f"action[{i}] = {action}")
            if action == 0:
                # Do nothing
                pass
            elif action == 1:
                # Case of buying an available option (go long)
                logger.info("Buying option (long position)")
                try:
                    # Check if there is an available option at index i
                    option = self.state.options_available[i]
                    if isinstance(option, Option):
                        # Check if option isinstance(Option)
                        self.state.portfolio.buy_option(option)
                        # Update the portfolio and state
                except IndexError:
                    logger.info(f"No available option at index {i}, cannot buy.")
                pass
            elif action == 2:
                # Case of selling an available option (go short)
                logger.info("selling option (short position)")
                try:
                    # Check if there is an available option at index i
                    option = self.state.options_available[i]
                    if isinstance(option, Option):
                        # Check if option isinstance(Option)
                        self.state.portfolio.go_short(option)
                        # self.state.portfolio.buy_option(option)
                        # Update the portfolio and state
                except IndexError:
                    logger.info(f"No available option at index {i}, cannot sell.")
                pass
            else:
                # Invalid action
                logger.info(f"Invalid action {action} at index {i}. Action must be 0 (hold), 1 (buy), or 2 (sell short).")
                pass

    def _perform_for_owned_options(self, actions):
        """ Perform actions related to owned options (sell or do nothing).
            Actions for owned options are in the last self.max_options elements of the actions array.

            If action is 1, will instantiate the option. If no option in that index, just skip
        """
        # logger.info(f"Performing actions: {actions}")
        for i, action in enumerate(actions):
            # logger.info(f"action[{i}] = {action}")
            if action == 0:
                # Do nothing
                pass
            elif action == 1:
                # Case of selling an available option
                logger.info("selling option")
                try:
                    # Check if there is an available option at index i
                    # option = self.state.portfolio.owned_options[i]
                    if isinstance(self.state.portfolio.owned_options[i], Option):
                        # Sell the option
                        self.state.portfolio.close_option(i, self.state.current_date)
                        # Update the portfolio and state
                except IndexError:
                    logger.info(f"No available option at index {i}, cannot sell.")
                pass
            else:
                # Invalid action
                logger.warninginfo(f"Invalid action {action} at index {i}. Action must be 0 (hold), 1 (sell).")
                pass
        
    
    def get_reward(self):
        """ Calculate the reward for the current step
        returns: float : The reward for the current step
        """
        reward = self.state.portfolio.value_diff
        # --- check Reward ---
        assert isinstance(reward, (float)), f"Reward must be a float, got {type(reward)}"
        logger.info(f"Step {self.state.current_step}: Reward={reward}")
        # --- End of check ---
        
        return reward

    def get_portfolio_value(self) -> float:
        """ Calculates and return the current portfolio value. 
        Is used to update the state.portfolio.portfolio_value attribute inside the portfolio object, which is part of the state.
        This happens every step, and also checks if options are expired. If so, sells them and removes them from the list
        """
        return self.state.portfolio.get_current_total_value(self.state.current_price, self.state.current_date)

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
        self.options = [opt for opt in self.options if opt.expiry_date >= self.state.current_date and opt.premium > 0]

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
        # state = initialize_state(current_step = self.window_size, initial_cash = self.initial_cash)
        state = initialize_state(current_step = self.window_size, initial_cash = self.initial_cash, max_options = self.max_options)
        assert state.current_step == self.window_size, "Initial step mismatch."
        assert state.portfolio.cash == self.initial_cash, "Initial cash mismatch."
        assert state.portfolio.portfolio_value == self.initial_cash, "Initial portfolio value mismatch."
        logger.debug(f"State initialized: current_step={state.current_step}, cash={state.portfolio.cash}")
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
        expected_shape = 4 + 5 + 2*N + 4*(M + K)
        logger.info(f"Expected observation shape: {expected_shape}")

        # --- 0. Current state for cash and portfolio value ---
        portfolio = {
            "cash": float(self.state.portfolio.cash),
            "portfolio_value": float(self.state.portfolio.portfolio_value),
            "value_diff": float(self.state.portfolio.value_diff),
            "total_value_diff": float(self.state.portfolio.total_value_diff)
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
            # if i < len(self.state.portfolio.owned_options):
            if self.state.portfolio.owned_options[i] is not None:
                opt = self.state.portfolio.owned_options[i]
                owned_options.append([
                    type_map.get(opt.option_type, -1),
                    opt.strike,
                    opt.premium,
                    opt.days_to_expire
                ])
            else:
                owned_options.append([0, 0.0, 0.0, 0.0])
        owned_options = np.array(owned_options, dtype=np.float32)
        assert owned_options.shape == (K, 4), f"owned_options shape mismatch: {owned_options.shape} vs {(K, 4)}"

        # obs = {
        #     "portfolio": portfolio,
        #     "today": today,
        #     "last_closes": closes,
        #     "last_volumes": volumes,
        #     "available_options": available_options,
        #     "owned_options": owned_options
        # }
        obs = {
            "cash": np.array([self.state.portfolio.cash], dtype=np.float32),
            "portfolio_value": np.array([self.state.portfolio.portfolio_value], dtype=np.float32),
            "value_diff": np.array([self.state.portfolio.value_diff], dtype=np.float32),
            "total_value_diff": np.array([self.state.portfolio.total_value_diff], dtype=np.float32),

            "open": np.array([today["open"]], dtype=np.float32),
            "close": np.array([today["close"]], dtype=np.float32),
            "low": np.array([today["low"]], dtype=np.float32),
            "high": np.array([today["high"]], dtype=np.float32),
            "volume": np.array([today["volume"]], dtype=np.float32),

            "last_closes": closes.astype(np.float32),
            "last_volumes": volumes.astype(np.float32),
            "available_options": available_options.astype(np.float32).flatten(),
            "owned_options": owned_options.astype(np.float32).flatten(),
        }


        # Make sure the observation matches the expected shape
        assert len(flatten_obs(obs)) == expected_shape, f"Observation shape mismatch: got {len(flatten_obs(obs))}, expected {expected_shape}"
        
        # Return flattened observations if requested
        if self.flatten_observations:
            return flatten_obs(obs)
        
        return obs
    
    def _initialize_observation_space(self):
        """ Initialize the observation space based on features and options.
         The observation space includes:
         - Portfolio: cash, portfolio_value, value_diff, total_value_diff
         - The features (self._nb_features): open, high, low, close, volume
         - Include windows of past closes.
         - For each option, 4 features: type (call/put), strike, premium, days_to_expiry

         For example:
         - Portfolio: 4 (cash, portfolio_value, value_diff, total_value_diff)
         - Features: 5 (open, high, low, close, volume)
         - last_closes: window_size (e.g., 10)
         - last_volumes: window_size (e.g., 10)
         - Options: n_options * 4 (e.g., 10 options * 4 features = 40)
         - Owned options: max_options * 4 (e.g., 2 options * 4 features = 8)
         - Total: 4 + 5 + 10 + 10 + 40 + 8 = 73
         """
        N = self.window_size  # window size for last closes
        M = self.n_options # available options
        K = self.max_options # max possible owned options

        logger.info(f"Observation space initialized with N={N}, M={M} (n_options), K={K} (max_options)")
        
        if self.flatten_observations:
            # Calculate total flattened size
            total_size = 4 + 5 + 2*N + 4*(M + K)
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)
            logger.info(f"Using flattened observation space with shape: {total_size}")
        else:
            self.observation_space = Dict({
                # Portfolio
                "cash": Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
                "portfolio_value": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                "value_diff": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                "total_value_diff": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),

                # Today
                "open": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),
                "close": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),
                "low": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),
                "high": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),
                "volume": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),

                # Time series + options
                "last_closes": Box(-np.inf, np.inf, shape=(N,), dtype=np.float32),
                "last_volumes": Box(-np.inf, np.inf, shape=(N,), dtype=np.float32),
                "available_options": Box(-np.inf, np.inf, shape=(M * 4, ), dtype=np.float32),
                "owned_options": Box(-np.inf, np.inf, shape=(K * 4, ), dtype=np.float32),
            })

        logger.debug(f"Full observation space definition: {self.observation_space}")

    def _get_info(self):
        from dataclasses import asdict
        return asdict(self.state)


if __name__ == "__main__":
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)

    logger.info("Initializing environment for basic test...")
    env = TradingEnv(df, window_size=10, n_months=1, go_short=True) # Use a basic reward

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

    while not done and not truncated:
        action = env.action_space.sample() 
        observation, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)

        flat_obs = flatten_obs(observation)
        # Verify flattened observation shape in loop
        assert len(flat_obs) == expected_flat_obs_shape, f"Step {info['current_step']}: Flattened observation shape mismatch: {len(flat_obs)} vs {expected_flat_obs_shape}"

        # High-level checks for critical external state
        assert info["portfolio"].portfolio_value >= 0, f"Step {info['current_step']}: Portfolio value became negative: {info['portfolio'].portfolio_value}"
        assert info["portfolio"].cash >= 0, f"Step {info['current_step']}: Cash became negative: {info['portfolio'].cash}"
        assert len(info["options_available"]) <= env.n_options, f"Step {info['current_step']}: Too many available options: {len(info['options_available'])} > {env.n_options}"
        
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