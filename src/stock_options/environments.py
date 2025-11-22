from dataclasses import asdict

from typing import Optional
import gymnasium as gym
from gymnasium import spaces

import polars as pl
import numpy as np
from gymnasium.spaces import Dict, Box

from stock_options.stateManagement import initialize_state, State
from stock_options.utils.data import load_data, flatten_obs, TECHNICAL_INDICATORS_CONFIG
from stock_options.utils.indicators import (
    IncrementalIndicatorCalculator,
    RollingNormalizer,
    create_normalized_features
)
from stock_options.utils.rewards import RewardFactory, BaseReward

# Check env for SB3 compatibility
from stable_baselines3.common.env_checker import check_env
    
import warnings
warnings.filterwarnings("error")

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)



class TradingEnv(gym.Env):
    """
    A trading environment for OpenAI gymnasium to trade stocks with incremental indicator calculation.
    
    This environment follows the standard `gymnasium.Env` interface. At each step, the agent
    receives an observation of the market and its portfolio and must take an action
    related to buying or selling stocks. The goal is to maximize the portfolio's value.
    
    **Key Feature: No Look-Ahead Bias**
    This environment calculates technical indicators incrementally at each step using only
    historical data up to the current timestep. This ensures:
    - Realistic backtesting that matches production behavior
    - No future information leakage
    - Same code works for both training and live trading

    Attributes:
        df (pl.DataFrame): The DataFrame containing all historical OHLCV market data.
        action_space (gym.spaces.Space): The space of actions the agent can take.
        observation_space (gym.spaces.Dict): The structure of the observation space.
        state: A state object (e.g., a dataclass) containing the full internal state of the environment.
        portfolio (Portfolio): The object that manages cash and owned stocks.
        indicator_calculator: Incremental calculator for technical indicators
        normalizer: Rolling normalizer to avoid look-ahead bias in feature scaling

    inputs:
    - df : polars.DataFrame : A dataframe containing the historical OHLCV price data (open, high, low, close, volume).
    - initial_cash : float : The initial value of the portfolio. Default is 1000.
    - window_size : int : The size of the observation window (number of past days to include). Default is 0 (no window).
    - max_shares_short : int : The maximum number of shares that can be shorted.
    - use_incremental_indicators : bool : If True, calculates indicators incrementally (recommended). If False, uses pre-calculated indicators from df.

    On every step, the environment:
        - Calculates technical indicators using only data up to current step
        - Performs the action (buy/sell/hold)
        - Updates the state (current step, date, price)
        - Updates portfolio value
        - Calculates reward
        - Returns normalized observation
    """

    def __init__(self,
                df : pl.DataFrame,
                initial_cash = 1000,
                window_size = 0,
                mode: str = 'train',  # 'train' or 'test' - controls indicator calculation strategy
                flatten_observations: bool = True,  # Default to True for SB3 compatibility
                max_short_positions: int = 5,
                normalization_window: int = 100,  # Window for rolling normalization
                episode_length: Optional[int] = 252,  # Fixed episode length in days (252 = 1 trading year)
                reward_type: str = 'simple',  # Reward strategy: 'simple', 'multi', 'esg', etc.
                reward_config: Optional[dict] = None  # Additional reward configuration
                ):
        """
        Initialize TradingEnv with hybrid indicator calculation strategy.
        
        HYBRID STRATEGY (recommended for TFM):
        =====================================
        
        TRAINING MODE (fast):
            env = TradingEnv(df, mode='train')
            → Uses pre-calculated indicators from data.py (batch mode)
            → ~100x faster for training millions of steps
            → No look-ahead bias IF data.py uses rolling windows correctly
            → Perfect for PPO training iterations
            
        TEST/EVALUATION MODE (realistic):
            env = TradingEnv(df, mode='test')
            → Calculates indicators step-by-step (incremental mode)
            → Slower but proves model works without future data
            → Same code that will run in production with API data
            → Use for final validation and walk-forward testing
            
        This hybrid approach gives you:
        ✅ Fast training iterations
        ✅ Rigorous testing without look-ahead bias
        ✅ Production-ready code
        ✅ Academic rigor for your TFM
        
        EPISODE MANAGEMENT (for TFM comparability):
        ===========================================
        
        Fixed-length episodes (recommended):
            env = TradingEnv(df, episode_length=252)  # 1 trading year
            → Consistent episode length for fair algorithm comparison
            → Better training stability
            → Easier to interpret results
            → Recommended: 200-252 days (6-12 months)
            
        Variable-length episodes (legacy):
            env = TradingEnv(df, episode_length=None)
            → Episode runs until end of data or portfolio=0
            → Higher variance, harder to compare
            
        REWARD CONFIGURATION:
        =====================
        
        Simple baseline:
            env = TradingEnv(df, reward_type='simple')
            
        Risk-adjusted:
            env = TradingEnv(df, reward_type='risk_adjusted', 
                           reward_config={'w_drawdown': 0.2})
            
        Multi-objective:
            env = TradingEnv(df, reward_type='multi',
                           reward_config={'w_returns': 1.0, 'w_drawdown': 0.15})
            
        ESG multi-objective (for TFM):
            env = TradingEnv(df, reward_type='esg',
                           reward_config={'alpha': 0.7, 'secondary_metric': 'esg_score'})
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            initial_cash: Starting portfolio cash
            window_size: Historical window size for observations
            mode: 'train' (fast, batch indicators) or 'test' (slow, incremental)
            flatten_observations: If True, returns flat numpy array (for SB3)
            max_short_positions: Maximum number of short positions allowed
            normalization_window: Window for rolling feature normalization
            episode_length: Fixed number of days per episode (None = variable length)
            reward_type: Type of reward function ('simple', 'multi', 'esg', etc.)
            reward_config: Additional configuration for reward function
        """
        self.initial_cash = float(initial_cash)
        self.input_df = df
        self.window_size = window_size
        self.flatten_observations = flatten_observations
        self.mode = mode.lower()  # Normalize to lowercase
        self.normalization_window = normalization_window
        self.episode_length = episode_length

        self.max_short_positions = max_short_positions
        
        # Initialize reward calculator
        reward_config = reward_config or {}
        self.reward_calculator = RewardFactory.create(reward_type, **reward_config)
        logger.info(f"Reward function: {reward_type} with config {reward_config}")
        
        # Initialize buffers for cumulative returns tracking
        from collections import deque
        self.returns_5d = deque(maxlen=5)   # Last 5 days returns
        self.returns_10d = deque(maxlen=10) # Last 10 days returns
        self.returns_20d = deque(maxlen=20) # Last 20 days returns
        self.portfolio_values = deque(maxlen=20)  # Track portfolio values for returns calculation

        # Determine indicator calculation strategy based on mode
        self.use_incremental_indicators = (self.mode == 'test')
        
        # Initialize incremental indicator calculator and normalizer
        if self.use_incremental_indicators:
            self.indicator_calculator = IncrementalIndicatorCalculator(TECHNICAL_INDICATORS_CONFIG)
            self.normalizer = RollingNormalizer(window=normalization_window)
            logger.info("=" * 70)
            logger.info("🧪 TEST MODE: Step-by-step incremental indicator calculation")
            logger.info("=" * 70)
            logger.info("   Use case: Final testing, evaluation, production deployment")
            logger.info("   Performance: Slower (~100x) but realistic")
            logger.info("   Look-ahead bias: GUARANTEED NONE (uses only past data)")
            logger.info("   Streaming: Ready for real-time API data")
            logger.info("=" * 70)
        else:
            # Use pre-calculated indicators from batch processing (MUCH FASTER for training)
            self._features = [col for col in df.columns if "feature" in col]
            self._nb_features = len(self._features)
            logger.info("=" * 70)
            logger.info("⚡ TRAIN MODE: Pre-calculated batch indicators")
            logger.info("=" * 70)
            logger.info("   Use case: Training (PPO, millions of steps)")
            logger.info("   Performance: FAST (~100x faster than test mode)")
            logger.info("   Look-ahead bias: None (data.py uses rolling windows)")
            logger.info("   Note: Switch to mode='test' for final validation")
            logger.info("=" * 70)

        self.action_space = self.define_action_space()
        
        self._initialize_observation_space()
        self.reset()

    def reset(self, seed: Optional[int] = None):
        """Start a new episode. Initialize the state and prepare incremental indicator calculation.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        # Use default seed if None is provided (e.g., by check_env)
        effective_seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        print(f"Resetting environment with seed {effective_seed}")
        
        self.df = self._get_random_df(effective_seed)
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Reset return tracking buffers
        self.returns_5d.clear()
        self.returns_10d.clear()
        self.returns_20d.clear()
        self.portfolio_values.clear()
        
        # Reset incremental indicator calculator and normalizer
        if self.use_incremental_indicators:
            self.indicator_calculator.reset()
            self.normalizer.reset()
            
            # Warm up the indicator calculator with historical data
            # This ensures indicators are properly initialized before trading starts
            warmup_steps = min(
                self.indicator_calculator.get_required_warmup_steps(),
                len(self.df) - 100  # Leave at least 100 steps for trading
            )
            
            logger.info(f"Warming up indicator calculator with {warmup_steps} historical data points")
            
            for i in range(warmup_steps):
                row = self.df[i]
                ohlcv = {
                    'open': float(row['open'][0]),
                    'high': float(row['high'][0]),
                    'low': float(row['low'][0]),
                    'close': float(row['close'][0]),
                    'volume': float(row['volume'][0]) if 'volume' in row.columns else 0.0
                }
                # Update calculator without using the indicators yet
                _ = self.indicator_calculator.update(ohlcv)
            
            # Start trading after warmup
            self.warmup_offset = warmup_steps
            logger.info(f"Warmup complete. Trading starts at index {self.warmup_offset}")
        else:
            self.warmup_offset = 0
        
        self.state = self._initialize_state()
        self.state.current_date = self.df[self.state.current_step + self.warmup_offset, "date"]
        self.state.current_price = self.df[self.state.current_step + self.warmup_offset, "close"]
        
        # ---Checks added for debugging---
        assert self.state.portfolio.cash == self.initial_cash, "Initial cash mismatch after reset."
        assert self.state.portfolio.portfolio_value == self.initial_cash, "Initial portfolio value mismatch after reset."
        logger.debug(f"Reset: State initialized. Cash: {self.state.portfolio.cash}, Portfolio Value: {self.state.portfolio.portfolio_value}")
        # --- End of check ---

        observation = self._get_obs()
        logger.info(f"Initial observation shape: {observation.shape if isinstance(observation, np.ndarray) else 'dict'}")
        info = self._get_info()
        return observation, info

    def define_action_space(self):
        """
        Define a simple action space (buy/sell/hold) for stocks
        """
        return spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

    def _get_random_df(self, seed):
        """
        Get a random subset from the input_df based on the seed.
        
        Strategy:
        1. Select a random company from available tickers
        2. If episode_length is set, ensure we have enough data for fixed-length episode
        3. Select random start date with enough remaining data
        
        This ensures:
        - Consistent episode lengths for fair algorithm comparison
        - Balanced use of historical data
        - Reproducible experiments with same seed
        
        Returns:
            pl.DataFrame: Filtered dataframe for the episode
        """
        df = self.input_df
        
        # Step 1: Select random company
        company_codes = df["company_code"].unique().to_list()
        random_company_code = company_codes[seed % len(company_codes)]
        df = df.filter(pl.col("company_code") == random_company_code)
        logger.info(f"Selected company: {random_company_code}")

        # Step 2: Select random start date
        dates = df["date"].unique().sort().to_list()
        
        if self.episode_length is not None:
            # Fixed-length episodes: ensure we have enough data
            min_required_rows = self.episode_length + 100  # +100 for warmup
            
            if len(dates) < min_required_rows:
                logger.warning(
                    f"Company {random_company_code} has only {len(dates)} days, "
                    f"but episode_length={self.episode_length} requires {min_required_rows}. "
                    f"Using all available data."
                )
                random_initial_date = dates[0]
            else:
                # Select start date such that we have episode_length days ahead
                max_start_index = len(dates) - min_required_rows
                start_index = seed % max(1, max_start_index)
                random_initial_date = dates[start_index]
                
                logger.info(
                    f"Fixed episode: {self.episode_length} days starting from "
                    f"{random_initial_date} (index {start_index}/{len(dates)})"
                )
        else:
            # Variable-length episodes (legacy mode)
            # Reserve at least 1000 rows for the environment
            min_required_rows = 1000
            max_start_index = max(0, len(dates) - min_required_rows) if len(dates) > min_required_rows else 0
            
            if max_start_index <= 0:
                random_initial_date = dates[0]
            else:
                random_initial_date = dates[seed % max(1, max_start_index)]
            
            logger.info(f"Variable episode: starting from {random_initial_date}")
        
        # Filter data from start date onwards
        df = df.filter(pl.col("date") >= random_initial_date)
        df = df.sort("date")
        
        logger.info(f"DataFrame after filtering: {len(df)} rows available")
        
        return df

    def render(self):
        pass

    def step(self, action):
        """Take a step in the environment with incremental indicator calculation.

        Steps:
        - Save previous state for reward calculation
        - Perform the action (buy/sell/hold)
        - Update the state (current step, date, price)
        - Update indicators with current OHLCV data (no look-ahead bias)
        - Update portfolio value
        - Calculate reward using configured reward function
        - Check termination conditions (fixed episode length, portfolio=0, data end)
        - Get normalized observation
        - Return info

        Args:
            action: The action to take (0=hold, 1=buy, 2=sell)
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

        # Save previous state for reward calculation
        previous_state = self.state
        
        self.perform_action(action)

        self.state.current_step += 1
        logger.info(f"Current step: {self.state.current_step}")

        # Calculate actual dataframe index (accounting for warmup offset)
        df_index = self.state.current_step + self.warmup_offset
        
        # Check termination conditions
        
        # 1. Fixed episode length reached
        if self.episode_length is not None and self.state.current_step >= self.episode_length:
            truncated = True
            logger.info(
                f"Fixed episode length ({self.episode_length}) reached. "
                f"Portfolio value: {self.state.portfolio.portfolio_value:.2f}"
            )
            # Still update state with last available data
            df_index = min(df_index, len(self.df) - 1)
            self.state.current_date = self.df[df_index, "date"]
            self.state.current_price = self.df[df_index, "close"]
        
        # 2. Reached end of available data
        elif df_index >= len(self.df) - 1:
            truncated = True
            logger.info(
                f"Reached end of data. Portfolio value: {self.state.portfolio.portfolio_value:.2f}"
            )
            # Still update state with last available data
            df_index = len(self.df) - 1
            self.state.current_date = self.df[df_index, "date"]
            self.state.current_price = self.df[df_index, "close"]
        else:
            # Normal step: update with next data point
            self.state.current_date = self.df[df_index, "date"]
            self.state.current_price = self.df[df_index, "close"]
            
            # Update incremental indicators with new OHLCV data
            if self.use_incremental_indicators:
                row = self.df[df_index]
                ohlcv = {
                    'open': float(row['open'][0]),
                    'high': float(row['high'][0]),
                    'low': float(row['low'][0]),
                    'close': float(row['close'][0]),
                    'volume': float(row['volume'][0]) if 'volume' in row.columns else 0.0
                }
                # Update calculator - this uses only data up to current step
                _ = self.indicator_calculator.update(ohlcv)

        self.state.portfolio.portfolio_value = self.get_portfolio_value()

        observation = self._get_obs()
        info = self._get_info()

        # Calculate reward using configured reward function
        reward = self.reward_calculator.calculate(previous_state, action, self.state)

        # 3. Portfolio value dropped to zero or below
        if self.state.portfolio.portfolio_value <= 0:
            logger.warning("Portfolio value has dropped to zero or below. Terminating episode.")
            terminated = True
            truncated = True

        # print(f"Step {self.state.current_step}: \ncash: {observation['cash']}, portfolio value: {observation['portfolio_value']}, shares: {observation['shares']}, action: {action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info

    def perform_action(self, action):
        """ Perform the given action in the environment. Buy/Sell/Do nothing.
        
        actions:
            - 0: Do nothing
            - 1: Buy
            - 2: Sell
        """
        logger.info(f"Action received: {action}")
        if action == 1:  # BUY
            price = self.state.current_price
            # case 1: close short position
            if self.state.portfolio.shares < 0:
                # only if agent has cash
                if self.state.portfolio.cash >= price:
                    self.state.portfolio.shares += 1
                    self.state.portfolio.cash -= price
                    logger.info(f"Closing short: buying at {price}. Shares now = {self.state.portfolio.shares}")

            # case 2: long or neutral position
            else:
                if self.state.portfolio.cash >= price:
                    self.state.portfolio.shares += 1
                    self.state.portfolio.cash -= price
                    logger.info(f"Buying long at {price}. Shares now = {self.state.portfolio.shares}")

        elif action == 2:  # SELL
            price = self.state.current_price

            # case 1: agent has long shares, sell one
            if self.state.portfolio.shares > 0:
                self.state.portfolio.shares -= 1
                self.state.portfolio.cash += price
                logger.info(f"Closing long: selling at {price}. Shares now = {self.state.portfolio.shares}")

            # case 2: no shares, open a short
            elif self.state.portfolio.shares == 0:
                if self.state.portfolio.shares - 1 >= -self.max_short_positions:
                    self.state.portfolio.shares -= 1
                    self.state.portfolio.cash += price
                    logger.info(f"Opening short: selling at {price}. Shares now = {self.state.portfolio.shares}")

            # case 3: already short, increase the short until max_short_positions
            else:
                if self.state.portfolio.shares - 1 >= -self.max_short_positions:
                    self.state.portfolio.shares -= 1
                    self.state.portfolio.cash += price
                    logger.info(f"Adding to short: selling at {price}. Shares now = {self.state.portfolio.shares}")
        else:
            logger.info("Holding position: no action taken.")

    def get_reward(self):
        """
        DEPRECATED: Legacy reward method. Use reward_calculator instead.
        
        This method is kept for backward compatibility but is not used
        when reward_calculator is configured (which is always in current version).
        
        Returns: float : The reward for the current step
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
        return self.state.portfolio.get_current_total_value(self.state.current_price)
 
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
         - history: History : The history of the episode.

        returns:
            state: The initialized state object.
        """
        # state = initialize_state(current_step = self.window_size, initial_cash = self.initial_cash)
        state = initialize_state(current_step = self.window_size, initial_cash = self.initial_cash)
        assert state.current_step == self.window_size, "Initial step mismatch."
        assert state.portfolio.cash == self.initial_cash, "Initial cash mismatch."
        assert state.portfolio.portfolio_value == self.initial_cash, "Initial portfolio value mismatch."
        logger.debug(f"State initialized: current_step={state.current_step}, cash={state.portfolio.cash}")
        return state

    def _get_obs(self):
        """"
        Get observation with technical indicators calculated incrementally (no look-ahead bias).
        
        The observation contains:
        
        Portfolio State (5 features):
            - cash: Available cash
            - portfolio_value: Total portfolio value
            - shares: Number of shares held (can be negative for short positions)
            - value_diff: Change in portfolio value since last step
            - total_value_diff: Total change since episode start
            
        Technical Indicators (8 features - calculated incrementally):
            - ma_short: Short-term moving average ratio (MA/close)
            - ma_medium: Medium-term moving average ratio (MA/close)
            - ma_long: Long-term moving average ratio (MA/close)
            - rsi: Relative Strength Index (scaled 0-1)
            - roc: Rate of Change (scaled percentage)
            - volatility: Historical volatility (rolling std of returns)
            - atr: Average True Range (normalized by close)
            - bb_width: Bollinger Bands width (market volatility indicator)
            
        Cumulative Returns (3 features):
            - cum_return_5d: Cumulative return over last 5 days
            - cum_return_10d: Cumulative return over last 10 days
            - cum_return_20d: Cumulative return over last 20 days
            
        Price Extremes (6 features):
            - max_5d, min_5d: Max/min portfolio value in last 5 days
            - max_10d, min_10d: Max/min portfolio value in last 10 days
            - max_20d, min_20d: Max/min portfolio value in last 20 days
            
        Total observation shape: 5 + 8 + 3 + 6 = 22 features
        
        All features are normalized using rolling statistics to avoid look-ahead bias.

        Returns: dict or flattened array: The normalized observation data.
        """""
        
        if self.use_incremental_indicators:
            # Get current raw indicators from incremental calculator
            # These are already calculated using only data up to current step
            df_index = self.state.current_step + self.warmup_offset
            row = self.df[min(df_index, len(self.df) - 1)]
            current_close = float(row['close'][0])
            
            # Get latest indicators from calculator (already using only past data)
            raw_indicators = self.indicator_calculator.update({
                'open': float(row['open'][0]),
                'high': float(row['high'][0]),
                'low': float(row['low'][0]),
                'close': current_close,
                'volume': float(row['volume'][0]) if 'volume' in row.columns else 0.0
            })
            
            # Create normalized features
            features = create_normalized_features(raw_indicators, current_close)
            
            # Apply rolling normalization (no look-ahead bias)
            normalized_features = self.normalizer.normalize(features)
            
            # Build observation from normalized features
            technical_indicators = {
                "ma_short": normalized_features.get("feature_ma_short", 0.0),
                "ma_medium": normalized_features.get("feature_ma_medium", 0.0),
                "ma_long": normalized_features.get("feature_ma_long", 0.0),
                "rsi": normalized_features.get("feature_rsi", 0.0),
                "roc": normalized_features.get("feature_roc", 0.0),
                "volatility": normalized_features.get("feature_volatility", 0.0),
                "atr": normalized_features.get("feature_atr", 0.0),
                "bb_width": normalized_features.get("feature_bb_width", 0.0)
            }
        else:
            # Fallback: use pre-calculated indicators from DataFrame
            # WARNING: This may have look-ahead bias if calculated in batch
            df_index = self.state.current_step + self.warmup_offset
            row = self.df[min(df_index, len(self.df) - 1)]
            technical_indicators = {
                "ma_short": float(row["feature_ma_short"][0]),
                "ma_medium": float(row["feature_ma_medium"][0]),
                "ma_long": float(row["feature_ma_long"][0]),
                "rsi": float(row["feature_rsi"][0]),
                "roc": float(row["feature_roc"][0]),
                "volatility": float(row["feature_volatility"][0]),
                "atr": float(row["feature_atr"][0]),
                "bb_width": float(row["feature_bb_width"][0])
            }
        
        # Calculate cumulative returns and extremes
        current_value = self.state.portfolio.portfolio_value
        
        # Update portfolio value history
        self.portfolio_values.append(current_value)
        
        # Calculate cumulative returns for different windows
        def calc_cumulative_return(window_size):
            """Calculate cumulative return over last N days"""
            if len(self.portfolio_values) < 2:
                return 0.0
            
            # Get values for the window
            values = list(self.portfolio_values)
            if len(values) <= window_size:
                # Not enough history, use all available
                start_value = values[0]
                end_value = values[-1]
            else:
                # Use last N values
                start_value = values[-(window_size)]
                end_value = values[-1]
            
            if start_value == 0:
                return 0.0
            
            return (end_value - start_value) / start_value
        
        def calc_extremes(window_size):
            """Calculate max and min values over last N days"""
            if len(self.portfolio_values) == 0:
                return 0.0, 0.0
            
            values = list(self.portfolio_values)
            if len(values) <= window_size:
                # Use all available
                window_values = values
            else:
                # Use last N values
                window_values = values[-window_size:]
            
            return max(window_values), min(window_values)
        
        # Calculate features
        cum_return_5d = calc_cumulative_return(5)
        cum_return_10d = calc_cumulative_return(10)
        cum_return_20d = calc_cumulative_return(20)
        
        max_5d, min_5d = calc_extremes(5)
        max_10d, min_10d = calc_extremes(10)
        max_20d, min_20d = calc_extremes(20)
        
        obs = {
            # Portfolio state (not normalized - these are absolute values the agent needs)
            "cash": np.array([self.state.portfolio.cash], dtype=np.float32),
            "portfolio_value": np.array([self.state.portfolio.portfolio_value], dtype=np.float32),
            "shares": np.array([self.state.portfolio.shares], dtype=np.int32),
            "value_diff": np.array([self.state.portfolio.value_diff], dtype=np.float32),
            "total_value_diff": np.array([self.state.portfolio.total_value_diff], dtype=np.float32),

            # Technical indicators (normalized with rolling statistics)
            "ma_short": np.array([technical_indicators["ma_short"]], dtype=np.float32),
            "ma_medium": np.array([technical_indicators["ma_medium"]], dtype=np.float32),
            "ma_long": np.array([technical_indicators["ma_long"]], dtype=np.float32),
            "rsi": np.array([technical_indicators["rsi"]], dtype=np.float32),
            "roc": np.array([technical_indicators["roc"]], dtype=np.float32),
            "volatility": np.array([technical_indicators["volatility"]], dtype=np.float32),
            "atr": np.array([technical_indicators["atr"]], dtype=np.float32),
            "bb_width": np.array([technical_indicators["bb_width"]], dtype=np.float32),
            
            # Cumulative returns
            "cum_return_5d": np.array([cum_return_5d], dtype=np.float32),
            "cum_return_10d": np.array([cum_return_10d], dtype=np.float32),
            "cum_return_20d": np.array([cum_return_20d], dtype=np.float32),
            
            # Price extremes
            "max_5d": np.array([max_5d], dtype=np.float32),
            "min_5d": np.array([min_5d], dtype=np.float32),
            "max_10d": np.array([max_10d], dtype=np.float32),
            "min_10d": np.array([min_10d], dtype=np.float32),
            "max_20d": np.array([max_20d], dtype=np.float32),
            "min_20d": np.array([min_20d], dtype=np.float32),
        }

        # Verify observation shape (5 portfolio + 8 technical + 3 returns + 6 extremes)
        expected_shape = 5 + 8 + 3 + 6
        flat_size = len(flatten_obs(obs))
        assert flat_size == expected_shape, f"Observation shape mismatch: got {flat_size}, expected {expected_shape}"
        
        # Return flattened observations if requested
        if self.flatten_observations:
            return flatten_obs(obs)
        
        return obs
    
    def _initialize_observation_space(self):
        """ 
        Initialize the observation space using technical indicators (trend + volatility).
        
        The observation space includes:
         - Portfolio state: cash, portfolio_value, shares, value_diff, total_value_diff (5 features)
         - Technical indicators: MA_short, MA_medium, MA_long, RSI, ROC, Volatility, ATR, BB_Width (8 features)
         - Cumulative returns: cum_return_5d, cum_return_10d, cum_return_20d (3 features)
         - Price extremes: max_5d, min_5d, max_10d, min_10d, max_20d, min_20d (6 features)
         
         Total: 5 + 8 + 3 + 6 = 22 features
         """        
        logger.info(f"Observation space initialized")
        
        if self.flatten_observations:
            # Calculate total flattened size
            total_size = 5 + 8 + 3 + 6  # portfolio + indicators + returns + extremes
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)
            logger.info(f"Using flattened observation space with shape: {total_size}")
        else:
            self.observation_space = Dict({
                # Portfolio state
                "cash": Box(low=0, high=np.inf, shape=(1, ), dtype=np.float32),
                "portfolio_value": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                "shares": Box(low=-self.max_short_positions, high=np.inf, shape=(1, ), dtype=np.int32),
                "value_diff": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                "total_value_diff": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),

                # Technical indicators (trend indicators)
                "ma_short": Box(0.0, 2.0, shape=(1, ), dtype=np.float32),  # Short MA/close ratio
                "ma_medium": Box(0.0, 2.0, shape=(1, ), dtype=np.float32), # Medium MA/close ratio
                "ma_long": Box(0.0, 2.0, shape=(1, ), dtype=np.float32),   # Long MA/close ratio
                "rsi": Box(0.0, 1.0, shape=(1, ), dtype=np.float32),       # RSI scaled to 0-1
                "roc": Box(-1.0, 1.0, shape=(1, ), dtype=np.float32),      # Rate of change scaled
                
                # Volatility indicators
                "volatility": Box(0.0, 1.0, shape=(1, ), dtype=np.float32),  # Historical volatility
                "atr": Box(0.0, 1.0, shape=(1, ), dtype=np.float32),         # ATR normalized
                "bb_width": Box(0.0, 1.0, shape=(1, ), dtype=np.float32),    # BB width
                
                # Cumulative returns (last N days)
                "cum_return_5d": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),   # 5-day cumulative return
                "cum_return_10d": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),  # 10-day cumulative return
                "cum_return_20d": Box(-np.inf, np.inf, shape=(1, ), dtype=np.float32),  # 20-day cumulative return
                
                # Price extremes (max/min in last N days)
                "max_5d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),  # Max value in last 5 days
                "min_5d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),  # Min value in last 5 days
                "max_10d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32), # Max value in last 10 days
                "min_10d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32), # Min value in last 10 days
                "max_20d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32), # Max value in last 20 days
                "min_20d": Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32), # Min value in last 20 days
            })

        logger.debug(f"Full observation space definition: {self.observation_space}")

    def _get_info(self):
        return asdict(self.state)


if __name__ == "__main__":
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)

    logger.info("Initializing environment for basic test...")
    env = TradingEnv(df, window_size=10)

    check_env(env, warn=True, skip_render_check=True)

    # --- Initial State Check ---
    observation, info = env.reset()
    logger.info("Environment reset. Performing initial state assertions.")
    # Assertions on the very first state after reset
    expected_flat_obs_shape = 5 + 8  # 5 portfolio + 8 technical indicators
    assert len(flatten_obs(observation)) == expected_flat_obs_shape, f"Initial flattened observation shape mismatch: {len(flatten_obs(observation))} vs {expected_flat_obs_shape}"
    assert info["current_step"] == env.window_size, f"Initial step should be {env.window_size}, got {info['current_step']}"
    assert info["portfolio"].cash == env.initial_cash, f"Initial cash should be {env.initial_cash}, got {info['portfolio'].cash}"
    assert info["portfolio"].portfolio_value == env.initial_cash, f"Initial portfolio value should be {env.initial_cash}, got {info['portfolio'].portfolio_value}"
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
        
        logger.info(f"Step {info["current_step"]}: Reward={reward:.4f}, Portfolio={info["portfolio"].portfolio_value:.2f}, Cash={info["portfolio"].cash:.2f}")
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