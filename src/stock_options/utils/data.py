"""
Data loading and preprocessing utilities for stock trading RL environments.

IMPORTANT ARCHITECTURE NOTE - Hybrid Strategy to Avoid Look-Ahead Bias:
========================================================================

This module provides TWO modes of technical indicator calculation:

1. TRAIN MODE (this file - data.py):
   - Used for: Fast model training with PPO/RL algorithms
   - Calculates indicators for entire DataFrame at once using Polars
   - ~100x faster than test mode
   - NO LOOK-AHEAD BIAS: Uses rolling windows (.rolling_mean, .shift, .ewm_mean)
   - Use case: Training iterations, hyperparameter tuning
   
2. TEST MODE (indicators.py):
   - Used for: Final validation, walk-forward testing, production simulation
   - Calculates indicators step-by-step using only past + current data
   - EXPLICIT PROOF of no look-ahead bias (incremental calculation)
   - Same code that runs in production with API data
   - Use case: Thesis validation, final testing, deployment
   
For your TFM, use:
- mode='train' in TradingEnv during model development (FAST)
- mode='test' in TradingEnv for final validation (RIGOROUS)

The batch-calculated indicators in this file are kept for training efficiency.
When mode='test', the environment uses IncrementalIndicatorCalculator instead.

See docs/HYBRID_INDICATOR_STRATEGY.md for detailed explanation.
"""

import random
import polars as pl
import numpy as np
import logging
from typing import Iterator, Tuple

# Technical indicators configuration
TECHNICAL_INDICATORS_CONFIG = {
    "ma_short": 5,     # Short-term moving average (5 days)
    "ma_medium": 25,   # Medium-term moving average (25 days)
    "ma_long": 50,     # Long-term moving average (50 days)
    "rsi_period": 14,  # RSI calculation period
    "roc_period": 10,  # Rate of Change calculation period
    "volatility_period": 20,  # Historical volatility period
    "atr_period": 14,  # Average True Range period
    "bb_period": 20,   # Bollinger Bands period
    "bb_std": 2.0      # Bollinger Bands standard deviations
}

# Minimum required historical data for technical indicators
MIN_HISTORICAL_DATA = max(TECHNICAL_INDICATORS_CONFIG.values()) + 10  # 60 days buffer


import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def load_data(csv_path):
    """
    Load price data from a CSV file and calculate technical indicators.
    
    The function now calculates technical indicators (moving averages, RSI, ROC)
    instead of basic OHLC ratios, providing more meaningful features for trading.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with technical indicators and feature columns
    """
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(f"{len(df)} rows loaded from {csv_path}")
    
    # Take more data to ensure we have enough for technical indicators
    # We need at least MIN_HISTORICAL_DATA rows before the actual trading data
    total_needed = 1000 + MIN_HISTORICAL_DATA
    df = df[-total_needed:] if len(df) > total_needed else df
    
    # Ensure we have enough data for technical indicators
    if len(df) < MIN_HISTORICAL_DATA:
        logger.warning(f"Insufficient data for technical indicators. Need at least {MIN_HISTORICAL_DATA} rows, got {len(df)}")
        return df
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create feature columns
    df = create_feature_columns(df)
    
    # Drop rows with NaN values (from technical indicator calculations)
    df = df.drop_nulls()
    
    logger.info(f"Data processed with technical indicators. Final shape: {len(df)} rows")
    
    return df

def load_random_data(csv_path, seed=None):
    """
    Load the dataset "stock_data_2025_09_10.csv", select a random ticker and initial date
    depending on the seed, and return the corresponding DataFrame.
    """
    seed = seed if seed is not None else np.random.randint(0, 1_000_000)
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(f"{len(df)} rows loaded from {csv_path}")

    company_codes = df["company_code"].unique().to_list()
    random_company_code = company_codes[seed % len(company_codes)]
    df = df.filter(pl.col("company_code") == random_company_code)
    logger.info(f"Selected company code: {random_company_code}")

    # Select random initial date within the selected company code data
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
    logger.info(f"Selected initial date: {random_initial_date}")
    logger.info(f"DataFrame after date filter has {len(df)} rows")

    # Final check: ensure we have enough rows for technical indicators + trading environment
    min_rows_needed = MIN_HISTORICAL_DATA + 200  # Need historical data + trading data
    if len(df) < min_rows_needed:
        logger.warning(f"Not enough data after filtering ({len(df)} rows). Using full company data.")
        # Use the full company data if filtered data is too small
        df = pl.read_csv(csv_path, try_parse_dates=True)
        df = df.filter(pl.col("company_code") == random_company_code)
        logger.info(f"Using full company data: {len(df)} rows")

    # Make sure date is datetime
    df = df.with_columns(pl.col("date").cast(pl.Datetime))

    # Sort by date to ensure proper technical indicator calculation
    df = df.sort("date")

    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create feature columns  
    df = create_feature_columns(df)
    
    # Drop rows with NaN values (from technical indicator calculations)
    df = df.drop_nulls()
    
    logger.info(f"Random data processed with technical indicators. Final shape: {len(df)} rows")
    
    return df


def flatten_obs(obs)-> np.ndarray:
    """ 
    Flatten the observation dictionary into a 1D numpy array.
    
    Args:
        obs: Either a dictionary of observations or an already flattened numpy array
        
    Returns:
        Flattened numpy array
    """
    # If already a numpy array, just return it
    if isinstance(obs, np.ndarray):
        return obs.flatten()
    
    # If it's a dict, flatten it
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


def generate_walk_forward_windows(
    df_complete_data: pl.DataFrame, 
    T_train: int = 65, 
    T_eval: int = 65, 
    T_step: int = 65
) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Generate sequential pairs (Training, Evaluation) using the 
    Walk-Forward Validation strategy.
    
    This function now accounts for technical indicators by ensuring we start
    after sufficient historical data is available (MIN_HISTORICAL_DATA).
    
    Args:
        df_complete_data: DataFrame with all daily data from the company (with technical indicators)
        T_train: Training window duration (market days) - Fixed value: 65
        T_eval: Evaluation window duration (market days) - Fixed value: 65
        T_step: Window displacement for the next pair - Fixed value: 65
        
    Yields:
        Tuple[pl.DataFrame, pl.DataFrame]: Pair (data_train, data_eval) for each window
    """
    # Start after we have enough historical data for technical indicators
    # This ensures all technical indicators are properly calculated
    start_index = MIN_HISTORICAL_DATA
    total_length = len(df_complete_data)
    
    logger.info(f"Starting Walk-Forward with {total_length} total rows")
    logger.info(f"Starting from index {start_index} (after {MIN_HISTORICAL_DATA} historical data points)")
    logger.info(f"Parameters: T_train={T_train}, T_eval={T_eval}, T_step={T_step}")
    
    window_num = 0
    
    # Main loop: continue while there's enough space for both windows
    while start_index + T_train + T_eval <= total_length:
        # Define training window (include historical data for technical indicators)
        train_start_with_history = max(0, start_index - MIN_HISTORICAL_DATA)
        end_train_index = start_index + T_train
        data_train = df_complete_data[train_start_with_history:end_train_index]
        
        # Define evaluation window (consecutive, non-overlapping)
        # Also include historical data for technical indicators 
        start_eval_index = end_train_index
        eval_start_with_history = max(0, start_eval_index - MIN_HISTORICAL_DATA)
        end_eval_index = start_eval_index + T_eval
        data_eval = df_complete_data[eval_start_with_history:end_eval_index]
        
        # Log current window information
        if "date" in df_complete_data.columns:
            train_start_date = data_train["date"].min()
            train_end_date = data_train["date"].max()
            eval_start_date = data_eval["date"].min()
            eval_end_date = data_eval["date"].max()
            
            logger.info(f"Window {window_num}: Train[{start_index}:{end_train_index}] "
                       f"({train_start_date} - {train_end_date}), "
                       f"Eval[{start_eval_index}:{end_eval_index}] "
                       f"({eval_start_date} - {eval_end_date})")
        else:
            logger.info(f"Window {window_num}: Train[{start_index}:{end_train_index}], "
                       f"Eval[{start_eval_index}:{end_eval_index}]")
        
        # Verify that windows have the correct size (including historical data)
        expected_train_size = T_train + MIN_HISTORICAL_DATA
        expected_eval_size = T_eval + MIN_HISTORICAL_DATA
        
        # Allow for some flexibility at the beginning where we might not have full history
        actual_train_size = len(data_train)
        actual_eval_size = len(data_eval)
        
        logger.debug(f"Window {window_num}: Train size {actual_train_size} (expected ~{expected_train_size}), "
                    f"Eval size {actual_eval_size} (expected ~{expected_eval_size})")
        
        # Ensure we have at least the minimum trading window size
        assert actual_train_size >= T_train, f"Training window too small: {actual_train_size} < {T_train}"
        assert actual_eval_size >= T_eval, f"Evaluation window too small: {actual_eval_size} < {T_eval}"
        
        yield (data_train, data_eval)
        
        # Slide the window
        start_index += T_step
        window_num += 1
    
    logger.info(f"Walk-Forward completed. Generated {window_num} window pairs")


def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate technical indicators for the given DataFrame.
    
    Indicators calculated:
    - Moving averages: short, medium, long periods
    - RSI: Relative Strength Index
    - Rate of Change
    - Historical Volatility: Rolling standard deviation of returns
    - ATR: Average True Range
    - Bollinger Bands Width: Normalized width of Bollinger Bands
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicator columns
    """
    config = TECHNICAL_INDICATORS_CONFIG
    
    # Ensure the DataFrame is sorted by date
    df = df.sort("date")
    
    # Calculate moving averages
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=config["ma_short"]).alias("ma_short"),
        pl.col("close").rolling_mean(window_size=config["ma_medium"]).alias("ma_medium"), 
        pl.col("close").rolling_mean(window_size=config["ma_long"]).alias("ma_long"),
    ])
    
    # Calculate returns for volatility
    df = df.with_columns([
        pl.col("close").pct_change().alias("returns")
    ])
    
    # Calculate Historical Volatility (rolling std of returns)
    df = df.with_columns([
        pl.col("returns").rolling_std(window_size=config["volatility_period"]).alias("historical_volatility")
    ])
    
    # Calculate True Range components for ATR
    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("high_low"),
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("high_close"),
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("low_close")
    ])
    
    # Calculate True Range (max of the three)
    df = df.with_columns([
        pl.max_horizontal("high_low", "high_close", "low_close").alias("true_range")
    ])
    
    # Calculate ATR (Average True Range)
    df = df.with_columns([
        pl.col("true_range").rolling_mean(window_size=config["atr_period"]).alias("atr")
    ])
    
    # Calculate Bollinger Bands
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=config["bb_period"]).alias("bb_middle"),
        pl.col("close").rolling_std(window_size=config["bb_period"]).alias("bb_std")
    ])
    
    df = df.with_columns([
        (pl.col("bb_middle") + config["bb_std"] * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - config["bb_std"] * pl.col("bb_std")).alias("bb_lower")
    ])
    
    # Calculate Bollinger Bands Width (normalized by middle band)
    df = df.with_columns([
        ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias("bb_width")
    ])
    
    # Calculate RSI
    df = df.with_columns([
        # Price changes
        pl.col("close").diff().alias("price_change")
    ])
    
    # Calculate gains and losses for RSI
    df = df.with_columns([
        pl.when(pl.col("price_change") > 0)
        .then(pl.col("price_change"))
        .otherwise(0.0)
        .alias("gain"),
        
        pl.when(pl.col("price_change") < 0)
        .then(-pl.col("price_change"))
        .otherwise(0.0)
        .alias("loss")
    ])
    
    # Calculate RSI using exponential moving average
    df = df.with_columns([
        pl.col("gain").ewm_mean(span=config["rsi_period"]).alias("avg_gain"),
        pl.col("loss").ewm_mean(span=config["rsi_period"]).alias("avg_loss")
    ])
    
    # Calculate RSI
    df = df.with_columns([
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias("rsi")
    ])
    
    # Calculate Rate of Change (ROC)
    df = df.with_columns([
        ((pl.col("close") - pl.col("close").shift(config["roc_period"])) / 
         pl.col("close").shift(config["roc_period"]) * 100.0).alias("roc")
    ])
    
    # NOTE: We keep intermediate columns (returns, true_range, etc.) for transparency
    # and potential future use. In production, these can be recalculated incrementally.
    
    logger.info(f"Technical indicators calculated: MA({config['ma_short']},{config['ma_medium']},{config['ma_long']}), RSI({config['rsi_period']}), ROC({config['roc_period']}), Volatility({config['volatility_period']}), ATR({config['atr_period']}), BB_Width({config['bb_period']})")
    
    return df


def create_feature_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create simplified feature columns using technical indicators instead of raw OHLC.
    
    Features created:
    - feature_ma_short: Short-term moving average (normalized by close)
    - feature_ma_medium: Medium-term moving average (normalized by close)  
    - feature_ma_long: Long-term moving average (normalized by close)
    - feature_rsi: RSI indicator (scaled to 0-1)
    - feature_roc: Rate of change (scaled percentage)
    - feature_volatility: Historical volatility (already a percentage)
    - feature_atr: Average True Range (normalized by close)
    - feature_bb_width: Bollinger Bands width (already normalized)
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with feature columns added
    """
    df = df.with_columns([
        # Normalize moving averages by current close price
        (pl.col("ma_short") / pl.col("close")).alias("feature_ma_short"),
        (pl.col("ma_medium") / pl.col("close")).alias("feature_ma_medium"),
        (pl.col("ma_long") / pl.col("close")).alias("feature_ma_long"),
        
        # Scale RSI to 0-1 range
        (pl.col("rsi") / 100.0).alias("feature_rsi"),
        
        # Rate of change as percentage (scaled to -1 to 1 range approximately)
        (pl.col("roc") / 100.0).alias("feature_roc"),
        
        # Volatility features (already meaningful as-is, just ensure proper scaling)
        pl.col("historical_volatility").alias("feature_volatility"),
        
        # ATR normalized by close price
        (pl.col("atr") / pl.col("close")).alias("feature_atr"),
        
        # Bollinger Bands width (already normalized)
        pl.col("bb_width").alias("feature_bb_width")
    ])
    
    logger.info("Feature columns created: MA, RSI, ROC, Volatility, ATR, BB Width")
    
    return df


if __name__ == "__main__":
    seed = random.randint(0, 10000)
    df = load_random_data("data/stock_data_2025_09_10.csv", seed=seed)
    # print(df)