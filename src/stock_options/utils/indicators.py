"""
Incremental Technical Indicator Calculator for Reinforcement Learning Trading Environments.

This module provides a robust, production-ready system for calculating technical indicators
incrementally without look-ahead bias. It's designed to work seamlessly in both:
- Backtesting: Step-by-step replay of historical data
- Production: Real-time streaming data from APIs

Key Features:
- No look-ahead bias: Only uses data up to current timestep
- Stateful: Maintains rolling buffers efficiently
- Memory efficient: Only keeps necessary historical data
- Deterministic: Reproducible results with same data sequence

Architecture:
    IncrementalIndicatorCalculator: Main class for indicator calculation
    RollingNormalizer: Normalizes features using rolling statistics
    
Usage:
    # Initialize
    calculator = IncrementalIndicatorCalculator(config)
    normalizer = RollingNormalizer(window=100)
    
    # Update with new OHLCV data
    indicators = calculator.update(ohlcv_dict)
    normalized_features = normalizer.normalize(indicators)
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class IncrementalIndicatorCalculator:
    """
    Calculate technical indicators incrementally, maintaining only necessary historical buffers.
    
    This class is designed to avoid look-ahead bias by:
    1. Only using data up to the current timestep
    2. Maintaining rolling buffers with fixed size
    3. Calculating indicators step-by-step as new data arrives
    
    Supports the following indicators:
    - Moving Averages (MA): Short, medium, and long-term
    - Relative Strength Index (RSI): Momentum oscillator
    - Rate of Change (ROC): Price momentum
    - Historical Volatility: Rolling standard deviation of returns
    - Average True Range (ATR): Volatility measure
    - Bollinger Bands Width: Normalized volatility bands
    
    Args:
        config: Dictionary with indicator configuration parameters
        
    Example:
        >>> config = {
        ...     "ma_short": 5, "ma_medium": 25, "ma_long": 50,
        ...     "rsi_period": 14, "roc_period": 10,
        ...     "volatility_period": 20, "atr_period": 14,
        ...     "bb_period": 20, "bb_std": 2.0
        ... }
        >>> calculator = IncrementalIndicatorCalculator(config)
        >>> indicators = calculator.update({"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000})
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the calculator with configuration parameters.
        
        Args:
            config: Dictionary containing all indicator periods and parameters
        """
        self.config = config
        
        # Extract periods
        self.ma_short = config["ma_short"]
        self.ma_medium = config["ma_medium"]
        self.ma_long = config["ma_long"]
        self.rsi_period = config["rsi_period"]
        self.roc_period = config["roc_period"]
        self.volatility_period = config["volatility_period"]
        self.atr_period = config["atr_period"]
        self.bb_period = config["bb_period"]
        self.bb_std = config["bb_std"]
        
        # Calculate maximum buffer size needed
        self.max_period = max(
            self.ma_long,
            self.rsi_period,
            self.roc_period,
            self.volatility_period,
            self.atr_period,
            self.bb_period
        ) + 1  # +1 for safety with diff/shift operations
        
        # Initialize rolling buffers (using deque for O(1) append and automatic size limit)
        self.close_buffer = deque(maxlen=self.max_period)
        self.high_buffer = deque(maxlen=self.max_period)
        self.low_buffer = deque(maxlen=self.max_period)
        self.volume_buffer = deque(maxlen=self.max_period)
        
        # Buffers for derived values
        self.returns_buffer = deque(maxlen=self.volatility_period)
        self.true_range_buffer = deque(maxlen=self.atr_period)
        
        # State for exponential moving averages (RSI calculation)
        self.rsi_avg_gain: Optional[float] = None
        self.rsi_avg_loss: Optional[float] = None
        self.rsi_initialized = False
        
        # Counter for number of updates
        self.n_updates = 0
        
        logger.info(f"IncrementalIndicatorCalculator initialized with max_period={self.max_period}")
    
    def reset(self):
        """
        Reset all buffers and state. Use when starting a new episode/environment.
        """
        self.close_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.volume_buffer.clear()
        self.returns_buffer.clear()
        self.true_range_buffer.clear()
        
        self.rsi_avg_gain = None
        self.rsi_avg_loss = None
        self.rsi_initialized = False
        self.n_updates = 0
        
        logger.debug("IncrementalIndicatorCalculator reset")
    
    def update(self, ohlcv: Dict[str, float]) -> Dict[str, float]:
        """
        Update buffers with new OHLCV data and calculate all indicators.
        
        This method:
        1. Adds new data to rolling buffers
        2. Calculates all technical indicators using only historical + current data
        3. Returns a dictionary with all indicator values
        
        Args:
            ohlcv: Dictionary with keys: 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            Dictionary with all calculated indicators. Returns NaN for indicators
            that don't have enough historical data yet.
            
        Note:
            The first ~50 calls will return NaN for most indicators until
            sufficient historical data is accumulated.
        """
        # Extract OHLCV values
        close = float(ohlcv['close'])
        high = float(ohlcv['high'])
        low = float(ohlcv['low'])
        volume = float(ohlcv.get('volume', 0))
        
        # Add to buffers
        self.close_buffer.append(close)
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.volume_buffer.append(volume)
        
        # Calculate all indicators
        indicators = {}
        
        # 1. Moving Averages
        indicators['ma_short'] = self._calculate_ma(self.ma_short)
        indicators['ma_medium'] = self._calculate_ma(self.ma_medium)
        indicators['ma_long'] = self._calculate_ma(self.ma_long)
        
        # 2. Returns (needed for volatility)
        current_return = self._calculate_return()
        if not np.isnan(current_return):
            self.returns_buffer.append(current_return)
        
        # 3. Historical Volatility
        indicators['historical_volatility'] = self._calculate_volatility()
        
        # 4. True Range and ATR
        true_range = self._calculate_true_range()
        if not np.isnan(true_range):
            self.true_range_buffer.append(true_range)
        indicators['atr'] = self._calculate_atr()
        
        # 5. Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands()
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = bb_width
        
        # 6. RSI
        indicators['rsi'] = self._calculate_rsi()
        
        # 7. Rate of Change
        indicators['roc'] = self._calculate_roc()
        
        # Increment update counter
        self.n_updates += 1
        
        return indicators
    
    def _calculate_ma(self, period: int) -> float:
        """Calculate simple moving average for given period."""
        if len(self.close_buffer) < period:
            return np.nan
        
        # Get last 'period' values and calculate mean
        recent_closes = list(self.close_buffer)[-period:]
        return np.mean(recent_closes)
    
    def _calculate_return(self) -> float:
        """Calculate percentage return from previous close."""
        if len(self.close_buffer) < 2:
            return np.nan
        
        prev_close = list(self.close_buffer)[-2]
        curr_close = list(self.close_buffer)[-1]
        
        if prev_close == 0:
            return np.nan
        
        return (curr_close - prev_close) / prev_close
    
    def _calculate_volatility(self) -> float:
        """Calculate historical volatility (rolling std of returns)."""
        if len(self.returns_buffer) < self.volatility_period:
            return np.nan
        
        returns = list(self.returns_buffer)
        return np.std(returns, ddof=1)  # Sample std
    
    def _calculate_true_range(self) -> float:
        """
        Calculate True Range for current bar.
        
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        """
        if len(self.close_buffer) < 2:
            return np.nan
        
        curr_high = list(self.high_buffer)[-1]
        curr_low = list(self.low_buffer)[-1]
        prev_close = list(self.close_buffer)[-2]
        
        tr = max(
            curr_high - curr_low,
            abs(curr_high - prev_close),
            abs(curr_low - prev_close)
        )
        
        return tr
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range."""
        if len(self.true_range_buffer) < self.atr_period:
            return np.nan
        
        tr_values = list(self.true_range_buffer)[-self.atr_period:]
        return np.mean(tr_values)
    
    def _calculate_bollinger_bands(self) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Tuple of (upper, middle, lower, width) where width is normalized by middle
        """
        if len(self.close_buffer) < self.bb_period:
            return np.nan, np.nan, np.nan, np.nan
        
        recent_closes = list(self.close_buffer)[-self.bb_period:]
        middle = np.mean(recent_closes)
        std = np.std(recent_closes, ddof=1)
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        # Calculate normalized width
        if middle != 0:
            width = (upper - lower) / middle
        else:
            width = np.nan
        
        return upper, middle, lower, width
    
    def _calculate_rsi(self) -> float:
        """
        Calculate RSI using exponential moving average (Wilder's method).
        
        This implementation uses EMA for smoothing, which is the standard
        approach for RSI calculation.
        """
        if len(self.close_buffer) < 2:
            return np.nan
        
        # Calculate price change
        prev_close = list(self.close_buffer)[-2]
        curr_close = list(self.close_buffer)[-1]
        price_change = curr_close - prev_close
        
        # Separate into gain and loss
        gain = max(price_change, 0)
        loss = max(-price_change, 0)
        
        # Initialize RSI on first calculation
        if not self.rsi_initialized:
            if len(self.close_buffer) < self.rsi_period + 1:
                # Not enough data yet
                return np.nan
            
            # Calculate initial averages using simple mean
            gains = []
            losses = []
            closes = list(self.close_buffer)
            
            for i in range(len(closes) - self.rsi_period, len(closes)):
                if i > 0:
                    change = closes[i] - closes[i-1]
                    gains.append(max(change, 0))
                    losses.append(max(-change, 0))
            
            self.rsi_avg_gain = np.mean(gains) if gains else 0
            self.rsi_avg_loss = np.mean(losses) if losses else 0
            self.rsi_initialized = True
        else:
            # Update using exponential moving average (Wilder's smoothing)
            alpha = 1.0 / self.rsi_period
            self.rsi_avg_gain = (1 - alpha) * self.rsi_avg_gain + alpha * gain
            self.rsi_avg_loss = (1 - alpha) * self.rsi_avg_loss + alpha * loss
        
        # Calculate RSI
        if self.rsi_avg_loss == 0:
            if self.rsi_avg_gain == 0:
                return 50.0  # No movement
            else:
                return 100.0  # All gains
        
        rs = self.rsi_avg_gain / self.rsi_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _calculate_roc(self) -> float:
        """
        Calculate Rate of Change.
        
        ROC = ((close - close_n_periods_ago) / close_n_periods_ago) * 100
        """
        if len(self.close_buffer) < self.roc_period + 1:
            return np.nan
        
        closes = list(self.close_buffer)
        curr_close = closes[-1]
        past_close = closes[-(self.roc_period + 1)]
        
        if past_close == 0:
            return np.nan
        
        roc = ((curr_close - past_close) / past_close) * 100.0
        return roc
    
    def get_required_warmup_steps(self) -> int:
        """
        Get the number of warmup steps needed before all indicators are valid.
        
        Returns:
            Number of OHLCV updates needed before indicators are fully calculated
        """
        return self.max_period


class RollingNormalizer:
    """
    Normalize features using rolling statistics to avoid look-ahead bias.
    
    This normalizer:
    1. Maintains a rolling window of historical feature values
    2. Calculates mean and std using only past data
    3. Normalizes current features using these statistics
    
    This approach ensures that normalization doesn't use future information,
    which is critical for realistic backtesting and production deployment.
    
    Args:
        window: Size of rolling window for statistics calculation
        epsilon: Small constant to avoid division by zero
        
    Example:
        >>> normalizer = RollingNormalizer(window=100)
        >>> normalized = normalizer.normalize({"ma_short": 0.98, "rsi": 65.5})
    """
    
    def __init__(self, window: int = 100, epsilon: float = 1e-8):
        """
        Initialize the normalizer.
        
        Args:
            window: Number of past observations to use for statistics
            epsilon: Small constant added to std to prevent division by zero
        """
        self.window = window
        self.epsilon = epsilon
        
        # Dictionary of deques, one per feature
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        
        # Cache for statistics (updated each normalize call)
        self.stats_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"RollingNormalizer initialized with window={window}")
    
    def reset(self):
        """Reset all feature histories. Use when starting a new episode."""
        self.history.clear()
        self.stats_cache.clear()
        logger.debug("RollingNormalizer reset")
    
    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features using rolling statistics.
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            Dictionary of normalized features (z-score normalization)
            
        Note:
            - Returns 0.0 for features with insufficient history
            - Uses (value - mean) / (std + epsilon) for normalization
        """
        normalized = {}
        
        for key, value in features.items():
            # Skip NaN values
            if np.isnan(value):
                normalized[key] = 0.0
                continue
            
            # Add to history
            self.history[key].append(value)
            
            # Calculate statistics if we have enough data
            if len(self.history[key]) >= 2:
                hist_values = np.array(list(self.history[key]))
                mean = np.mean(hist_values)
                std = np.std(hist_values, ddof=1)
                
                # Cache statistics for inspection/debugging
                self.stats_cache[key] = {'mean': mean, 'std': std}
                
                # Normalize using z-score
                normalized[key] = (value - mean) / (std + self.epsilon)
            else:
                # Not enough data, return neutral value
                normalized[key] = 0.0
        
        return normalized
    
    def get_stats(self, feature_name: str) -> Optional[Dict[str, float]]:
        """
        Get cached statistics for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with 'mean' and 'std' keys, or None if not available
        """
        return self.stats_cache.get(feature_name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get all cached statistics.
        
        Returns:
            Dictionary mapping feature names to their statistics
        """
        return self.stats_cache.copy()


def create_normalized_features(
    indicators: Dict[str, float],
    close_price: float
) -> Dict[str, float]:
    """
    Create normalized feature columns from raw indicators.
    
    This function applies the same normalization as in data.py but works
    on individual indicator values rather than DataFrame columns.
    
    Normalization strategy:
    - Moving averages: Normalized by close price (ratio)
    - RSI: Scaled to 0-1 range
    - ROC: Scaled to approximate -1 to 1 range
    - Volatility: Used as-is (already a ratio/percentage)
    - ATR: Normalized by close price
    - BB Width: Used as-is (already normalized)
    
    Args:
        indicators: Dictionary from IncrementalIndicatorCalculator.update()
        close_price: Current close price for normalization
        
    Returns:
        Dictionary with normalized features ready for RL observation
    """
    features = {}
    
    # Normalize moving averages by close price
    if close_price > 0:
        features['feature_ma_short'] = indicators.get('ma_short', np.nan) / close_price
        features['feature_ma_medium'] = indicators.get('ma_medium', np.nan) / close_price
        features['feature_ma_long'] = indicators.get('ma_long', np.nan) / close_price
        features['feature_atr'] = indicators.get('atr', np.nan) / close_price
    else:
        features['feature_ma_short'] = np.nan
        features['feature_ma_medium'] = np.nan
        features['feature_ma_long'] = np.nan
        features['feature_atr'] = np.nan
    
    # Scale RSI to 0-1 range
    rsi = indicators.get('rsi', np.nan)
    features['feature_rsi'] = rsi / 100.0 if not np.isnan(rsi) else np.nan
    
    # Scale ROC (percentage) to approximate -1 to 1 range
    roc = indicators.get('roc', np.nan)
    features['feature_roc'] = roc / 100.0 if not np.isnan(roc) else np.nan
    
    # Volatility and BB width are already normalized
    features['feature_volatility'] = indicators.get('historical_volatility', np.nan)
    features['feature_bb_width'] = indicators.get('bb_width', np.nan)
    
    return features


if __name__ == "__main__":
    # Simple test to verify the calculator works correctly
    import random
    
    # Test configuration
    config = {
        "ma_short": 5,
        "ma_medium": 25,
        "ma_long": 50,
        "rsi_period": 14,
        "roc_period": 10,
        "volatility_period": 20,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0
    }
    
    # Create calculator and normalizer
    calculator = IncrementalIndicatorCalculator(config)
    normalizer = RollingNormalizer(window=100)
    
    print("Testing IncrementalIndicatorCalculator...")
    print(f"Required warmup steps: {calculator.get_required_warmup_steps()}")
    
    # Simulate 100 price bars
    base_price = 100.0
    for i in range(100):
        # Simulate random walk
        base_price += random.uniform(-2, 2)
        high = base_price + random.uniform(0, 1)
        low = base_price - random.uniform(0, 1)
        close = random.uniform(low, high)
        
        ohlcv = {
            'open': base_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': random.randint(1000, 10000)
        }
        
        # Update indicators
        indicators = calculator.update(ohlcv)
        
        # Create features
        features = create_normalized_features(indicators, close)
        
        # Normalize features
        normalized = normalizer.normalize(features)
        
        # Print sample output every 20 steps
        if (i + 1) % 20 == 0:
            print(f"\nStep {i+1}:")
            print(f"  Close: {close:.2f}")
            print(f"  MA Short: {indicators.get('ma_short', np.nan):.2f}")
            print(f"  RSI: {indicators.get('rsi', np.nan):.2f}")
            print(f"  Feature MA Short (normalized): {normalized.get('feature_ma_short', 0):.4f}")
            print(f"  Feature RSI (normalized): {normalized.get('feature_rsi', 0):.4f}")
    
    print("\n✓ Test completed successfully!")
