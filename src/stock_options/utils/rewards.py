"""
Multi-Objective Reward System for Trading Reinforcement Learning.

This module provides a flexible, configurable reward system for trading agents.
It supports multiple reward strategies from simple to complex multi-objective functions.

Key Features:
- Modular: Easy to add new reward components
- Configurable: Tune weights for different objectives
- Academic: Suitable for ablation studies in research
- Production-ready: Tested reward functions used in real trading

Reward Strategies:
    SimpleReward: Just portfolio value change (baseline)
    RiskAdjustedReward: Returns with drawdown penalty
    MultiObjectiveReward: Combines multiple trading objectives
    SharpeReward: Maximize Sharpe-like ratio
    
Usage:
    # Simple baseline
    reward_calc = RewardFactory.create('simple')
    
    # Multi-objective with custom weights
    reward_calc = RewardFactory.create('multi', 
        w_returns=1.0, w_drawdown=0.1, w_transaction=0.01)
    
    # Calculate reward
    reward = reward_calc.calculate(state, action, next_state)
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from abc import ABC, abstractmethod
from collections import deque

logger = logging.getLogger(__name__)


class BaseReward(ABC):
    """
    Abstract base class for reward calculators.
    
    All reward strategies must inherit from this class and implement
    the calculate() method.
    """
    
    @abstractmethod
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """
        Calculate reward for a transition.
        
        Args:
            state: Current state before action
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            next_state: Resulting state after action
            
        Returns:
            Scalar reward value
        """
        pass
    
    def reset(self):
        """Reset any internal state. Called at episode start."""
        pass


class SimpleReward(BaseReward):
    """
    Simple baseline reward: just portfolio value change.
    
    reward = portfolio_value(t+1) - portfolio_value(t)
    
    Pros:
    - Very simple to understand
    - Easy for agent to learn
    - Good baseline for comparison
    
    Cons:
    - Doesn't consider risk
    - Doesn't penalize excessive trading
    - Can lead to risky behavior
    
    Use for:
    - Initial baseline experiments
    - Sanity checks
    - Comparison benchmark
    """
    
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """Return portfolio value change."""
        return next_state.portfolio.value_diff


class RiskAdjustedReward(BaseReward):
    """
    Risk-adjusted reward with drawdown penalty.
    
    reward = returns - drawdown_penalty
    
    This encourages the agent to:
    - Maximize returns
    - Avoid large drawdowns
    
    Args:
        w_returns: Weight for returns component (default: 1.0)
        w_drawdown: Weight for drawdown penalty (default: 0.1)
        max_drawdown_threshold: Drawdown % that triggers penalty (default: 0.05 = 5%)
    
    Example:
        >>> reward_calc = RiskAdjustedReward(w_drawdown=0.2)
        >>> reward = reward_calc.calculate(state, action, next_state)
    """
    
    def __init__(self, 
                 w_returns: float = 1.0,
                 w_drawdown: float = 0.1,
                 max_drawdown_threshold: float = 0.05):
        self.w_returns = w_returns
        self.w_drawdown = w_drawdown
        self.max_drawdown_threshold = max_drawdown_threshold
        self.peak_value = None
        
    def reset(self):
        """Reset peak value tracking."""
        self.peak_value = None
        
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """Calculate risk-adjusted reward."""
        # Component 1: Returns
        returns = next_state.portfolio.value_diff
        
        # Component 2: Drawdown penalty
        current_value = next_state.portfolio.portfolio_value
        
        # Track peak value
        if self.peak_value is None:
            self.peak_value = current_value
        else:
            self.peak_value = max(self.peak_value, current_value)
        
        # Calculate drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            drawdown = 0.0
        
        # Apply penalty if drawdown exceeds threshold
        drawdown_penalty = 0.0
        if drawdown > self.max_drawdown_threshold:
            drawdown_penalty = (drawdown - self.max_drawdown_threshold) * current_value
        
        # Combine components
        reward = (
            self.w_returns * returns -
            self.w_drawdown * drawdown_penalty
        )
        
        return reward


class MultiObjectiveReward(BaseReward):
    """
    Multi-objective reward combining multiple trading goals.
    
    reward = w_returns * returns 
             - w_drawdown * drawdown_penalty
             - w_transaction * transaction_cost
             - w_risk * risk_penalty
             + w_milestone * milestone_bonus
    
    This is the most sophisticated reward function, suitable for:
    - Academic research (ablation studies)
    - Production deployment
    - Fine-tuning agent behavior
    
    Args:
        w_returns: Weight for portfolio returns (default: 1.0)
        w_drawdown: Weight for drawdown penalty (default: 0.1)
        w_transaction: Weight for transaction costs (default: 0.001)
        w_risk: Weight for volatility/risk penalty (default: 0.0, disabled by default)
        w_milestone: Weight for milestone bonuses (default: 0.0, disabled by default)
        transaction_cost_pct: Transaction cost as % of trade value (default: 0.001 = 0.1%)
        max_drawdown_threshold: Drawdown % triggering penalty (default: 0.05 = 5%)
        volatility_window: Window for volatility calculation (default: 20)
        milestone_targets: List of portfolio value milestones for bonuses
        
    Example:
        >>> reward_calc = MultiObjectiveReward(
        ...     w_returns=1.0,
        ...     w_drawdown=0.15,
        ...     w_transaction=0.01,
        ...     w_risk=0.05
        ... )
    """
    
    def __init__(self,
                 w_returns: float = 1.0,
                 w_drawdown: float = 0.1,
                 w_transaction: float = 0.001,
                 w_risk: float = 0.0,
                 w_milestone: float = 0.0,
                 transaction_cost_pct: float = 0.001,
                 max_drawdown_threshold: float = 0.05,
                 volatility_window: int = 20,
                 milestone_targets: Optional[list] = None):
        
        self.w_returns = w_returns
        self.w_drawdown = w_drawdown
        self.w_transaction = w_transaction
        self.w_risk = w_risk
        self.w_milestone = w_milestone
        
        self.transaction_cost_pct = transaction_cost_pct
        self.max_drawdown_threshold = max_drawdown_threshold
        self.volatility_window = volatility_window
        
        # Default milestones: 10%, 25%, 50%, 100% gains
        if milestone_targets is None:
            milestone_targets = [1.1, 1.25, 1.5, 2.0]
        self.milestone_targets = milestone_targets
        self.achieved_milestones = set()
        
        # State tracking
        self.peak_value = None
        self.returns_history = deque(maxlen=volatility_window)
        
        logger.info(f"MultiObjectiveReward initialized: returns={w_returns}, "
                   f"drawdown={w_drawdown}, transaction={w_transaction}, risk={w_risk}")
    
    def reset(self):
        """Reset all tracking variables."""
        self.peak_value = None
        self.returns_history.clear()
        self.achieved_milestones.clear()
        logger.debug("MultiObjectiveReward reset")
    
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """Calculate multi-objective reward."""
        
        # Component 1: Portfolio Returns
        returns = next_state.portfolio.value_diff
        
        # Component 2: Drawdown Penalty
        drawdown_penalty = self._calculate_drawdown_penalty(next_state)
        
        # Component 3: Transaction Cost
        transaction_cost = self._calculate_transaction_cost(state, action)
        
        # Component 4: Risk Penalty (volatility-based)
        risk_penalty = self._calculate_risk_penalty(returns)
        
        # Component 5: Milestone Bonus
        milestone_bonus = self._calculate_milestone_bonus(state, next_state)
        
        # Combine all components
        reward = (
            self.w_returns * returns -
            self.w_drawdown * drawdown_penalty -
            self.w_transaction * transaction_cost -
            self.w_risk * risk_penalty +
            self.w_milestone * milestone_bonus
        )
        
        # Log detailed breakdown (debug level)
        logger.debug(
            f"Reward breakdown: returns={returns:.4f}, "
            f"drawdown_penalty={drawdown_penalty:.4f}, "
            f"transaction_cost={transaction_cost:.4f}, "
            f"risk_penalty={risk_penalty:.4f}, "
            f"milestone_bonus={milestone_bonus:.4f}, "
            f"total={reward:.4f}"
        )
        
        return reward
    
    def _calculate_drawdown_penalty(self, next_state: Any) -> float:
        """Calculate penalty for portfolio drawdown."""
        current_value = next_state.portfolio.portfolio_value
        
        # Track peak value
        if self.peak_value is None:
            self.peak_value = current_value
        else:
            self.peak_value = max(self.peak_value, current_value)
        
        # Calculate drawdown percentage
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            drawdown = 0.0
        
        # Only penalize if drawdown exceeds threshold
        if drawdown > self.max_drawdown_threshold:
            # Penalty proportional to excess drawdown
            excess_drawdown = drawdown - self.max_drawdown_threshold
            penalty = excess_drawdown * current_value
            return penalty
        
        return 0.0
    
    def _calculate_transaction_cost(self, state: Any, action: int) -> float:
        """Calculate transaction cost for the action taken."""
        # No cost for holding
        if action == 0:
            return 0.0
        
        # Cost proportional to trade value
        trade_value = abs(state.current_price)
        cost = trade_value * self.transaction_cost_pct
        
        return cost
    
    def _calculate_risk_penalty(self, current_return: float) -> float:
        """Calculate penalty based on return volatility."""
        if self.w_risk == 0.0:
            return 0.0  # Skip calculation if disabled
        
        # Add current return to history
        self.returns_history.append(current_return)
        
        # Need sufficient history
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(list(self.returns_history), ddof=1)
        
        # Penalty is proportional to volatility
        # Higher volatility = higher penalty
        return volatility
    
    def _calculate_milestone_bonus(self, state: Any, next_state: Any) -> float:
        """Calculate bonus for reaching portfolio value milestones."""
        if self.w_milestone == 0.0:
            return 0.0  # Skip if disabled
        
        initial_value = state.portfolio.cash if hasattr(state.portfolio, 'initial_cash') else 1000.0
        current_value = next_state.portfolio.portfolio_value
        
        # Check if we reached any new milestones
        bonus = 0.0
        for target in self.milestone_targets:
            target_value = initial_value * target
            
            # If we just reached this milestone
            if current_value >= target_value and target not in self.achieved_milestones:
                self.achieved_milestones.add(target)
                # Bonus proportional to achievement (e.g., 10% gain = 10 bonus)
                bonus += (target - 1.0) * 100.0
                logger.info(f"🎉 Milestone achieved: {target:.1%} portfolio growth!")
        
        return bonus


class SharpeReward(BaseReward):
    """
    Sharpe-like reward: maximize risk-adjusted returns.
    
    reward = mean_return / std_return
    
    This encourages consistent returns with low volatility.
    
    Args:
        window: Window size for calculating mean and std (default: 30)
        risk_free_rate: Annualized risk-free rate (default: 0.02 = 2%)
        
    Note:
        This reward can be unstable early in episode when window isn't full.
        Consider using MultiObjectiveReward with risk component instead.
    """
    
    def __init__(self, window: int = 30, risk_free_rate: float = 0.02):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.returns_history = deque(maxlen=window)
        
    def reset(self):
        """Reset return history."""
        self.returns_history.clear()
        
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """Calculate Sharpe-like reward."""
        current_return = next_state.portfolio.value_diff
        self.returns_history.append(current_return)
        
        # Need sufficient history
        if len(self.returns_history) < 2:
            return current_return  # Fallback to simple return
        
        returns = list(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Avoid division by zero
        if std_return < 1e-8:
            return mean_return
        
        # Sharpe-like ratio (without annualization for simplicity)
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return
        
        return sharpe


class ESGMultiObjectiveReward(BaseReward):
    """
    Multi-objective reward combining financial returns with secondary criteria (ESG, growth, etc.).
    
    This reward function is designed for TFM research to show:
    - Multi-objective optimization in RL
    - Socially responsible investing
    - Beyond pure profit maximization
    
    reward = α * normalized_return + (1-α) * normalized_secondary_score
    
    Where secondary score can be:
    - ESG rating (Environmental, Social, Governance)
    - Revenue growth last year
    - Innovation index
    - Any other company-level metric
    
    This is a WEIGHTED SUM approach (simpler than Pareto optimization) suitable
    for a TFM where you want to show multi-objective concepts without excessive
    mathematical complexity.
    
    Args:
        alpha: Weight for financial returns (default: 0.7 = 70% profit, 30% secondary)
        secondary_metric: Name of the column in company metadata with secondary score
        secondary_scale: Expected range of secondary metric for normalization (min, max)
        normalize_returns: Whether to normalize returns to similar scale as secondary metric
        
    Example:
        >>> # 70% profit, 30% ESG
        >>> reward_calc = ESGMultiObjectiveReward(
        ...     alpha=0.7,
        ...     secondary_metric='esg_score',
        ...     secondary_scale=(0, 100)
        ... )
        >>> 
        >>> # 80% profit, 20% revenue growth
        >>> reward_calc = ESGMultiObjectiveReward(
        ...     alpha=0.8,
        ...     secondary_metric='revenue_growth_pct',
        ...     secondary_scale=(-50, 200)
        ... )
    
    Usage in TFM:
        1. Add secondary metrics to company metadata in data processing
        2. Configure reward with desired alpha (test 0.5, 0.7, 0.9 in ablation study)
        3. Compare agents trained with different alphas:
           - Pure profit (alpha=1.0)
           - Balanced (alpha=0.7)
           - ESG-focused (alpha=0.5)
        4. Show trade-off: higher ESG compliance vs lower pure returns
    """
    
    def __init__(self,
                 alpha: float = 0.7,
                 secondary_metric: str = 'esg_score',
                 secondary_scale: tuple = (0, 100),
                 normalize_returns: bool = True,
                 return_window: int = 50):
        """
        Initialize ESG multi-objective reward.
        
        Args:
            alpha: Weight for returns (1-alpha will be weight for secondary metric)
            secondary_metric: Column name with secondary score in company metadata
            secondary_scale: (min, max) expected range for normalization
            normalize_returns: If True, normalize returns to 0-1 scale
            return_window: Window for return normalization statistics
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        self.alpha = alpha
        self.secondary_metric = secondary_metric
        self.secondary_min, self.secondary_max = secondary_scale
        self.normalize_returns = normalize_returns
        self.return_window = return_window
        
        # For return normalization
        self.returns_history = deque(maxlen=return_window)
        
        logger.info(
            f"ESGMultiObjectiveReward: α={alpha:.2f} "
            f"({alpha:.0%} returns, {1-alpha:.0%} {secondary_metric})"
        )
    
    def reset(self):
        """Reset return history."""
        self.returns_history.clear()
    
    def calculate(self, state: Any, action: int, next_state: Any) -> float:
        """
        Calculate multi-objective reward.
        
        The state/next_state objects should have a company_metadata attribute
        with the secondary metric value.
        """
        # Component 1: Financial returns
        raw_return = next_state.portfolio.value_diff
        self.returns_history.append(raw_return)
        
        if self.normalize_returns and len(self.returns_history) >= 2:
            # Normalize to 0-1 range using historical stats
            returns = list(self.returns_history)
            min_ret = np.min(returns)
            max_ret = np.max(returns)
            
            if max_ret - min_ret > 1e-8:
                normalized_return = (raw_return - min_ret) / (max_ret - min_ret)
            else:
                normalized_return = 0.5  # Neutral if no variance
        else:
            # First few steps or normalization disabled
            normalized_return = raw_return
        
        # Component 2: Secondary metric (ESG, growth, etc.)
        secondary_value = self._get_secondary_score(state)
        
        # Normalize secondary metric to 0-1 range
        if self.secondary_max - self.secondary_min > 0:
            normalized_secondary = (
                (secondary_value - self.secondary_min) / 
                (self.secondary_max - self.secondary_min)
            )
            # Clip to [0, 1] in case value is out of expected range
            normalized_secondary = np.clip(normalized_secondary, 0, 1)
        else:
            normalized_secondary = 0.5  # Neutral if no range
        
        # Combine with weighted sum
        reward = (
            self.alpha * normalized_return +
            (1 - self.alpha) * normalized_secondary
        )
        
        logger.debug(
            f"ESG Reward: return={raw_return:.4f} (norm={normalized_return:.4f}), "
            f"{self.secondary_metric}={secondary_value:.2f} (norm={normalized_secondary:.4f}), "
            f"total={reward:.4f}"
        )
        
        return reward
    
    def _get_secondary_score(self, state: Any) -> float:
        """
        Extract secondary metric from state.
        
        Expects state to have company_metadata dict with the metric.
        Returns 0 if not found (graceful degradation).
        """
        if hasattr(state, 'company_metadata'):
            metadata = state.company_metadata
            if isinstance(metadata, dict):
                return float(metadata.get(self.secondary_metric, 0))
            elif hasattr(metadata, self.secondary_metric):
                return float(getattr(metadata, self.secondary_metric))
        
        # Fallback: no metadata available
        logger.warning(
            f"No {self.secondary_metric} found in state.company_metadata. "
            f"Using 0. Make sure to add company metadata to environment."
        )
        return 0.0


class RewardFactory:
    """
    Factory for creating reward calculators.
    
    Usage:
        >>> reward_calc = RewardFactory.create('simple')
        >>> reward_calc = RewardFactory.create('risk_adjusted', w_drawdown=0.2)
        >>> reward_calc = RewardFactory.create('multi', w_returns=1.0, w_drawdown=0.15)
    """
    
    @staticmethod
    def create(reward_type: str, **kwargs) -> BaseReward:
        """
        Create a reward calculator.
        
        Args:
            reward_type: Type of reward ('simple', 'risk_adjusted', 'multi', 'sharpe', 'esg')
            **kwargs: Additional arguments for the reward calculator
            
        Returns:
            Configured reward calculator instance
            
        Raises:
            ValueError: If reward_type is unknown
        """
        reward_type = reward_type.lower()
        
        if reward_type == 'simple':
            return SimpleReward()
        
        elif reward_type == 'risk_adjusted':
            return RiskAdjustedReward(**kwargs)
        
        elif reward_type in ['multi', 'multiobjective', 'multi_objective']:
            return MultiObjectiveReward(**kwargs)
        
        elif reward_type == 'sharpe':
            return SharpeReward(**kwargs)
        
        elif reward_type in ['esg', 'esg_multi', 'multi_esg']:
            return ESGMultiObjectiveReward(**kwargs)
        
        else:
            raise ValueError(
                f"Unknown reward type: '{reward_type}'. "
                f"Available: 'simple', 'risk_adjusted', 'multi', 'sharpe', 'esg'"
            )
    
    @staticmethod
    def get_available_types() -> list:
        """Get list of available reward types."""
        return ['simple', 'risk_adjusted', 'multi', 'sharpe', 'esg']
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """
        Get pre-configured reward settings.
        
        Presets:
        - 'conservative': Low risk, penalizes drawdowns heavily
        - 'balanced': Balanced risk/return
        - 'aggressive': High risk tolerance, maximizes returns
        - 'academic': Good for ablation studies
        - 'esg_balanced': 70% returns, 30% ESG score
        - 'esg_focused': 50% returns, 50% ESG score
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Dictionary with reward configuration
        """
        presets = {
            'conservative': {
                'reward_type': 'multi',
                'w_returns': 1.0,
                'w_drawdown': 0.3,  # Heavy drawdown penalty
                'w_transaction': 0.01,  # Discourage excessive trading
                'w_risk': 0.1
            },
            'balanced': {
                'reward_type': 'multi',
                'w_returns': 1.0,
                'w_drawdown': 0.15,
                'w_transaction': 0.005,
                'w_risk': 0.05
            },
            'aggressive': {
                'reward_type': 'multi',
                'w_returns': 1.0,
                'w_drawdown': 0.05,  # Low drawdown penalty
                'w_transaction': 0.001,  # Don't care much about costs
                'w_risk': 0.0  # No risk penalty
            },
            'academic': {
                'reward_type': 'multi',
                'w_returns': 1.0,
                'w_drawdown': 0.1,
                'w_transaction': 0.01,
                'w_risk': 0.05,
                'w_milestone': 0.1  # For motivation
            },
            'esg_balanced': {
                'reward_type': 'esg',
                'alpha': 0.7,  # 70% returns, 30% ESG
                'secondary_metric': 'esg_score',
                'secondary_scale': (0, 100),
                'normalize_returns': True
            },
            'esg_focused': {
                'reward_type': 'esg',
                'alpha': 0.5,  # 50% returns, 50% ESG
                'secondary_metric': 'esg_score',
                'secondary_scale': (0, 100),
                'normalize_returns': True
            },
            'growth_oriented': {
                'reward_type': 'esg',
                'alpha': 0.8,  # 80% returns, 20% revenue growth
                'secondary_metric': 'revenue_growth_pct',
                'secondary_scale': (-50, 200),
                'normalize_returns': True
            }
        }
        
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: '{preset_name}'. "
                f"Available: {list(presets.keys())}"
            )
        
        return presets[preset_name]


if __name__ == "__main__":
    # Simple test
    from dataclasses import dataclass
    
    @dataclass
    class MockPortfolio:
        cash: float
        portfolio_value: float
        shares: int
        value_diff: float
        total_value_diff: float
    
    @dataclass
    class MockState:
        portfolio: MockPortfolio
        current_price: float
        current_step: int
    
    # Test simple reward
    print("Testing SimpleReward...")
    simple = SimpleReward()
    
    state = MockState(
        portfolio=MockPortfolio(1000, 1000, 0, 0, 0),
        current_price=100,
        current_step=0
    )
    next_state = MockState(
        portfolio=MockPortfolio(900, 1100, 1, 100, 100),
        current_price=110,
        current_step=1
    )
    
    reward = simple.calculate(state, 1, next_state)
    print(f"Simple reward: {reward}")
    
    # Test multi-objective
    print("\nTesting MultiObjectiveReward...")
    multi = MultiObjectiveReward(
        w_returns=1.0,
        w_drawdown=0.1,
        w_transaction=0.01
    )
    
    reward = multi.calculate(state, 1, next_state)
    print(f"Multi-objective reward: {reward}")
    
    # Test factory
    print("\nTesting RewardFactory...")
    reward_calc = RewardFactory.create('risk_adjusted', w_drawdown=0.2)
    reward = reward_calc.calculate(state, 1, next_state)
    print(f"Factory-created reward: {reward}")
    
    # Test preset
    print("\nTesting preset configurations...")
    preset = RewardFactory.get_preset('balanced')
    print(f"Balanced preset: {preset}")
    
    print("\n✓ All tests passed!")
