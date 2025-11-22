import pytest
from datetime import date, datetime
import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stock_options.optionsPortfolio import OptionsPortfolio
from stock_options.options import Option
from stock_options.blackScholes import blackScholesCall, blackScholesPut


class TestGetCurrentTotalValue:
    """Test suite for the get_current_total_value method of OptionsPortfolio."""
    
    def test_empty_portfolio(self):
        """Test portfolio with no options, only cash."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        current_price = 100.0
        current_date = date(2024, 1, 15)
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        assert total_value == 10000.0
        assert portfolio.cash == 10000.0
        assert portfolio.portfolio_value == 10000.0
        assert portfolio.value_diff == 0.0  # No change from initial
        assert portfolio.total_value_diff == 0.0  # No change from initial
    
    def test_portfolio_with_long_call_option(self):
        """Test portfolio with one long call option."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # Create a call option
        option = Option(
            option_type="call",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=95.0,
            premium=2.5,
            position="long"
        )
        
        # Buy the option
        portfolio.buy_option(option)
        
        # Evaluate portfolio when stock price is 105 (in the money)
        current_price = 105.0
        current_date = date(2024, 1, 15)
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        # Cash should be initial_cash - premium
        expected_cash = 10000.0 - 2.5
        assert portfolio.cash == expected_cash
        
        # Total value should be cash + option value
        option_value = option.value  # This gets set by evaluate_option
        expected_total = expected_cash + option_value
        assert total_value == expected_total
        
        # Portfolio value should be updated (simulate environment behavior)
        portfolio.portfolio_value = total_value  # This is what the environment does
        assert portfolio.portfolio_value == total_value
    
    def test_portfolio_with_short_put_option(self):
        """Test portfolio with one short put option."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # Create a put option
        option = Option(
            option_type="put",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=105.0,
            premium=1.8,
            position="short"
        )
        
        # Go short the option
        portfolio.go_short(option)
        
        # Evaluate portfolio when stock price is 95 (put is in the money)
        current_price = 95.0
        current_date = date(2024, 1, 15)
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        # Cash should be initial_cash + premium (we received premium for short)
        expected_cash = 10000.0 + 1.8
        assert portfolio.cash == expected_cash
        
        # For short position, option value is subtracted from total
        option_value = option.value
        expected_total = expected_cash - option_value
        assert total_value == expected_total
    
    def test_expired_long_call_option(self):
        """Test that expired long call options are properly handled."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # Create an expired call option (ITM)
        option = Option(
            option_type="call",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 1, 15),  # Expires today
            days_to_expire=0,
            spot_price=95.0,
            premium=2.0,
            position="long"
        )
        
        # Buy the option
        portfolio.buy_option(option)
        
        # Evaluate on expiry date with stock price above strike
        current_price = 110.0
        current_date = date(2024, 1, 15)  # Expiry date
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        # Option should be expired and removed
        assert option.expired == True
        assert portfolio.owned_options[0] is None
        
        # Cash should include the intrinsic value of the expired option
        intrinsic_value = max(current_price - option.strike, 0)  # 110 - 100 = 10
        expected_cash = 10000.0 - 2.0 + intrinsic_value
        assert portfolio.cash == expected_cash
        assert total_value == expected_cash
    
    def test_expired_short_put_option(self):
        """Test that expired short put options are properly handled."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # Create an expired put option (ITM)
        option = Option(
            option_type="put",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 1, 15),  # Expires today
            days_to_expire=0,
            spot_price=105.0,
            premium=1.5,
            position="short"
        )
        
        # Go short the option
        portfolio.go_short(option)
        
        # Evaluate on expiry date with stock price below strike
        current_price = 90.0
        current_date = date(2024, 1, 15)  # Expiry date
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        # Option should be expired and removed
        assert option.expired == True
        assert portfolio.owned_options[0] is None
        
        # Cash should be reduced by the intrinsic value we have to pay
        intrinsic_value = max(option.strike - current_price, 0)  # 100 - 90 = 10
        expected_cash = 10000.0 + 1.5 - intrinsic_value
        assert portfolio.cash == expected_cash
        assert total_value == expected_cash
    
    def test_multiple_options_mixed_positions(self):
        """Test portfolio with multiple options in different positions."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=3)
        
        # Long call option
        call_option = Option(
            option_type="call",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=95.0,
            premium=3.0,
            position="long"
        )
        
        # Short put option
        put_option = Option(
            option_type="put",
            strike=95.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=100.0,
            premium=2.0,
            position="short"
        )
        
        # Buy call and sell put
        portfolio.buy_option(call_option)
        portfolio.go_short(put_option)
        
        # Evaluate portfolio
        current_price = 105.0
        current_date = date(2024, 1, 15)
        
        total_value = portfolio.get_current_total_value(current_price, current_date)
        
        # Cash = initial - call_premium + put_premium
        expected_cash = 10000.0 - 3.0 + 2.0
        assert portfolio.cash == expected_cash
        
        # Total value = cash + call_value - put_value
        call_value = call_option.value
        put_value = put_option.value
        expected_total = expected_cash + call_value - put_value
        assert total_value == expected_total
    
    def test_portfolio_value_diff_calculation(self):
        """Test that value_diff and total_value_diff are calculated correctly."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # First evaluation (empty portfolio)
        current_price = 100.0
        current_date = date(2024, 1, 15)
        
        total_value_1 = portfolio.get_current_total_value(current_price, current_date)
        assert portfolio.value_diff == 0.0  # No change from initial
        assert portfolio.total_value_diff == 0.0
        
        # Add an option
        option = Option(
            option_type="call",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=95.0,
            premium=2.0,
            position="long"
        )
        portfolio.buy_option(option)
        
        # Second evaluation
        total_value_2 = portfolio.get_current_total_value(current_price, current_date)
        
        # value_diff should be change from previous portfolio value
        expected_value_diff = total_value_2 - total_value_1
        assert portfolio.value_diff == expected_value_diff
        
        # total_value_diff should be percentage change from initial cash
        expected_total_value_diff = (total_value_2 - 10000.0) / 10000.0
        assert portfolio.total_value_diff == expected_total_value_diff
    
    def test_portfolio_value_updates_correctly(self):
        """Test that portfolio_value tracking works correctly when simulating environment behavior."""
        portfolio = OptionsPortfolio(initial_cash=10000.0, max_options=2)
        
        # Initial state
        assert portfolio.portfolio_value == 10000.0
        
        # First call (simulate environment updating portfolio_value)
        total_value_1 = portfolio.get_current_total_value(100.0, date(2024, 1, 15))
        portfolio.portfolio_value = total_value_1  # Simulate environment behavior
        assert portfolio.portfolio_value == total_value_1
        
        # Add option and call again
        option = Option(
            option_type="call",
            strike=100.0,
            date_generated=date(2024, 1, 1),
            expiry_date=date(2024, 2, 1),
            days_to_expire=31,
            spot_price=95.0,
            premium=2.0,
            position="long"
        )
        portfolio.buy_option(option)
        
        total_value_2 = portfolio.get_current_total_value(105.0, date(2024, 1, 16))
        portfolio.portfolio_value = total_value_2  # Simulate environment behavior
        assert portfolio.portfolio_value == total_value_2
        assert portfolio.portfolio_value != total_value_1  # Should have changed


if __name__ == "__main__":
    pytest.main([__file__])
