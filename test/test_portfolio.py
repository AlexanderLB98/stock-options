import pytest
from datetime import date
# Make sure to import your OptionsPortfolio class correctly
# If your file is named portfolio.py, this will work:
from stock_options.utils.optionsPortfolio import OptionsPortfolio 
# If your provided code is in a file named, for example, `my_portfolio_classes.py`,
# you would change the import to: from my_portfolio_classes import OptionsPortfolio


# --- Mock Option Class for Testing ---
# We create a fake "Option" class to simulate the behavior of a real Option
# without needing the full blackScholes implementation for this specific test.
class MockOption:
    """A mock Option class for testing purposes."""
    def __init__(self, name, value_to_return):
        self.name = name  # For easier debugging
        self.mock_value = value_to_return

    def evaluate_option(self, price: float, eval_date: date) -> float:
        """
        A mock evaluation method that ignores price and date, and simply returns
        the pre-configured value for predictable testing.
        """
        return self.mock_value

    def __repr__(self):
        return f"MockOption(name='{self.name}', mock_value={self.mock_value})"


# --- Test Functions for get_current_total_value ---

def test_get_current_total_value_with_no_options():
    """
    Tests that the portfolio value is equal to the cash when no options are owned.
    """
    # Arrange: Create a portfolio with initial cash and no options.
    initial_cash = 1000.0
    portfolio = OptionsPortfolio(initial_cash=initial_cash, max_options=2)
    
    # Act: Calculate the total value.
    current_price = 150.0
    current_date = date(2025, 9, 6)
    total_value = portfolio.get_current_total_value(price=current_price, date=current_date)
    
    # Assert: The total value should be just the cash.
    assert total_value == initial_cash

def test_get_current_total_value_with_one_option():
    """
    Tests that the portfolio value is the sum of cash and one owned option's value.
    """
    # Arrange: Create a portfolio with cash and one mock option.
    initial_cash = 1000.0
    portfolio = OptionsPortfolio(initial_cash=initial_cash, max_options=2)
    
    option_value = 150.75
    mock_option = MockOption(name="AAPL Call", value_to_return=option_value)
    portfolio.owned_options[0] = mock_option # Manually add the option for the test
    
    # Act: Calculate the total value.
    current_price = 150.0
    current_date = date(2025, 9, 6)
    total_value = portfolio.get_current_total_value(price=current_price, date=current_date)
    
    # Assert: The total value should be cash + option value.
    expected_value = initial_cash + option_value
    assert total_value == pytest.approx(expected_value) # Use pytest.approx for float comparison

def test_get_current_total_value_with_full_portfolio():
    """
    Tests that the portfolio value is the sum of cash and all owned options' values.
    """
    # Arrange: Create a portfolio with cash and two mock options.
    initial_cash = 1000.0
    portfolio = OptionsPortfolio(initial_cash=initial_cash, max_options=2)
    
    option1_value = 150.75
    option2_value = 85.25
    portfolio.owned_options[0] = MockOption(name="AAPL Call", value_to_return=option1_value)
    portfolio.owned_options[1] = MockOption(name="TSLA Put", value_to_return=option2_value)
    
    # Act: Calculate the total value.
    current_price = 150.0
    current_date = date(2025, 9, 6)
    total_value = portfolio.get_current_total_value(price=current_price, date=current_date)
    
    # Assert: The total value should be cash + value of option 1 + value of option 2.
    expected_value = initial_cash + option1_value + option2_value
    assert total_value == pytest.approx(expected_value)

def test_get_current_total_value_with_empty_slot():
    """
    Tests that the calculation correctly handles empty slots (None) in the portfolio.
    """
    # Arrange: Create a portfolio with an empty slot between owned options.
    initial_cash = 5000.0
    portfolio = OptionsPortfolio(initial_cash=initial_cash, max_options=3)
    
    option1_value = 200.0
    option3_value = 350.0
    portfolio.owned_options[0] = MockOption(name="GOOG Call", value_to_return=option1_value)
    portfolio.owned_options[1] = None # Empty slot
    portfolio.owned_options[2] = MockOption(name="MSFT Call", value_to_return=option3_value)
    
    # Act: Calculate the total value.
    current_price = 150.0
    current_date = date(2025, 9, 6)
    total_value = portfolio.get_current_total_value(price=current_price, date=current_date)
    
    # Assert: The total value should ignore the empty slot.
    expected_value = initial_cash + option1_value + option3_value
    assert total_value == pytest.approx(expected_value)

def test_get_current_total_value_when_option_value_is_zero():
    """
    Tests that an option with a zero value is correctly added, not affecting the total value.
    """
    # Arrange: Create a portfolio with an option that has a value of 0.
    initial_cash = 1000.0
    portfolio = OptionsPortfolio(initial_cash=initial_cash, max_options=2)
    
    option_value = 0.0
    mock_option = MockOption(name="Expired Call", value_to_return=option_value)
    portfolio.owned_options[0] = mock_option
    
    # Act: Calculate the total value.
    current_price = 150.0
    current_date = date(2025, 9, 6)
    total_value = portfolio.get_current_total_value(price=current_price, date=current_date)
    
    # Assert: The total value should be just the cash.
    expected_value = initial_cash + option_value
    assert total_value == pytest.approx(expected_value)