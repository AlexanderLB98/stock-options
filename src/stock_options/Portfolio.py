from datetime import datetime, date

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)



class Portfolio:
    """
    Portfolio for managing stocks.
    """
    def __init__(self, initial_cash:float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.shares: int = 0
        self.total_value_diff: float = 0.0
        self.value_diff: float = 0.0
        self.transaction_history = []  # List to log transactions

    def get_current_total_value(self, price: float) -> float:
        """
        Calculate total current value of owned stocks and cash.
        Initializes the total value with the current cash, then adds the evaluated value of each owned stock.

        Aditionally, it shows the difference with previous day.

        Parameters:
        - price (float): Current underlying asset price.
        - date (date): Current date for stock valuation.
        """
        total_value = self.cash + self.shares * price

        # Calculate differences
        self.value_diff = total_value - self.portfolio_value
        self.total_value_diff = (total_value - self.initial_cash) / self.initial_cash
        # self.total_value_diff += self.value_diff # Or just always total_value - initial_cash. SAME?
        self.previous_value = total_value
        return total_value

    def __repr__(self):
        return f"OptionsPortfolio(cash={self.cash})"
