from stock_options.options import Option
from datetime import datetime, date

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)



class OptionsPortfolio:
    """
    Portfolio for managing options contracts.
    """
    def __init__(self, initial_cash:float, max_options: int=2, transaction_cost: float=0.05):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.max_options = max_options
        self.owned_options = [None] * max_options  # Each slot can hold an Option or None
        self.total_value_diff: float = 0.0
        self.value_diff: float = 0.0
        self.transaction_cost: float = transaction_cost  # 5% transaction cost by default
        self.transaction_history = []  # List to log transactions

    def buy_option(self, option: Option):
        """
        Buy an option if there is space and enough cash. Long Position
        """
        if self.cash < option.premium:
            logger.info("Not enough cash to buy option.")
            return False
        for i in range(self.max_options):
            if self.owned_options[i] is None:
                option.position = "long"
                self.owned_options[i] = option
                self.cash -= option.premium
                logger.info(f"Bought option: {option}")
                return True
        logger.info("No slot available to buy more options.")
        return False

    def go_short(self, option: Option):
        """
        Sell an option if there is space and enough cash. Short Position
        In this case its not neccesary to have the cash, as we receive the premium.
        TBI: Margin requirements and risk management for short positions.
        """
        for i in range(self.max_options):
            if self.owned_options[i] is None:
                option.position = "short"
                self.owned_options[i] = option
                self.cash += option.premium
                logger.info(f"Sold option: {option} as a short position.")
                return True
        logger.info("No slot available to buy more options.")
        return False

    def close_option(self, index, current_date: datetime):
        """
        Sell the option at the given index if owned.

        UPDATE: Change name from sell_option to close_option. This is because we can also close a short position,
        where we will have to buy the option back. This will check if the option is owned (long) or short, and act accordingly.
        In both cases, the option is removed from the portfolio, but if short the value is subtracted from cash instead of added.
        If Long, the value is added to cash as before.
        """
        if 0 <= index < self.max_options and self.owned_options[index] is not None:
            option = self.owned_options[index]
            if current_date == option.date_generated:
                logger.info("Cannot sell an option on the same day it was bought.")
                return False
            if option.position == "short":
                # For short position, we need to buy back the option at current value
                self.cash -= option.value * (1 + self.transaction_cost)  # Pay the current value plus transaction cost
                self.transaction_history.append({
                    "Option": option, 
                    "Strike": option.strike,
                    "Asset_course": option.spot_price,
                    "benefit": option.premium - option.value*(1 + self.transaction_cost)
                })
            elif option.position == "long":
                self.cash += option.value * (1 - self.transaction_cost)  # Assuming value is updated before selling
                self.transaction_history.append({
                    "Option": option, 
                    "Strike": option.strike,
                    "Asset_course": option.spot_price,
                    "benefit": option.value * (1 - self.transaction_cost) - option.premium
                })
            logger.info(f"Sold option: {option}")
            logger.info(f"Sold option for {option.value}. Premium was {option.premium}. Difference: {option.value - option.premium:.2f}")
            self.owned_options[index] = None
            return True
        logger.info(f"No option to sell at slot {index}.")
        return False

    def get_current_total_value(self, price: float, date: datetime) -> float:
        """
        Calculate total current value of owned options and cash.
        Initializes the total value with the current cash, then adds the evaluated value of each owned option.

        If the option is expired, it is sold automatically at current value and removed from the portfolio.
        Aditionally, it shows the difference with previous day.

        Parameters:
        - price (float): Current underlying asset price.
        - date (date): Current date for option valuation.
        """

        total_value = self.cash
        for i, opt in enumerate(self.owned_options):
            if opt is not None:
                option_value = opt.evaluate_option(price, date)
                if opt.expired:
                    logger.info(f"Option at slot {i} has expired and is being removed from portfolio.")
                    self.owned_options[i] = None
                    if opt.position == "long":
                        self.cash += option_value
                    elif opt.position == "short":
                        self.cash -= option_value
                    # Update total_value to reflect the new cash amount
                    total_value = self.cash
                else:
                    # For portfolio valuation: long positions add value, short positions subtract value (liability)
                    if opt.position == "long":
                        total_value += option_value
                    elif opt.position == "short":
                        total_value -= option_value
        # Calculate differences
        self.value_diff = total_value - self.portfolio_value
        self.total_value_diff = (total_value - self.initial_cash) / self.initial_cash
        # self.total_value_diff += self.value_diff # Or just always total_value - initial_cash. SAME?
        self.previous_value = total_value
        return total_value

    def get_portfolio_distribution(self):
        """
        Returns a dictionary with the current portfolio distribution:
        - cash: current cash
        - owned_options: list of dicts with option details (or None if slot is empty)
        """
        options_summary = []
        for opt in self.owned_options:
            if opt is not None:
                options_summary.append({
                    "type": opt.option_type,
                    "strike": opt.strike,
                    "expiry": opt.expiry_date,
                    "premium": opt.premium
                })
            else:
                options_summary.append(None)
        return {
            "cash": self.cash,
            "owned_options": options_summary
        }

    def __repr__(self):
        return f"OptionsPortfolio(cash={self.cash}, owned_options={self.owned_options})"
