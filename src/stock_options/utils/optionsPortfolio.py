from stock_options.options import Option
from datetime import datetime, date

RENDER = False

class OptionsPortfolio:
    """
    Portfolio for managing options contracts.
    """
    def __init__(self, initial_cash, max_options=2):
        self.cash = initial_cash
        self.max_options = max_options
        self.owned_options = [None] * max_options  # Each slot can hold an Option or None

    def buy_option(self, option: Option):
        """
        Buy an option if there is space and enough cash.
        """
        if self.cash < option.premium:
            if RENDER:
                print("Not enough cash to buy option.")
            return False
        for i in range(self.max_options):
            if self.owned_options[i] is None:
                self.owned_options[i] = option
                self.cash -= option.premium
                if RENDER:
                    print(f"Bought option: {option}")
                return True
        if RENDER:
            print("No slot available to buy more options.")
        return False

    def sell_option(self, index):
        """
        Sell the option at the given index if owned.
        """
        if 0 <= index < self.max_options and self.owned_options[index] is not None:
            option = self.owned_options[index]

            self.cash += option.value  # Assuming value is updated before selling
            if RENDER:
                print(f"Sold option: {option}")
                print(f"Sold option for {option.value}. Premium was {option.premium}. Difference: {option.value - option.premium:.2f}")
            self.owned_options[index] = None
            return True
        if RENDER:
            print(f"No option to sell at slot {index}.")
        return False

    def get_current_total_value(self, price: float, date: datetime) -> float:
        """
        Calculate total current value of owned options and cash.
        Initializes the total value with the current cash, then adds the evaluated value of each owned option.
        Parameters:
        - price (float): Current underlying asset price.
        - date (date): Current date for option valuation.
        """
        total_value = self.cash
        for opt in self.owned_options:
            if opt is not None:
                option_value = opt.evaluate_option(price, date)

                total_value += option_value # opt.value
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
