from gym_trading_env.options import Option


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

    def get_portfolio_value(self, option_valuations):
        """
        Calculate total portfolio value (cash + sum of current option valuations).
        option_valuations: list of current values for each owned option slot (use 0 if None)
        """
        value = self.cash
        for i, opt in enumerate(self.owned_options):
            value += option_valuations[i] if opt is not None else 0
        return value

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

    def valorisation(self, current_price, current_date):
        """
        Returns the total portfolio value: cash + sum of current option valuations.
        Sells (removes) options that have expired.
        """
        value = self.cash
        for i, opt in enumerate(self.owned_options):
            if opt is not None:
                # Update days to expire
                opt.days_to_expire = (opt.expiry_date - current_date).days
                if opt.days_to_expire <= 0:
                    if RENDER:
                        print(f"Option {opt} has expired. Selling at current value.")
                    self.sell_option(i)  # This will add option.value to cash and remove the option
                    continue  # Skip further evaluation for this slot
                option_value = opt.evaluate_option(current_price, current_date)
                value += option_value
                if RENDER:
                    print(f"Evaluating option: {opt}")
                    print(f"Option value: {option_value}")
        if RENDER:
            print(f"Total portfolio value: {value}")
        return value