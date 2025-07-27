# from options import Option

class OptionsPortfolio:
    """
    Portfolio for managing options contracts.
    """
    def __init__(self, initial_cash, max_options=2):
        self.cash = initial_cash
        self.max_options = max_options
        self.owned_options = [None] * max_options  # Each slot can hold an Option or None

    def buy_option(self, option):
        """
        Buy an option if there is space and enough cash.
        """
        if self.cash < option.premium:
            print("Not enough cash to buy option.")
            return False
        for i in range(self.max_options):
            if self.owned_options[i] is None:
                self.owned_options[i] = option
                self.cash -= option.premium
                print(f"Bought option: {option}")
                return True
        print("No slot available to buy more options.")
        return False

    def sell_option(self, index):
        """
        Sell the option at the given index if owned.
        """
        if 0 <= index < self.max_options and self.owned_options[index] is not None:
            option = self.owned_options[index]
            # self.cash += price
            print(f"Sold option: {option}")
            self.owned_options[index] = None
            return True
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
        option_valuations: list or dict mapping each owned option slot to its current value.
        If not provided, assumes options are worth zero.
        """
        value = self.cash
        for opt in self.owned_options:
            print(f"Evaluating option: {opt}")
            option_value = opt.evaluate_option(current_price, current_date) if opt is not None else 0
            print(f"Option value: {option_value}")
            value += option_value
        print(f"Total portfolio value: {value}")
        return value


       #  option_valuations = None # STILL HAVE TO FIX THIS
       #  for i in len(self.owned_options):
       #      if self.owned_options[i] is None:
       #          option_valuations[i] = 0
       #      else:
       #          option_valuations[i] = self.owned_options[i].evaluate_option(current_price, current_date)
       #  value = self.cash
       #  for i, opt in enumerate(self.owned_options):
       #      if opt is not None:
       #          # If option_valuations is a dict, use opt as key; if list, use index
       #          if isinstance(option_valuations, dict):
       #              value += option_valuations.get(opt, 0)
       #          else:
       #              value += option_valuations[i]
       #  return value