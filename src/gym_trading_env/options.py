import gymnasium as gym
from gymnasium import spaces

from .blackScholes import blackScholesCall, blackScholesPut

class Option:
    def __init__(self, option_type, strike, date_generated, expiry_date, days_to_expire, spot_price, premium):
        self.option_type = option_type
        self.strike = strike
        self.date_generated = date_generated
        self.expiry_date = expiry_date
        self.days_to_expire = days_to_expire
        self.spot_price = spot_price
        self.premium = premium
        self.value = premium  # Initial value is the premium paid 

        self.r = 0.01  # Placeholder for risk-free rate, should be set based on market data
        self.sigma = 0.2  # Placeholder for volatility, should be set based on market data
        self.q = 0  # Placeholder for dividend yield, should be set based

    @classmethod
    def from_dict(cls, d):
        """
        Create an Option instance from a dictionary.
        """
        return cls(
            option_type=d["type"][0],
            strike=d["strike"][0],
            date_generated=d["current_date"][0],
            expiry_date=d["expiry_date"][0],
            days_to_expire=d["days_to_expiry"][0],

            spot_price=d["spot_price"][0],
            premium=d["premium"][0]
        )
    
    def __repr__(self):
        return f"Option(type={self.option_type}, strike={self.strike}, expiry={self.expiry_date}, generated on={self.date_generated})"

    def evaluate_option(self, current_price, current_date):
        """
        Evaluate the option's value based on the current price.
        This is a placeholder for actual valuation logic.
        """
        days_to_expire = (self.expiry_date - current_date).days
        if self.option_type == "call":
            # call_value = blackScholesCall(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            self.value = blackScholesCall(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            return self.value 
        elif self.option_type == "put":
            # put_value = blackScholesPut(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            self.value = blackScholesPut(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            return self.value

def define_action_space(env: gym.Env) -> spaces.Discrete:
    """
    Define the action space for the trading environment.
    The action space is a discrete space representing different trading actions.
    
    The action space is dependent on the variables:
    - `MAX_OPTIONS`: The maximum number of options that can be owned at once.
    - The number of options available, which depends on:
        - number of strikes: this is multiplied by 2 (2 strikes above and 2 below 
        the spot price) + the spot price itself.
        - number of months
        - All times 2 because of the two types of options (call and put).
    So its 
        {[(n_strikes * 2) + 1 ] * n_months } * 2 + MAX_OPTIONS
    The case with only 1 month, 1 strike, and max 2 optiones would be:
        {[(1 * 2) + 1] * 1} * 2
        = 3 * 2 = 6
    This means the action space is discrete with 6 actions.
    """
    # n_options = len(env.options)
    n_options = (env.n_strikes * 2 + 1) * env.n_months * 2  # 2 for call and put options
    # Add the maximum number of options that can be owned at once
    max_own = env.MAX_OPTIONS 
    return spaces.Discrete(n_options + max_own)


if __name__ == "__main__":
    import datetime

    # Example usage of Option class
    option_type = "call"
    strike = 120
    date_generated = datetime.date.today()
    expiry_date = date_generated + datetime.timedelta(days=30)
    days_to_expire = (expiry_date - date_generated).days
    spot_price = 123.45
    premium = 2.50

    option = Option(
        option_type=option_type,
        strike=strike,
        date_generated=date_generated,
        expiry_date=expiry_date,
        days_to_expire=days_to_expire,
        spot_price=spot_price,
        premium=premium
    )

    print(option)