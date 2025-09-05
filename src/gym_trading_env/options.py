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
        self.days_to_expire = (self.expiry_date - current_date).days
        if self.days_to_expire <= 0:
            """ Option has expired, return its intrinsic value. IMPLEMENT"""
            pass
        if self.option_type == "call":
            # call_value = blackScholesCall(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            self.value = blackScholesCall(current_price, self.strike, self.days_to_expire / 365.0, self.r, self.sigma, self.q)
            return self.value 
        elif self.option_type == "put":
            # put_value = blackScholesPut(current_price, self.strike, days_to_expire / 365.0, self.r, self.sigma, self.q)
            self.value = blackScholesPut(current_price, self.strike, self.days_to_expire / 365.0, self.r, self.sigma, self.q)
            return self.value

def define_action_space(env: gym.Env) -> spaces.MultiBinary:
    """
    Defines the action space for the trading environment as a MultiBinary space.

    Each element in the MultiBinary array corresponds to an available option (call or put).
    For each option:
        - 1 means "buy" (take action on this option)
        - 0 means "do nothing" (do not act on this option)
    The last options are the owned options slots, where:
        - 1 means "sell" (sell the option in that slot)
        - 0 means "do nothing" (keep the option in that slot)

    The total number of actions (bits) is determined by:
        n_options = ((n_strikes * 2) + 1) * n_months * 2
    where:
        - n_strikes: Number of strikes above and below the spot price
        - n_months: Number of months to consider for options
        - 2: For both call and put options

    Example:
        If n_strikes=1, n_months=1:
            n_options = ((1*2)+1)*1*2 = 6
            action = [1, 0, 0, 1, 0, 0]  # Buy options 0 and 3, do nothing on others

    Returns:
        gymnasium.spaces.MultiBinary: The action space for the environment.
    """
    n_options = (env.n_strikes * 2 + 1) * env.n_months * 2  # 2 for call and put options
    # Add the maximum number of options that can be owned at once
    max_own = env.max_options
    return spaces.MultiBinary(n_options + max_own)

def define_action_space_with_sell(env: gym.Env) -> spaces.MultiDiscrete:
    """
    Defines the action space for the trading environment as a MultiDiscrete space.

    Each element in the MultiDiscrete array corresponds to an available option (call or put).
    For each option:
        - 0 means "sell"
        - 1 means "hold" (do nothing)
        - 2 means "buy"

    The total number of actions is determined by:
        n_options = ((n_strikes * 2) + 1) * n_months * 2
    where:
        - n_strikes: Number of strikes above and below the spot price
        - n_months: Number of months to consider for options
        - 2: For both call and put options

    Example:
        If n_strikes=1, n_months=1:
            n_options = ((1*2)+1)*1*2 = 6
            action = [2, 1, 0, 2, 1, 1]  # Buy options 0 and 3, sell option 2, hold others

    Returns:
        gymnasium.spaces.MultiDiscrete: The action space for the environment.
    """
    n_options = (env.n_strikes * 2 + 1) * env.n_months * 2  # 2 for call and put options
    # Each option: 0=sell, 1=hold, 2=buy
    return spaces.MultiDiscrete([3] * n_options)


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