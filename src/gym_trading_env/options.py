import gymnasium as gym
from gymnasium import spaces

class Option:
    def __init__(self, option_type, strike, date_generated, expiry_date, days_to_expire, spot_price, premium):
        self.option_type = option_type
        self.strike = strike
        self.date_generated = date_generated
        self.expiry_date = expiry_date
        self.days_to_expire = days_to_expire
        self.spot_price = spot_price
        self.premium = premium
    
    def __repr__(self):
        return f"Option(type={self.option_type}, strike={self.strike}, expiry={self.expiry_date}, generated on={self.date_generated})"


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