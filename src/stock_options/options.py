from dataclasses import dataclass, field
from datetime import datetime, date
import gymnasium as gym
from gymnasium import spaces

from typing import Literal

import numpy as np
from scipy.stats import norm

# from gym_trading_env.blackScholes import blackScholesCall, blackScholesPut


@dataclass
class Option:
    option_type: str  # "call" or "put"
    strike: float
    date_generated: date
    expiry_date: date
    days_to_expire: int
    spot_price: float
    premium: float
    position: Literal["long", "short"] = field(default="long")  # "long" or "short", long by default    
    expired: bool = field(default=False)
    value: float = field(default=0.0)
    r: float = field(default=0.01)    # risk-free rate
    sigma: float = field(default=0.2) # volatility
    q: float = field(default=0.0)     # dividend yield

    def evaluate_option(self, current_price, current_date):
        self.days_to_expire = (self.expiry_date - current_date).days
        if self.days_to_expire <= 0:
            # Option has expired, return intrinsic value
            self.expired = True
            if self.option_type == "call":
                self.value = blackScholesCall(current_price, self.strike, 0, self.r, self.sigma, self.q)
            elif self.option_type == "put":
                self.value = blackScholesPut(current_price, self.strike, 0, self.r, self.sigma, self.q)
            return self.value
        if self.option_type == "call":
            self.value = blackScholesCall(current_price, self.strike, self.days_to_expire / 365.0, self.r, self.sigma, self.q)
        elif self.option_type == "put":
            self.value = blackScholesPut(current_price, self.strike, self.days_to_expire / 365.0, self.r, self.sigma, self.q)
        return self.value

    def __repr__(self):
        return (f"Position={self.position}, Option(type={self.option_type}, strike={self.strike}, "
                f"expiry={self.expiry_date}, premium={self.premium}, value={self.value} generated on={self.date_generated})")



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
        - 0 means "hold" (do nothing)
        - 1 means "buy"
        - 2 means "sell"

    Sell (short):
        It is possible to go short (sell a non-owned option) if there is an empty slot in the owned options.
        The Option will be added to the owned options with one of this options (still to decide):
            - negative premium (cash increases)
            - parameter to indicate short position
            - other?

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
    # Add the maximum number of options that can be owned at once
    max_own = env.max_options
    return spaces.MultiDiscrete([3]*n_options + [2]*max_own)


# Black Scholes functions

def blackScholesCall(S, K, T, r, sigma, q=0):
    """
    Calculates the price of a European call option using 
    the Black-Scholes model with optional continuous dividend yield.
    
    Parameters:
    - S: current price of the underlying asset
    - K: strike price
    - T: time to expiration in years
    - r: annual risk-free interest rate
    - sigma: annual volatility of the asset
    - q: continuous dividend yield (default 0, no dividends)
    
    Returns:
    - Call option price
    """
    if T <= 0:
        # If option expired, payoff is immediate intrinsic value
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def blackScholesPut(S, K, T, r, sigma, q=0):
    """
    Calculates the price of a European put option using 
    the Black-Scholes model with optional continuous dividend yield.
    
    Parameters:
    - S: current price of the underlying asset
    - K: strike price
    - T: time to expiration in years
    - r: annual risk-free interest rate
    - sigma: annual volatility of the asset
    - q: continuous dividend yield (default 0, no dividends)
    
    Returns:
    - Put option price
    """
    if T <= 0:
        # If option expired, payoff is immediate intrinsic value
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def gen_option_for_date(current_date: date, option_type: str, spot_price: float, 
                        num_strikes: int, strike_step_pct: float, n_months: int,
                        sigma: float = 0.2) -> list[Option]:
    """
    Generate a list of call or put options for future expiry dates.

    Inputs:
    - current_date (date): The reference date (usually today).
    - option_type (str): 'call' or 'put'
    - spot_price (float): Current price of the underlying asset
    - num_strikes (int): Number of strikes above and below spot
    - strike_step_pct (float): Step between strikes as a % of spot (e.g., 0.05 for 5%)
    - n_months (int): How many months ahead to generate expiries
    - sigma (float): Asset anual volatily. Default 20%.

    Returns:
    - List[Option]: List of option contracts
    """
    expiries = get_third_fridays(n_months=n_months, start_date=current_date)
    strike_step = round(spot_price * strike_step_pct, 2)

    # strikes = [
    #     round(spot_price + i * strike_step, 2)
    #     for i in range(-num_strikes, num_strikes + 1)
    # ]
    strikes = gen_even_int_strikes(spot_price, num_strikes)

    # REFACTOR
    r = 0.1
    

    options = []
    for expiry in expiries:
        days_to_expiry = (expiry - current_date).days
        T = days_to_expiry / 365.0
        for strike in strikes:
            if option_type.lower() == "call":
                premium = blackScholesCall(spot_price, strike, T, r, sigma)
            elif option_type.lower() == "put":
                premium = blackScholesPut(spot_price, strike, T, r, sigma)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            option = Option(
                option_type=option_type.lower(),
                strike=strike,
                date_generated=current_date,
                expiry_date=expiry,
                days_to_expire=days_to_expiry,
                spot_price=spot_price,
                premium=round(premium, 4)
            )
            options.append(option)

    return options

def gen_even_int_strikes(spot_price, num_strikes):
    """ 
    Generate a list of int even numbers around a given spot price
    """ 
    # Round the spot_price to nearest even integer
    nearest_even = int(round(spot_price / 2.0) * 2)

    # Generate strikes around it
    strikes = [nearest_even + 2 * i for i in range(-num_strikes, num_strikes + 1)]
    return strikes

def get_third_fridays(n_months: int, start_date: date = None):
    """
    Generate a list of dates corresponding to the third Friday of each month 
    for the next `n_months`, starting from today (or a custom start_date).

    Parameters:
    - n_months (int): Number of months to generate.
    - start_date (date, optional): Start date. Defaults to today.

    Returns:
    - List[date]: List of third Fridays as datetime.date objects.
    """
    if start_date is None:
        start_date = datetime.today()

    third_fridays = []
    year = start_date.year
    month = start_date.month

    for _ in range(n_months):
        # Get calendar matrix for the month
        month_cal = calendar.monthcalendar(year, month)
        # Find the third Friday
        if month_cal[0][calendar.FRIDAY] != 0:
            third_friday = month_cal[2][calendar.FRIDAY]
        else:
            third_friday = month_cal[3][calendar.FRIDAY]

        third_fridays.append(datetime(year, month, third_friday))

        # Increment month/year
        month += 1
        if month > 12:
            month = 1
            year += 1

    return third_fridays

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