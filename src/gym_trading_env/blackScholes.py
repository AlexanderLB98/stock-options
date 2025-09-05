from scipy.stats import norm
import numpy as np

from datetime import datetime, date
import calendar

import pandas as pd


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
                        sigma: float = 0.2):
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
    - List[dict]: List of option contracts
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

            option = {
                "type": option_type.lower(),
                "strike": strike,
                "current_date": current_date,
                "expiry_date": expiry,
                "days_to_expiry": days_to_expiry,
                "spot_price": spot_price,
                "premium": round(premium, 4)
            }
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
    S0 = 20.4
    K = 26
    T = 1
    r = 0.025
    sigma = 0.2
    q = 0.03

    print(f"Current underlying asset price (S0): {S0}")
    print(f"Option strike price (K): {K}")
    print(f"Time to expiration (T): {T} year(s)")
    print(f"Annual risk-free interest rate (r): {r * 100:.2f}%")
    print(f"Annual volatility (sigma): {sigma * 100:.2f}%")

    price_call = blackScholesCall(S0, K, T, r, sigma, q)
    price_put = blackScholesPut(S0, K, T, r, sigma, q)
    print(f"Black-Scholes call price: {price_call:.4f}")
    print(f"Black-Scholes put price: {price_put:.4f}")


    print("-----------------------------------------------")
    print("-----------------------------------------------")
    print("Option Generation test: ")
    options_call = gen_option_for_date(
        current_date=date(2025, 7, 8),
        option_type='call',
        spot_price=100.0,
        num_strikes=2,
        strike_step_pct=0.1,  # 5%
        n_months=3
        )
    options_put = gen_option_for_date(
        current_date=date(2025, 7, 8),
        option_type='put',
        spot_price=100.0,
        num_strikes=2,
        strike_step_pct=0.2,  # 5%
        n_months=3
        )


    options_pd_call = pd.DataFrame.from_dict(options_call)
    print(options_pd_call)
    options_pd_put = pd.DataFrame.from_dict(options_put)
    print(options_pd_put)

