import random
import polars as pl
import numpy as np
import logging


import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(f"{len(df)} rows loaded from {csv_path}")
    df = df[-1000:]

    # Create the features using Polars expressions
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("feature_close"),
        (pl.col("open") / pl.col("close")).alias("feature_open"),
        (pl.col("high") / pl.col("close")).alias("feature_high"),
        (pl.col("low") / pl.col("close")).alias("feature_low"),
    ])
    return df

def load_random_data(csv_path, seed=None):
    """
    Load the dataset "stock_data_2025_09_10.csv", select a random ticker and initial date
    depending on the seed, and return the corresponding DataFrame.
    """
    seed = seed if seed is not None else np.random.randint(0, 1_000_000)
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(f"{len(df)} rows loaded from {csv_path}")

    company_codes = df["company_code"].unique().to_list()
    random_company_code = company_codes[seed % len(company_codes)]
    df = df.filter(pl.col("company_code") == random_company_code)
    logger.info(f"Selected company code: {random_company_code}")

    # Selects random initial date within the selected company code data
    dates = df["date"].unique().sort().to_list()
    # Ensure we have enough dates and don't go out of bounds
    # Reserve at least 1000 rows for the environment to work with
    min_required_rows = 1000
    max_start_index = max(0, len(dates) - min_required_rows) if len(dates) > min_required_rows else 0
    
    if max_start_index <= 0:
        # If we don't have enough dates, just use the first date to get maximum data
        random_initial_date = dates[0]
    else:
        random_initial_date = dates[seed % max(1, max_start_index)]
    
    df = df.filter(pl.col("date") >= random_initial_date)
    logger.info(f"Selected initial date: {random_initial_date}")
    logger.info(f"DataFrame after date filter has {len(df)} rows")

    # Final check: ensure we have enough rows for the trading environment
    min_rows_needed = 50  # Minimum rows needed 
    if len(df) < min_rows_needed:
        logger.warning(f"Not enough data after filtering ({len(df)} rows). Using full company data.")
        # Use the full company data if filtered data is too small
        df = pl.read_csv(csv_path, try_parse_dates=True)
        df = df.filter(pl.col("company_code") == random_company_code)
        logger.info(f"Using full company data: {len(df)} rows")

    # Make sure date is datetime
    df = df.with_columns(pl.col("date").cast(pl.Datetime))

    # Create the features using Polars expressions
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("feature_close"),
        (pl.col("open") / pl.col("close")).alias("feature_open"),
        (pl.col("high") / pl.col("close")).alias("feature_high"),
        (pl.col("low") / pl.col("close")).alias("feature_low"),
    ])
    return df


def flatten_obs(obs: dict)-> np.ndarray:
    """ Flatten the observation dictionary into a 1D numpy array. """
    flat = []
    for v in obs.values():
        if isinstance(v, dict):
            # Flatten nested dicts (e.g., "today")
            flat.extend(list(v.values()))
        elif isinstance(v, np.ndarray):
            flat.extend(v.flatten())
        else:
            flat.append(v)
    return np.array(flat, dtype=np.float32)


if __name__ == "__main__":
    seed = random.randint(0, 10000)
    df = load_random_data("data/stock_data_2025_09_10.csv", seed=seed)
    # print(df)