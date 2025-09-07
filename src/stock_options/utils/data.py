import polars as pl
import numpy as np
import logging


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True # Overwrite any existing logging configuration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pl.read_csv(csv_path, try_parse_dates=True)
    logger.info(len(df), "rows loaded from", csv_path)
    df = df[-1000:]

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
