from dataclasses import dataclass, field
from typing import TypedDict, Optional, List
import polars as pl
import datetime
from stock_options.options import Option
from stock_options.utils.optionsPortfolio import OptionsPortfolio

@dataclass
class State:
    current_step: int = 0
    done: bool = False
    truncated: bool = False
    current_date: Optional[datetime.datetime] = None
    current_price: float = 0.0
    portfolio: OptionsPortfolio = None
    options_available: pl.DataFrame = field(default_factory=pl.DataFrame)
    history: List[dict] = field(default_factory=list)  # Or a more specific structure

def initialize_state(current_step: int = 0, initial_cash: float = 1000.0, max_options: int = 2) -> State:
    return State(
        current_step = current_step, 
        portfolio=OptionsPortfolio(initial_cash=initial_cash, max_options = max_options),
        )