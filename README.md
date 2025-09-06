# Stock Options Trading RL Environment

This project provides a customizable Reinforcement Learning (RL) environment for training agents to trade stock options, built on the `gymnasium` API. The environment simulates daily market movements using historical price data, generates dynamic lists of available options, and manages a portfolio of cash and owned option contracts.

The primary goal is to provide a robust framework for developing and testing RL agents on complex financial decision-making tasks involving options trading.

## Features

-   **Gymnasium Compatible**: Follows the standard `gymnasium.Env` interface for seamless integration with RL libraries like Stable Baselines3 or RLlib.
-   **Dynamic Option Generation**: Automatically generates tradable call and put options at each time step based on the current stock price, configurable strike steps, and expiration months. For now its only posible to go long (buy options)
-   **Portfolio Management**: Includes a detailed `OptionsPortfolio` class to manage cash, track owned options, and calculate total portfolio value.
-   **Structured Observation Space**: Provides a rich, dictionary-based (`gym.spaces.Dict`) observation space that gives the agent a comprehensive view of the market and its portfolio.
-   **Complex Action Space**: Implements a `gym.spaces.MultiDiscrete` action space that allows the agent to simultaneously decide whether to buy from a list of available options and sell from its owned options.
-   **Data-Driven**: Uses historical market data provided as a `polars.DataFrame`, allowing for backtesting on real-world scenarios.

### Future Features (TBI)
-   **Go short**: Include the possibility to go short with options. This will double the action space. 
-   **History**: Include a way to log all the data for future plots. 
-   **Go short**: Include the possibility to go short with options. This will double the action space. 

### Bugs/refactor
-   Include the owned_options field inside the portfolio from the Observation

## Core Components

The environment is built around three core concepts: the **State**, the **Observation**, and the **Action**.

### State Management

The internal **State** represents the complete "ground truth" of the environment at any given time step. It contains all the information needed to run the simulation, much of which is not directly visible to the agent.

The state is managed by a `State` dataclass and includes:
-   `current_step`: The current time step in the dataset.
-   `current_date`: The current simulated date.
-   `current_price`: The closing price of the underlying asset for the current date.
-   `portfolio`: An `OptionsPortfolio` object containing:
    -   `cash`: The current amount of cash available.
    -   `owned_options`: A list of `Option` objects currently held.
    -   `portfolio_value`: The total value (cash + value of owned options).
    -   `value_diff`: The change in portfolio value from the previous step.
-   `options_available`: A list of `Option` objects available to be purchased on the current step.

### Observation Space

The **Observation** is the subset of the state that is passed to the agent at each step. This is what the agent "sees" to make its decisions. The observation space is a `gym.spaces.Dict` with a detailed, structured layout.

The observation dictionary has the following keys:

-   **`portfolio`**: A dictionary containing key metrics about the agent's portfolio.
    -   `cash`: Current cash available.
    -   `portfolio_value`: Total current value of the portfolio.
    -   `value_diff`: The change in value from the previous day.
    -   `total_value_diff`: The cumulative change in value since the start of the episode.
-   **`today`**: A dictionary with the Open, High, Low, Close, and Volume (OHLCV) data for the current day.
-   **`last_closes`**: A NumPy array of shape `(window_size,)` containing the closing prices of the last `N` days.
-   **`last_volumes`**: A NumPy array of shape `(window_size,)` containing the trading volumes of the last `N` days.
-   **`available_options`**: A NumPy array of shape `(n_options, 4)` representing the options available for purchase. Each row corresponds to an option and contains:
    1.  `Type`: `0` for Call, `1` for Put.
    2.  `Strike Price`: The strike price of the option.
    3.  `Premium`: The purchase price of the option.
    4.  `Days to Expiry`: The number of days until the option expires.
-   **`owned_options`**: A NumPy array of shape `(max_options, 4)` representing the options currently in the portfolio. The structure is identical to `available_options`. Empty slots are padded with zeros.

### Action Space

The **Action Space** defines the set of all possible moves the agent can make. This environment uses a `gym.spaces.MultiDiscrete` space, which allows for multiple independent decisions to be made in a single action.

The action is a single array that is logically partitioned into two parts:

1.  **Actions for Available Options** (first `n_options` elements):
    -   This part of the array corresponds to the list of `available_options`.
    -   Each element can be `0` (Do Nothing) or `1` (Buy).
    -   For example, if the third element is `1`, the agent will attempt to buy the third option from the `available_options` list.

2.  **Actions for Owned Options** (last `max_options` elements):
    -   This part of the array corresponds to the slots for `owned_options` in the portfolio.
    -   Each element can be `0` (Hold) or `1` (Sell).
    -   For example, if the first element in this partition is `1`, the agent will attempt to sell the option in the first portfolio slot.

The total size of the action space is `n_options + max_options`.

## Environment Workflow

The simulation proceeds in a standard RL loop:

1.  **Reset**: At the start of an episode, the environment is reset to an initial state with a starting cash balance.
2.  **Step**: On each step, the agent receives an `observation`.
3.  **Action**: The agent chooses an `action` from the action space.
4.  **Execution**: The environment's `step` function executes the action:
    -   Buy actions are processed, and if successful, cash is debited and options are added to the portfolio.
    -   Sell actions are processed, and if successful, cash is credited and options are removed from the portfolio.
5.  **State Update**: The environment advances to the next day in the dataset, updating the stock price and re-evaluating the portfolio's total value. A new list of available options is generated.
6.  **Reward Calculation**: A reward is calculated based on the outcome of the action and the change in portfolio value. **Note**: The current `get_reward` function is a placeholder and should be customized for specific training goals.
7.  **Termination**: The episode ends if the agent runs out of money, reaches the end of the dataset, or another termination condition is met.

### What happens every step




## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**
    Ensure you have Poetry installed. Then, run the following command from the project's root directory to install the required packages into a virtual environment.

    ```bash
    poetry install
    ```
