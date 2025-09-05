import pandas as pd
import polars as pl
import gymnasium as gym
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from gym_trading_env.environments import TradingEnv
from gym_trading_env.options import define_action_space

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def load_data(csv_path):
    """Carga los datos de precios desde un CSV."""
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col= "date")
    print(len(df), "rows loaded from", csv_path)
    df = df[-5000:]

    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change().fillna(0)
     
    # Create the feature : open[t] / close[t]
    df["feature_open"] = df["open"]/df["close"]
     
    # Create the feature : high[t] / close[t]
    df["feature_high"] = df["high"]/df["close"]
     
    # Create the feature : low[t] / close[t]
    df["feature_low"] = df["low"]/df["close"]


    return df

def create_env(df):
    """Crea el entorno de trading con las opciones configuradas."""
    env = TradingEnv(
        df=df,
        positions=[0, 1],
        trading_fees = 0.01/100, # 0.01% per stock buy / sell 
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        portfolio_initial_value=10000,
    )
    return env

def run_random_episode(env, max_steps=50):
    """Ejecuta un episodio aleatorio en el entorno."""
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    print(f"Episodio terminado en {step} pasos. Última recompensa: {reward}")

def run(env):
    # Create the PPO model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/", device="cpu")
    model_path = "ppo_trading_model"
    model = PPO.load(model_path, device="cpu", env=env)

    # Run an episode until it ends :
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        # action = env.action_space.sample() 
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        # print(f"Observation: {observation}, action: {action}, reward: {reward}")
        # print(position_index)
        # To render
    print(env.options)
    # At the end of the episode you want to render
    env.save_for_render(dir = "render_logs")

def main():
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)
    env = create_env(df)

    obs, info = env.reset()
    print("Initial observation:", obs)
    run(env)

if __name__ == "__main__":
    main()

