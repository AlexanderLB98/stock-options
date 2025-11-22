"""
Quick Test Experiment - Verify Full Pipeline Works

This script:
1. Loads data
2. Creates environment with new features (22 features, 252-day episodes)
3. Trains PPO for 50k timesteps (quick test)
4. Evaluates on test set
5. Saves results

Purpose: Verify everything works before running long experiments for TFM.
Runtime: ~10-15 minutes
"""

import sys
sys.path.insert(0, '/mnt/home-data/lucas/projects/uoc/tfm/stock-options/src')

import time
from datetime import datetime
import json
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stock_options.environments import TradingEnv
from stock_options.utils.data import load_data

def main():
    print("="*70)
    print("QUICK TEST EXPERIMENT - Verifying Full Pipeline")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    print("1. Loading data...")
    df = load_data("data/PHIA.csv")
    print(f"   ✓ Loaded {len(df)} rows")
    
    # Split into train/test (simple split for this quick test)
    dates = df["date"].unique().sort()
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    df_train = df.filter(pl.col("date").is_in(train_dates))
    df_test = df.filter(pl.col("date").is_in(test_dates))
    
    print(f"   ✓ Train: {len(df_train)} rows ({train_dates[0]} to {train_dates[-1]})")
    print(f"   ✓ Test:  {len(df_test)} rows ({test_dates[0]} to {test_dates[-1]})")
    print()
    
    # ==========================================
    # 2. Create Environment
    # ==========================================
    print("2. Creating environment...")
    
    config = {
        'episode_length': 252,      # 1 year episodes
        'mode': 'train',            # Fast batch mode
        'reward_type': 'simple',    # Baseline reward
        'initial_cash': 10000,
        'flatten_observations': True
    }
    
    env_train = TradingEnv(df_train, **config)
    
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env_train.observation_space.shape}")
    print(f"   - Action space: {env_train.action_space}")
    print(f"   - Episode length: {config['episode_length']} days")
    print(f"   - Reward type: {config['reward_type']}")
    print()
    
    # Test observation
    obs, info = env_train.reset()
    print(f"   ✓ Test observation shape: {obs.shape} (expected: (22,))")
    assert obs.shape == (22,), f"Expected 22 features, got {obs.shape[0]}"
    print()
    
    # ==========================================
    # 3. Train PPO (Quick Test)
    # ==========================================
    print("3. Training PPO (50k timesteps - quick test)...")
    print("   This will take ~10-15 minutes...")
    print()
    
    start_time = time.time()
    
    model = PPO(
        'MlpPolicy',
        env_train,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard_quick_test/"
    )
    
    model.learn(total_timesteps=50_000)
    
    train_time = time.time() - start_time
    print()
    print(f"   ✓ Training complete in {train_time/60:.1f} minutes")
    print()
    
    # Save model
    model.save("models/quick_test_ppo")
    print(f"   ✓ Model saved to: models/quick_test_ppo.zip")
    print()
    
    # ==========================================
    # 4. Evaluate on Test Set
    # ==========================================
    print("4. Evaluating on test set...")
    
    # Create test environment (incremental mode for rigor)
    env_test = TradingEnv(
        df_test,
        episode_length=252,
        mode='test',  # Incremental indicators (rigorous)
        reward_type='simple',
        initial_cash=10000
    )
    
    print("   Running evaluation (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env_test,
        n_eval_episodes=20,
        deterministic=True
    )
    
    print()
    print(f"   ✓ Test Results:")
    print(f"     - Mean reward: {mean_reward:.2f}")
    print(f"     - Std reward:  {std_reward:.2f}")
    print()
    
    # ==========================================
    # 5. Save Results
    # ==========================================
    results = {
        'experiment': 'quick_test',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'training': {
            'timesteps': 50_000,
            'time_minutes': train_time / 60
        },
        'evaluation': {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'n_episodes': 20
        },
        'data': {
            'train_rows': len(df_train),
            'test_rows': len(df_test),
            'train_period': f"{train_dates[0]} to {train_dates[-1]}",
            'test_period': f"{test_dates[0]} to {test_dates[-1]}"
        }
    }
    
    results_file = f"experiments/quick_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Results saved to: {results_file}")
    print()
    
    # ==========================================
    # Summary
    # ==========================================
    print("="*70)
    print("EXPERIMENT COMPLETE ✅")
    print("="*70)
    print()
    print("Summary:")
    print(f"  - Training time: {train_time/60:.1f} minutes")
    print(f"  - Test mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  - Model saved: models/quick_test_ppo.zip")
    print(f"  - Results saved: {results_file}")
    print()
    print("Next steps:")
    print("  1. Review results in experiments/ folder")
    print("  2. Check tensorboard logs: tensorboard --logdir tensorboard_quick_test/")
    print("  3. If everything looks good, run full TFM experiments (1M timesteps)")
    print()
    print("Your pipeline is working! Ready for TFM experiments 🚀")
    print()


if __name__ == "__main__":
    main()
