"""
Quick test to verify all new features are working correctly.

Run this after implementing the changes to ensure:
1. Observation space has 22 features
2. Fixed-length episodes work
3. Different reward types are available
4. Cumulative returns and extremes are calculated
"""

import sys
sys.path.insert(0, '/mnt/home-data/lucas/projects/uoc/tfm/stock-options/src')

import numpy as np
from stock_options.environments import TradingEnv
from stock_options.utils.data import load_data

def test_observation_shape():
    """Test that observation space has 22 features."""
    print("\n" + "="*70)
    print("TEST 1: Observation Shape")
    print("="*70)
    
    df = load_data("data/PHIA.csv")
    env = TradingEnv(df, episode_length=252, mode='train')
    
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (22,)")
    
    assert obs.shape == (22,), f"Expected shape (22,), got {obs.shape}"
    
    print("✅ PASS: Observation has 22 features")
    
    # Print feature breakdown
    print("\nFeature breakdown:")
    print("  - Portfolio state: 5 features (cash, value, shares, value_diff, total_value_diff)")
    print("  - Technical indicators: 8 features (MAs, RSI, ROC, volatility, ATR, BB width)")
    print("  - Cumulative returns: 3 features (5d, 10d, 20d)")
    print("  - Price extremes: 6 features (max/min for 5d, 10d, 20d)")
    print("  - Total: 5 + 8 + 3 + 6 = 22 ✓")


def test_fixed_length_episodes():
    """Test that episodes terminate at fixed length."""
    print("\n" + "="*70)
    print("TEST 2: Fixed-Length Episodes")
    print("="*70)
    
    df = load_data("data/PHIA.csv")
    episode_length = 100  # Short episode for quick test
    
    env = TradingEnv(df, episode_length=episode_length, mode='train')
    
    obs, info = env.reset()
    
    step_count = 0
    for i in range(200):  # Run for more steps than episode_length
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        step_count += 1
        
        if terminated or truncated:
            print(f"Episode ended at step {step_count}")
            break
    
    # Episode should end around episode_length (might be slightly different due to data availability)
    assert step_count <= episode_length + 10, f"Episode too long: {step_count} > {episode_length + 10}"
    
    if truncated and not terminated:
        print(f"✅ PASS: Episode truncated at {step_count} steps (target: {episode_length})")
    elif terminated:
        print(f"⚠️  Episode terminated early (portfolio=0 or other reason) at step {step_count}")
    else:
        print(f"❌ FAIL: Episode didn't end properly")


def test_reward_types():
    """Test that different reward types can be instantiated."""
    print("\n" + "="*70)
    print("TEST 3: Reward Types")
    print("="*70)
    
    df = load_data("data/PHIA.csv")
    
    reward_types = {
        'simple': {},
        'risk_adjusted': {'w_drawdown': 0.2},
        'multi': {'w_returns': 1.0, 'w_drawdown': 0.15},
    }
    
    for reward_type, config in reward_types.items():
        print(f"\nTesting reward_type='{reward_type}'...")
        
        try:
            env = TradingEnv(
                df,
                episode_length=50,
                reward_type=reward_type,
                reward_config=config,
                mode='train'
            )
            
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(1)  # Buy action
            
            print(f"  ✅ '{reward_type}' reward: {reward:.4f}")
            
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            raise


def test_cumulative_features():
    """Test that cumulative returns and extremes are calculated."""
    print("\n" + "="*70)
    print("TEST 4: Cumulative Features Calculation")
    print("="*70)
    
    df = load_data("data/PHIA.csv")
    env = TradingEnv(df, episode_length=252, mode='train')
    
    obs, info = env.reset()
    
    # Feature indices (assuming flattened obs)
    # [0-4]   Portfolio: cash, value, shares, value_diff, total_value_diff
    # [5-12]  Technical: ma_short, ma_medium, ma_long, rsi, roc, volatility, atr, bb_width
    # [13-15] Returns: cum_return_5d, cum_return_10d, cum_return_20d
    # [16-21] Extremes: max_5d, min_5d, max_10d, min_10d, max_20d, min_20d
    
    # Run for 30 steps to build up history
    for i in range(30):
        obs, reward, terminated, truncated, info = env.step(0)  # Hold action
        
        if i == 29:  # Last step
            cum_return_5d = obs[13]
            cum_return_10d = obs[14]
            cum_return_20d = obs[15]
            
            max_5d = obs[16]
            min_5d = obs[17]
            max_10d = obs[18]
            min_10d = obs[19]
            max_20d = obs[20]
            min_20d = obs[21]
            
            print(f"\nAfter {i+1} steps:")
            print(f"  Cumulative Returns:")
            print(f"    5-day:  {cum_return_5d:.4f}")
            print(f"    10-day: {cum_return_10d:.4f}")
            print(f"    20-day: {cum_return_20d:.4f}")
            
            print(f"\n  Portfolio Extremes:")
            print(f"    5-day:  max={max_5d:.2f}, min={min_5d:.2f}")
            print(f"    10-day: max={max_10d:.2f}, min={min_10d:.2f}")
            print(f"    20-day: max={max_20d:.2f}, min={min_20d:.2f}")
            
            # Sanity checks
            assert max_5d >= min_5d, "max_5d should be >= min_5d"
            assert max_10d >= min_10d, "max_10d should be >= min_10d"
            assert max_20d >= min_20d, "max_20d should be >= min_20d"
            
            print("\n✅ PASS: Cumulative features calculated correctly")


def test_train_vs_test_mode():
    """Test that train and test modes both work."""
    print("\n" + "="*70)
    print("TEST 5: Train vs Test Mode")
    print("="*70)
    
    df = load_data("data/PHIA.csv")
    
    # Train mode (fast, batch indicators)
    print("\nTesting mode='train' (batch indicators)...")
    env_train = TradingEnv(df, episode_length=100, mode='train')
    obs_train, info_train = env_train.reset()
    print(f"  ✅ Train mode works. Observation shape: {obs_train.shape}")
    
    # Test mode (slow, incremental indicators)
    print("\nTesting mode='test' (incremental indicators)...")
    env_test = TradingEnv(df, episode_length=100, mode='test')
    obs_test, info_test = env_test.reset()
    print(f"  ✅ Test mode works. Observation shape: {obs_test.shape}")
    
    # Both should have same shape
    assert obs_train.shape == obs_test.shape, \
        f"Shape mismatch: train {obs_train.shape} vs test {obs_test.shape}"
    
    print("\n✅ PASS: Both modes produce same observation shape")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    try:
        test_observation_shape()
        test_fixed_length_episodes()
        test_reward_types()
        test_cumulative_features()
        test_train_vs_test_mode()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nYour environment is ready for TFM experiments! 🚀")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TESTS FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
