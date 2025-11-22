# Experiments Directory

This directory contains all experimental scripts and results for the TFM.

## Quick Start

### 1. Quick Test (Verify Pipeline Works)

```bash
python experiments/quick_test.py
```

**Purpose**: Verify that the full training/evaluation pipeline works  
**Runtime**: ~10-15 minutes  
**Output**: 
- Model: `models/quick_test_ppo.zip`
- Results: `experiments/quick_test_results_TIMESTAMP.json`
- Tensorboard logs: `tensorboard_quick_test/`

---

## Experiment Structure

### Phase 1: Algorithm Comparison (TFM Main Experiment)

**Goal**: Compare PPO vs A2C vs A3C under identical conditions

**Scripts** (to be created):
- `experiments/phase1_train_ppo.py`
- `experiments/phase1_train_a2c.py`
- `experiments/phase1_compare.py`

**Configuration**:
```python
STANDARD_CONFIG = {
    'episode_length': 252,       # 1 trading year
    'total_timesteps': 1_000_000,  # Same budget for all
    'initial_cash': 10000,
    'reward_type': 'simple',
    'mode': 'train'
}
```

### Phase 2: Reward Ablation Study

**Goal**: Compare different reward functions

**Scripts** (to be created):
- `experiments/phase2_ablation_rewards.py`

**Reward types to test**:
- `simple`: Baseline
- `risk_adjusted`: Penalize drawdowns
- `multi`: Multi-objective (returns + risk)

### Phase 3: Walk-Forward Validation (Optional)

**Goal**: Test robustness across time periods

**Scripts** (to be created):
- `experiments/phase3_walk_forward.py`

---

## Results Format

All experiments save results in JSON format:

```json
{
  "experiment": "quick_test",
  "timestamp": "2025-11-22T21:00:00",
  "config": {
    "episode_length": 252,
    "reward_type": "simple",
    ...
  },
  "training": {
    "timesteps": 50000,
    "time_minutes": 12.5
  },
  "evaluation": {
    "mean_reward": 1234.56,
    "std_reward": 123.45,
    "n_episodes": 20
  }
}
```

---

## Monitoring Training

### Tensorboard

```bash
# View training progress
tensorboard --logdir tensorboard_quick_test/

# Or for specific experiment
tensorboard --logdir ppo_trading_tensorboard/
```

### Check Results

```bash
# List all results
ls -lh experiments/*.json

# View latest result
cat experiments/quick_test_results_*.json | python -m json.tool
```

---

## Directory Structure

```
experiments/
├── README.md                  # This file
├── quick_test.py             # Quick verification experiment
├── quick_test_results_*.json # Results from quick test
│
├── phase1_train_ppo.py       # (To be created)
├── phase1_train_a2c.py       # (To be created)
├── phase1_compare.py         # (To be created)
│
├── phase2_ablation_rewards.py # (To be created)
│
└── phase3_walk_forward.py    # (To be created)
```

---

## Next Steps for TFM

1. ✅ **Run quick_test.py** to verify pipeline works
2. **Create Phase 1 scripts** for algorithm comparison
3. **Run Phase 1** experiments (PPO, A2C, A3C with 1M timesteps each)
4. **Analyze results** and create comparison table for TFM
5. **Optional**: Run Phase 2 (reward ablation) and Phase 3 (walk-forward)

---

## Tips

- Always use `episode_length=252` for fair comparison
- Use `mode='train'` for training (fast)
- Use `mode='test'` for final evaluation (rigorous)
- Save all hyperparameters in results JSON
- Use consistent random seeds for reproducibility
