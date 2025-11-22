"""
Chain training using Walk-Forward Validation with Technical Indicators
"""
from stock_options.utils.data import load_data, generate_walk_forward_windows, TECHNICAL_INDICATORS_CONFIG, MIN_HISTORICAL_DATA

def main():
    csv_path = "data/PHIA.csv"
    df = load_data(csv_path)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['date'].min()} - {df['date'].max()}")
    print(f"Available columns: {df.columns}")
    
    # Show technical indicators configuration
    print(f"\n=== Technical Indicators Configuration ===")
    print(f"Moving Averages: Short={TECHNICAL_INDICATORS_CONFIG['ma_short']}, Medium={TECHNICAL_INDICATORS_CONFIG['ma_medium']}, Long={TECHNICAL_INDICATORS_CONFIG['ma_long']} periods")
    print(f"RSI period: {TECHNICAL_INDICATORS_CONFIG['rsi_period']}")
    print(f"Rate of Change period: {TECHNICAL_INDICATORS_CONFIG['roc_period']}")
    print(f"Minimum historical data required: {MIN_HISTORICAL_DATA} days")
    
    # Show sample of technical indicators
    print(f"\n=== Sample Technical Indicators (last 5 rows) ===")
    sample_cols = ["date", "close", "ma_short", "ma_medium", "ma_long", "rsi", "roc", "feature_ma_short", "feature_rsi", "feature_roc"]
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df.select(available_cols).tail(5))

    # Generate all sequential windows using Walk-Forward
    print("\n=== Generating Walk-Forward Windows ===")
    all_windows = generate_walk_forward_windows(df, T_train=65, T_eval=65, T_step=65)
    
    # Convert to list to access by index
    window_list = list(all_windows)
    
    print(f"Total pairs generated: {len(window_list)}")
    
    # Example usage for training phases
    if len(window_list) >= 3:
        print("\n=== Example Usage by Phases ===")
        
        # PHASE 0: Base Comparison
        P1_train, P2_eval = window_list[0]
        print(f"PHASE 0 - Base Comparison:")
        print(f"  Training: {len(P1_train)} rows ({P1_train['date'].min()} - {P1_train['date'].max()})")
        print(f"  Evaluation: {len(P2_eval)} rows ({P2_eval['date'].min()} - {P2_eval['date'].max()})")
        
        # PHASE 1: Hyperparameter Tuning I
        P3_train, P4_eval = window_list[1]
        print(f"PHASE 1 - Hyperparameter Tuning I:")
        print(f"  Training: {len(P3_train)} rows ({P3_train['date'].min()} - {P3_train['date'].max()})")
        print(f"  Evaluation: {len(P4_eval)} rows ({P4_eval['date'].min()} - {P4_eval['date'].max()})")
        
        # PHASE 2: Next iteration
        P5_train, P6_eval = window_list[2]
        print(f"PHASE 2 - Next iteration:")
        print(f"  Training: {len(P5_train)} rows ({P5_train['date'].min()} - {P5_train['date'].max()})")
        print(f"  Evaluation: {len(P6_eval)} rows ({P6_eval['date'].min()} - {P6_eval['date'].max()})")
        
        # Verify that there is no overlap between consecutive windows
        print("\n=== Non-Overlap Verification ===")
        phase0_eval_end_date = P2_eval['date'].max()
        phase1_train_start_date = P3_train['date'].min()
        print(f"Phase 0 evaluation end: {phase0_eval_end_date}")
        print(f"Phase 1 training start: {phase1_train_start_date}")
        print(f"Consecutive and non-overlapping windows: {phase0_eval_end_date < phase1_train_start_date}")
    
    else:
        min_rows_needed = MIN_HISTORICAL_DATA + (65 + 65)  # Historical data + training + evaluation
        print(f"Not enough data to generate multiple windows.")
        print(f"Need at least {min_rows_needed} rows: {MIN_HISTORICAL_DATA} (historical) + 65 (train) + 65 (eval)")
        print(f"Current data has {len(df)} rows")
    
    # Show data requirements analysis
    print(f"\n=== Data Requirements Analysis ===")
    min_for_one_window = MIN_HISTORICAL_DATA + 65 + 65  # Historical + train + eval
    available_after_history = len(df) - MIN_HISTORICAL_DATA
    max_possible_windows = max(0, (available_after_history - 65 - 65) // 65 + 1) if available_after_history >= 130 else 0
    
    print(f"Minimum data for one window: {min_for_one_window} rows")
    print(f"Available data: {len(df)} rows")
    print(f"Data available after historical requirement: {available_after_history} rows")
    print(f"Maximum possible windows with current data: {max_possible_windows}")

if __name__ == "__main__":
    main()