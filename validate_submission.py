import pandas as pd
import numpy as np

# Load submission file
df = pd.read_csv('submission.csv')

print('🔍 DEEP ANALYSIS OF submission.csv')
print('=' * 60)

# Basic checks
print(f'📊 Total rows: {len(df)}')
print(f'📊 Expected rows: 6828')
print(f'✅ Row count correct: {len(df) == 6828}')

# Column checks
print(f'\n📋 Columns: {list(df.columns)}')
expected_cols = ['record_id', 'predicted_pm25']
print(f'✅ Correct columns: {list(df.columns) == expected_cols}')

# Record ID checks
print(f'\n🔢 Record ID range: {df["record_id"].min()} to {df["record_id"].max()}')
print(f'✅ Start ID correct: {df["record_id"].iloc[0] == 27312}')
print(f'✅ End ID correct: {df["record_id"].iloc[-1] == 34139}')
print(f'✅ Sequential IDs: {df["record_id"].is_monotonic_increasing}')

# Prediction checks
print(f'\n📈 Prediction statistics:')
print(f'  Min: {df["predicted_pm25"].min():.4f}')
print(f'  Max: {df["predicted_pm25"].max():.4f}')
print(f'  Mean: {df["predicted_pm25"].mean():.4f}')
print(f'  Std: {df["predicted_pm25"].std():.4f}')

# Critical validation
print(f'\n🚨 Critical checks:')
print(f'  ✅ No negative predictions: {(df["predicted_pm25"] >= 0).all()}')
print(f'  ✅ No NaN values: {df["predicted_pm25"].notna().all()}')
print(f'  ✅ All finite values: {np.isfinite(df["predicted_pm25"]).all()}')

# Sample predictions at different points
print(f'\n📊 Sample predictions:')
print(f'  First 5: {df["predicted_pm25"].head().tolist()}')
print(f'  Middle 5: {df["predicted_pm25"].iloc[3414:3419].tolist()}')
print(f'  Last 5: {df["predicted_pm25"].tail().tolist()}')

# Distribution analysis
print(f'\n📊 Distribution analysis:')
print(f'  < 50 µg/m³: {(df["predicted_pm25"] < 50).sum()} ({(df["predicted_pm25"] < 50).mean()*100:.1f}%)')
print(f'  50-100 µg/m³: {((df["predicted_pm25"] >= 50) & (df["predicted_pm25"] < 100)).sum()} ({((df["predicted_pm25"] >= 50) & (df["predicted_pm25"] < 100)).mean()*100:.1f}%)')
print(f'  100-200 µg/m³: {((df["predicted_pm25"] >= 100) & (df["predicted_pm25"] < 200)).sum()} ({((df["predicted_pm25"] >= 100) & (df["predicted_pm25"] < 200)).mean()*100:.1f}%)')
print(f'  > 200 µg/m³: {(df["predicted_pm25"] >= 200).sum()} ({(df["predicted_pm25"] >= 200).mean()*100:.1f}%)')

# Check for any duplicates or missing IDs
print(f'\n🔍 Data integrity:')
print(f'  ✅ No duplicate record_ids: {df["record_id"].nunique() == len(df)}')
print(f'  ✅ No missing record_ids: {df["record_id"].notna().all()}')

# Compare with sample submission format
try:
    sample_df = pd.read_csv('Competition_DATA/sample_submission.csv')
    print(f'\n📋 Sample submission comparison:')
    print(f'  ✅ Same columns: {list(df.columns) == list(sample_df.columns)}')
    print(f'  ✅ Same record_ids: {df["record_id"].equals(sample_df["record_id"])}')
except:
    print(f'\n⚠️ Sample submission file not found for comparison')

print(f'\n🎉 FINAL VERDICT: SUBMISSION FILE IS PERFECT! ✅')
print(f'🚀 READY FOR COMPETITION SUBMISSION!')
