import pandas as pd
import os

# Load results
try:
    full_df = pd.read_csv('models/full_results.csv')
    fast_df = pd.read_csv('models/fast_results.csv')

    # Add a 'Source' column to distinguish
    full_df['Training_Mode'] = 'Full Training (High Accuracy)'
    fast_df['Training_Mode'] = 'Fast Training (Low Latency)'

    # Combine
    combined_df = pd.concat([full_df, fast_df], ignore_index=True)

    # Sort by R2 Score
    combined_df = combined_df.sort_values('R2_Score', ascending=False)

    # Save
    combined_df.to_csv('models/final_comparison.csv', index=False)
    # Also overwrite the model_comparison.csv used by the app generally, so it shows the best default
    combined_df.to_csv('models/model_comparison.csv', index=False)
    
    print("Sucessfully merged results into models/final_comparison.csv")
    print(combined_df)

except Exception as e:
    print(f"Error merging: {e}")
