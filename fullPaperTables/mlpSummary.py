import pandas as pd
import numpy as np
import re
from pathlib import Path  # Import the Path object for cross-platform compatibility

# --- Configuration ---
# Use Path objects to ensure file paths are handled correctly on any OS (Windows, WSL, Linux)
INPUT_FILENAME = Path('mlpResults.csv')
OUTPUT_FILENAME = Path('mlpSummaryResults.csv')

def analyze_experiment_data():
    """
    Reads raw experiment data, calculates statistics across multiple versions of each run,
    and saves the aggregated results to a new CSV file in a cross-platform manner.
    """
    try:
        # 1. Load the dataset from the specified file
        #    pandas works seamlessly with pathlib.Path objects.
        print(f"Reading data from '{INPUT_FILENAME}'...")
        df = pd.read_csv(INPUT_FILENAME)

        # 2. Adjust model IDs for 'individual' runs to be 1-based instead of 0-based
        print("Adjusting model IDs for individual runs...")
        df.loc[df['run_type'] == 'individual', 'model_id_or_k'] += 1

        # 3. Create a consistent 'experiment' key for grouping by removing version numbers
        df['experiment'] = df['train_file'].apply(lambda x: re.sub(r'_\d+\.fasta$', '', x))

        # 4. Define the numeric columns for which to calculate statistics
        metric_cols = [
            'train_extract_time_s', 'test_extract_time_s', 'total_train_time_s',
            'prediction_time_s', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro'
        ]

        # 5. Define the columns that identify a unique experiment run
        id_cols = ['experiment', 'num_train_seqs', 'num_test_seqs', 'run_type', 'model_id_or_k']

        # 6. Group by the experiment identifiers and aggregate to get statistics
        print("Calculating statistics (mean, std, var, min, max)...")
        aggregations = {col: ['mean', 'std', 'var', 'min', 'max'] for col in metric_cols}
        df_agg = df.groupby(id_cols).agg(aggregations).reset_index()

        # 7. --- CORRECTED SECTION: Robustly flatten the multi-level column names ---
        new_cols = []
        for col_name in df_agg.columns.values:
            if isinstance(col_name, tuple) and col_name[1] != '':
                # This is a metric column, e.g., ('accuracy', 'mean')
                new_cols.append(f'{col_name[0]}_{col_name[1]}')
            else:
                # This is an ID column, e.g., ('num_train_seqs', '') or just 'num_train_seqs'
                new_cols.append(col_name[0] if isinstance(col_name, tuple) else col_name)
        df_agg.columns = new_cols
        # --- End of Correction ---

        # 8. Calculate the Coefficient of Variation (CV) for each metric
        print("Calculating Coefficient of Variation (CV) for result stability...")
        for col in metric_cols:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            cv_col = f'{col}_cv'
            
            # Calculate CV = (Standard Deviation / Mean), handling potential division by zero
            df_agg[cv_col] = np.where(df_agg[mean_col] != 0, df_agg[std_col] / df_agg[mean_col], 0)

        # 9. Sort the final dataframe for a clear, logical order
        print("Sorting data...")
        df_final = df_agg.sort_values(by=['num_train_seqs', 'run_type', 'model_id_or_k'])

        # 10. Save the results to the specified new CSV file
        df_final.to_csv(OUTPUT_FILENAME, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"Aggregated and statistical results have been saved to '{OUTPUT_FILENAME}'.")

    except FileNotFoundError:
        print(f"\nFATAL ERROR: The input file '{INPUT_FILENAME}' was not found.")
        print("Please make sure it is in the same directory as this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    analyze_experiment_data()