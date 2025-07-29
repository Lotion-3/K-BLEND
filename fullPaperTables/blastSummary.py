import pandas as pd
import re
import numpy as np
from pathlib import Path

def analyze_blast_final(input_csv_path='blastResults.csv', output_csv_path='blastSummary.csv'):
    """
    Parses BLAST results, correctly calculates and sorts by database sequence count
    (using robust rounding for percentage math), calculates summary statistics,
    and saves the accurate results to a new CSV file.

    This version is compatible with both Windows and WSL (Windows Subsystem for Linux)
    by using the pathlib module for handling file paths.

    Args:
        input_csv_path (str): The path to the source CSV file (will not be modified).
        output_csv_path (str): The path where the new, final summary will be created.
    """
    # --- Step 0: Ensure cross-platform path compatibility ---
    # Convert string paths to Path objects for OS-independent handling
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)

    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: The input file '{input_path}' was not found.")
        return

    # --- Step 1: Standardize experiment names for reliable grouping ---
    def get_standardized_name(filename):
        base_name = re.sub(r'(_\d+|\(\d+\))?\.fasta$', '', str(filename))
        standardized_name = base_name.replace('(', '_').replace(')', '')
        return standardized_name

    # --- Step 2: Implement the CORRECT and ROBUST logic for calculating DB sequences ---
    def calculate_db_sequences(exp_name):
        """
        Calculates the number of DB sequences using robust math.
        """
        numbers = re.findall(r'\d+', exp_name)

        if len(numbers) == 2: # Percentage pattern: e.g., 'train70_11000'
            percentage = int(numbers[0])
            total_sequences = int(numbers[1])
            # !!! CORRECTION: Round the result before converting to an integer !!!
            return int(round(percentage * total_sequences / 100.0))
        elif len(numbers) == 1: # Direct count pattern: e.g., 'train_11'
            return int(numbers[0])
        else:
            return None

    df['experiment'] = df['train_database_file'].apply(get_standardized_name)
    df['num_db_sequences'] = df['experiment'].apply(calculate_db_sequences)

    print("Generated standardized experiment names and applied corrected sequence count logic.")

    # --- Step 3: Group by the corrected columns and calculate statistics ---
    performance_metrics = [
        'db_creation_time_sec', 'blast_prediction_time_sec', 'accuracy_percent',
        'macro_precision', 'macro_recall', 'macro_f1_score',
        'weighted_precision', 'weighted_recall', 'weighted_f1_score'
    ]

    aggregations = {metric: ['mean', 'std', 'var', 'min', 'max'] for metric in performance_metrics}
    aggregations['num_db_sequences'] = 'first'

    summary_df = df.groupby('experiment').agg(aggregations)

    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    summary_df.rename(columns={'num_db_sequences_first': 'num_db_sequences'}, inplace=True)

    # --- Step 4: Calculate Coefficient of Variation (CV) ---
    for metric in performance_metrics:
        mean_col, std_col, cv_col = f'{metric}_mean', f'{metric}_std', f'{metric}_cv'
        summary_df[cv_col] = np.where(summary_df[mean_col] != 0, (summary_df[std_col] / summary_df[mean_col]) * 100, 0)

    # --- Step 5: Finalize and save the corrected and sorted summary ---
    summary_df.reset_index(inplace=True)

    # Sort the final DataFrame by the number of DB sequences
    summary_df = summary_df.sort_values(by='num_db_sequences', ascending=True)

    # Reorder columns for better readability
    info_cols = ['experiment', 'num_db_sequences']
    stat_cols = sorted([col for col in summary_df.columns if col not in info_cols])
    summary_df = summary_df[info_cols + stat_cols]

    try:
        summary_df.to_csv(output_path, index=False)
        print(f"\nFinal analysis complete! The accurate, sorted summary has been saved to '{output_path}'")

        print("\nPreview of the final summary, including the corrected 'train70_11000' row:")
        print(summary_df.to_string())
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    # You can specify different input and output directories here.
    # For example, to read from a 'data' subdirectory and write to a 'results' subdirectory:
    #
    # from pathlib import Path
    # data_dir = Path('data')
    # results_dir = Path('results')
    # results_dir.mkdir(exist_ok=True) # Ensure the results directory exists
    #
    # input_file = data_dir / 'blastResults.csv'
    # output_file = results_dir / 'blastSummary.csv'
    #
    # analyze_blast_final(input_file, output_file)

    # By default, this will run in the current directory.
    analyze_blast_final()