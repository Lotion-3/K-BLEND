import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from graphviz import Digraph
from matplotlib.lines import Line2D
import os
import math
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit, fsolve
from matplotlib.patches import Patch
from pathlib import Path # Import the Path object for cross-platform compatibility

# =============================================================================
# === 1. SCRIPT SETUP AND DATA LOADING
# =============================================================================

# This line is for running matplotlib in headless server environments.
# It is generally safe to keep for both Windows and WSL.
os.environ["GSETTINGS_BACKEND"] = "memory"

def setup_plot_style():
    """Sets consistent styling for all plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 150 # Standard DPI for viewing
    plt.rcParams['savefig.dpi'] = 300 # High DPI for publication
    plt.rcParams['font.family'] = 'Arial'

def load_and_clean_data(mlp_path, blast_path):
    """
    Loads and cleans data from specified file paths using pathlib for OS compatibility.
    
    Args:
        mlp_path (Path): Path object for the MLP results CSV.
        blast_path (Path): Path object for the BLAST results CSV.
    """
    print("--- 1. LOADING AND CLEANING DATA ---")
    try:
        df = pd.read_csv(mlp_path, dtype=str)
        print(f"  - Successfully loaded '{mlp_path}'")
    except FileNotFoundError:
        raise SystemExit(f"FATAL ERROR: '{mlp_path}' not found.")
    try:
        blast_df = pd.read_csv(blast_path)
        BLAST_DATA_AVAILABLE = True
        print(f"  - Successfully loaded '{blast_path}'")
    except FileNotFoundError:
        BLAST_DATA_AVAILABLE = False
        print(f"\n  - WARNING: '{blast_path}' not found. Some plots will be incomplete.\n")
        blast_df = pd.DataFrame()

    # Continue with data cleaning as before
    numeric_cols = [
        'num_train_seqs', 'model_id_or_k', 'f1_macro_mean', 'f1_macro_std',
        'f1_macro_min', 'f1_macro_max',
        'prediction_time_s_mean', 'prediction_time_s_std'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    if BLAST_DATA_AVAILABLE:
        blast_df = blast_df.rename(columns={
            'blast_prediction_time_sec_mean': 'prediction_time_s_mean',
            'blast_prediction_time_sec_std': 'prediction_time_s_std',
            'macro_f1_score_mean': 'f1_macro_mean',
            'macro_f1_score_std': 'f1_macro_std',
            'macro_f1_score_min': 'f1_macro_min',
            'macro_f1_score_max': 'f1_macro_max'
        })
        exp_info = df[['experiment', 'num_train_seqs']].drop_duplicates()
        blast_df = pd.merge(blast_df, exp_info, on='experiment', how='left')

    print("  - Data loading and cleaning complete.")
    return df, blast_df, BLAST_DATA_AVAILABLE

# =============================================================================
# === 2. FIGURE GENERATION FUNCTIONS (ARCHIVAL MODE)
# =============================================================================



def generate_figure_2_condensed_facet_plot(mlp_data, blast_data, blast_available, output_dir):
    """Generates the condensed facet plot for performance justification."""
    print("\n>>> GENERATING FIGURE 2: The Justification (Condensed Facet Plot)...")
    exp_info = mlp_data[['experiment', 'num_train_seqs']].drop_duplicates().sort_values('num_train_seqs')
    all_experiments = exp_info['experiment'].tolist()
    low_data_exps = [e for e in all_experiments if exp_info.loc[exp_info['experiment'] == e, 'num_train_seqs'].iloc[0] < 1000]
    high_data_exps = [e for e in all_experiments if exp_info.loc[exp_info['experiment'] == e, 'num_train_seqs'].iloc[0] >= 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), sharey=False)
    fig.suptitle('ensemble Performance vs. BLAST Baseline', fontsize=20, weight='bold')

    colors_low = sns.color_palette("Set1", n_colors=len(low_data_exps))
    colors_high = sns.color_palette("tab10", n_colors=len(high_data_exps))

    print("\n  --- DIAGNOSTICS FOR FIGURE 2 ---")
    print("  [PANEL A: LOW-DATA SCENARIOS]")
    for i, exp in enumerate(low_data_exps):
        num_seqs = exp_info.loc[exp_info['experiment'] == exp, 'num_train_seqs'].iloc[0]
        ensemble_data = mlp_data[(mlp_data['experiment'] == exp) & (mlp_data['run_type'] == 'ensemble')].sort_values('model_id_or_k')
        final_f1 = ensemble_data.loc[ensemble_data['model_id_or_k'] == 25, 'f1_macro_mean'].iloc[0]
        blast_score = blast_data.loc[blast_data['experiment'] == exp, 'f1_macro_mean'].iloc[0]
        print(f"    - EXP: {exp} ({num_seqs} genomes) | Final ensemble F1: {final_f1:.4f} | BLAST F1: {blast_score:.4f}")
        ax1.plot(ensemble_data['model_id_or_k'], ensemble_data['f1_macro_mean'], color=colors_low[i], linestyle='-')
        if blast_available:
            ax1.axhline(y=blast_score, color=colors_low[i], linestyle='--')

    ax1.set_title('(A) Low-Data Scenarios', fontsize=16)
    ax1.set_xlabel('ensemble Size (k)', fontsize=14)
    ax1.set_ylabel('Macro F1-Score', fontsize=14)

    print("\n  [PANEL B: HIGH-DATA SCENARIOS]")
    all_y_values = []
    for i, exp in enumerate(high_data_exps):
        num_seqs = exp_info.loc[exp_info['experiment'] == exp, 'num_train_seqs'].iloc[0]
        ensemble_data = mlp_data[(mlp_data['experiment'] == exp) & (mlp_data['run_type'] == 'ensemble')].sort_values('model_id_or_k')
        final_f1 = ensemble_data.loc[ensemble_data['model_id_or_k'] == 25, 'f1_macro_mean'].iloc[0]
        blast_score = blast_data.loc[blast_data['experiment'] == exp, 'f1_macro_mean'].iloc[0]
        print(f"    - EXP: {exp} ({num_seqs} genomes) | Final ensemble F1: {final_f1:.4f} | BLAST F1: {blast_score:.4f}")
        ax2.plot(ensemble_data['model_id_or_k'], ensemble_data['f1_macro_mean'], color=colors_high[i], linestyle='-')
        all_y_values.extend(ensemble_data['f1_macro_mean'])
        if blast_available:
            ax2.axhline(y=blast_score, color=colors_high[i], linestyle='--')
            all_y_values.append(blast_score)

    if all_y_values:
        ax2.set_ylim(min(all_y_values) - 0.0005, 1.0001)
    ax2.set_title('(B) High-Data Scenarios (Zoomed)', fontsize=16)
    ax2.set_xlabel('ensemble Size (k)', fontsize=14)

    legend_handles = []
    for i, exp in enumerate(low_data_exps):
        num_seqs = exp_info.loc[exp_info['experiment'] == exp, 'num_train_seqs'].iloc[0]
        legend_handles.append(Line2D([0], [0], color=colors_low[i], lw=2, label=f'{num_seqs} genomes (Low)'))
    for i, exp in enumerate(high_data_exps):
        num_seqs = exp_info.loc[exp_info['experiment'] == exp, 'num_train_seqs'].iloc[0]
        legend_handles.append(Line2D([0], [0], color=colors_high[i], lw=2, label=f'{num_seqs} genomes (High)'))
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', lw=2, label='ensemble 356 ensemble'))
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', lw=2, label='BLAST Baseline'))
    fig.legend(handles=legend_handles, loc='center right', title='Legend', bbox_to_anchor=(1.0, 0.5))

    filename = output_dir / 'Figure_2_Condensed_Facet_Plot.png'
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"  - SUCCESS: Figure saved as '{filename}'")

def generate_figure_3_hybrid_crossover_plot(mlp_data, blast_data, blast_available, output_dir):
    """Generates the performance crossover and stability deep-dive plot."""
    print("\n>>> GENERATING FIGURE 3: The Master Plot - Crossover & Stability Deep Dive...")
    ensemble_data = mlp_data[(mlp_data['run_type'] == 'ensemble') & (mlp_data['model_id_or_k'] == 25)].sort_values('num_train_seqs').copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [3, 2]})
    fig.suptitle('Analytical Crossover and High-Data Stability Comparison', fontsize=20, weight='bold')

    def logistic_func(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
    x_ensemble_log = np.log10(ensemble_data['num_train_seqs'])
    y_ensemble = ensemble_data['f1_macro_mean']
    popt_ensemble, _ = curve_fit(logistic_func, x_ensemble_log, y_ensemble, p0=[max(y_ensemble), 1, np.median(x_ensemble_log)], maxfev=5000)
    L_ensemble, k_ensemble, x0_ensemble = popt_ensemble
    popt_blast = np.polyfit(np.log10(blast_data['num_train_seqs']), blast_data['f1_macro_mean'], 1)
    blast_fit_func = np.poly1d(popt_blast)
    m_blast, b_blast = popt_blast
    def difference(log_x): return logistic_func(log_x, *popt_ensemble) - blast_fit_func(log_x)
    log_crossover_x = fsolve(difference, x0=np.log10(500))[0]
    crossover_x = 10**log_crossover_x
    crossover_y = logistic_func(log_crossover_x, *popt_ensemble)

    print("\n  --- DIAGNOSTICS FOR FIGURE 3 ---")
    print("  ensemble 356 Data (k=25) used for fitting:")
    print(ensemble_data[['num_train_seqs', 'f1_macro_mean', 'f1_macro_std', 'f1_macro_min', 'f1_macro_max']].to_string(index=False))
    print("\n  BLAST Data used for fitting:")
    print(blast_data[['num_train_seqs', 'f1_macro_mean', 'f1_macro_std', 'f1_macro_min', 'f1_macro_max']].to_string(index=False))
    print(f"\n  ensemble Logistic Fit Parameters (L, k, x0): {popt_ensemble}")
    print(f"    - EQUATION: F1(N) = {L_ensemble:.4f} / (1 + exp(-{k_ensemble:.4f} * (log10(N) - {x0_ensemble:.4f})))")
    print(f"  BLAST Linear Fit Parameters (slope, intercept): {popt_blast}")
    print(f"    - EQUATION: F1(N) = {m_blast:.4f} * log10(N) + {b_blast:.4f}")
    print(f"\n  *** CALCULATED CROSSOVER POINT: {crossover_x:.2f} genomes ***")

    for _, row in ensemble_data.iterrows():
        x, y, std, y_min, y_max = row['num_train_seqs'], row['f1_macro_mean'], row['f1_macro_std'], row['f1_macro_min'], row['f1_macro_max']
        ax1.plot([x, x], [y_min, y_max], color='gray', linewidth=1.5, zorder=1)
        ax1.plot([x, x], [y - std, y + std], color='deepskyblue', linewidth=6, solid_capstyle='round', zorder=2, alpha=0.7)
    if blast_available:
        for _, row in blast_data.iterrows():
            x, y, std, y_min, y_max = row['num_train_seqs'], row['f1_macro_mean'], row['f1_macro_std'], row['f1_macro_min'], row['f1_macro_max']
            ax1.plot([x, x], [y_min, y_max], color='gray', linewidth=1.5, zorder=1)
            ax1.plot([x, x], [y - std, y + std], color='red', linewidth=6, solid_capstyle='round', zorder=2, alpha=0.7)

    ax1.scatter(ensemble_data['num_train_seqs'], ensemble_data['f1_macro_mean'], color='blue', s=100, ec='white', zorder=10)
    ax1.scatter(blast_data['num_train_seqs'], blast_data['f1_macro_mean'], color='darkred', marker='s', s=100, ec='white', zorder=9)
    x_smooth = np.logspace(np.log10(ensemble_data['num_train_seqs'].min()), np.log10(ensemble_data['num_train_seqs'].max()), 400)
    ax1.plot(x_smooth, logistic_func(np.log10(x_smooth), *popt_ensemble), color='blue', linewidth=3)
    blast_fit_y = blast_fit_func(np.log10(x_smooth))
    ax1.plot(x_smooth, np.minimum(1.0, blast_fit_y), color='darkred', linestyle='--', linewidth=3)
    ax1.axvline(x=crossover_x, color='purple', linestyle=':', linewidth=3)
    ax1.annotate(f'Calculated Crossover\n~{crossover_x:.0f} genomes', xy=(crossover_x, crossover_y), xytext=(crossover_x * 1.5, crossover_y - 0.4), arrowprops=dict(facecolor='purple', shrink=0.05, width=1.5, headwidth=8), fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="lavender", ec="purple", lw=1))

    ax1.set_title('(A) Regression Analysis of Performance Crossover with Stability', fontsize=16)
    ax1.set_xlabel('Number of Genomes in Training Set (Log Scale)', fontsize=14)
    ax1.set_ylabel('Macro F1-Score', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0.2, top=1.01)

    ensemble_data['Method'] = 'ensemble 356 ensemble'
    blast_data['Method'] = 'BLAST'
    combined_df = pd.concat([ensemble_data, blast_data], ignore_index=True)
    combined_df['display_label'] = combined_df['num_train_seqs'].apply(lambda x: f'{x} genomes')
    high_data_df = combined_df[combined_df['num_train_seqs'] >= 1000]

    print("\n  [PANEL B: HIGH-DATA ZOOMED VIEW]")
    print(high_data_df[['display_label', 'Method', 'f1_macro_mean', 'f1_macro_std', 'f1_macro_min', 'f1_macro_max']].to_string(index=False))

    x_labels = high_data_df['display_label'].unique()
    x_pos = np.arange(len(x_labels))
    width = 0.35
    for i, method in enumerate(['ensemble 356 ensemble', 'BLAST']):
        method_df = high_data_df[high_data_df['Method'] == method]
        if not method_df.empty:
            offset = width/2 * (1 if method == 'BLAST' else -1)
            color = 'red' if method == 'BLAST' else 'deepskyblue'
            for j, x_label in enumerate(x_labels):
                row = method_df[method_df['display_label'] == x_label]
                if not row.empty:
                    x = x_pos[j] + offset
                    mean, std, y_min, y_max = row['f1_macro_mean'].iloc[0], row['f1_macro_std'].iloc[0], row['f1_macro_min'].iloc[0], row['f1_macro_max'].iloc[0]
                    ax2.plot([x, x], [y_min, y_max], color='gray', linewidth=1.5, zorder=1)
                    ax2.plot([x, x], [mean - std, mean + std], color=color, linewidth=6, solid_capstyle='round', zorder=2, alpha=0.7)
                    ax2.scatter(x, mean, color=color, s=100, zorder=3, ec='black', lw=1)
    ax2.set_title('(B) High-Data Scenarios (Zoomed)', fontsize=16)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_ylim(0.995, 1.001)

    legend_elements = [ Patch(facecolor='deepskyblue', alpha=0.7, label='ensemble 356 (±1 Std Dev)'), Patch(facecolor='red', alpha=0.7, label='BLAST (±1 Std Dev)'), Line2D([0], [0], color='gray', lw=1.5, label='Min-Max Range'), Line2D([0], [0], color='blue', lw=3, label='ensemble 356 (Logistic Fit)'), Line2D([0], [0], color='darkred', lw=3, linestyle='--', label='BLAST (Linear Fit)') ]
    fig.legend(handles=legend_elements, loc='upper left', fontsize=12, bbox_to_anchor=(0.01, 0.95))

    filename = output_dir / 'Figure_3_Crossover_and_Stability.png'
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"  - SUCCESS: Figure saved as '{filename}'")

def generate_figure_4_speed_linechart(mlp_data, blast_data, blast_available, output_dir):
    """Generates the speed comparison line chart."""
    print("\n>>> GENERATING FIGURE 4: The Speed Chasm...")
    ensemble_data = mlp_data[(mlp_data['run_type'] == 'ensemble') & (mlp_data['model_id_or_k'] == 25)].sort_values('num_train_seqs')

    print("\n  --- DIAGNOSTICS FOR FIGURE 4 ---")
    print("  Data used for Speed Comparison Plot:")
    merged_speed = pd.merge( ensemble_data[['num_train_seqs', 'prediction_time_s_mean']], blast_data[['num_train_seqs', 'prediction_time_s_mean']], on='num_train_seqs', suffixes=('_ensemble', '_blast') )
    print(merged_speed.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(ensemble_data['num_train_seqs'], ensemble_data['prediction_time_s_mean'], marker='o', color='deepskyblue', label='ensemble 356 ensemble')
    if blast_available:
        blast_plot_data = blast_data.sort_values('num_train_seqs')
        ax.plot(blast_plot_data['num_train_seqs'], blast_plot_data['prediction_time_s_mean'], marker='s', linestyle='--', color='red', label='BLAST')
    ax.set_yscale('log')
    ax.set_title('Prediction Speed Comparison', fontsize=18, weight='bold')
    ax.set_xlabel('Number of Genomes in Training Set', fontsize=14)
    ax.set_ylabel('Mean Prediction Time (seconds, log scale)', fontsize=14)
    ax.legend(title='Method', fontsize=12)
    ax.grid(True, which="both", ls="--")

    filename = output_dir / 'Figure_4_Speed_Comparison.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  - SUCCESS: Figure saved as '{filename}'")

def generate_figure_5_time_stability_heatmap(mlp_data, blast_data, blast_available, output_dir):
    """Generates the time stability heatmap."""
    print("\n>>> GENERATING FIGURE 5: The Time Stability Landscape ('Alarm Red' Heatmap)...")
    ensemble_data = mlp_data[mlp_data['run_type'] == 'ensemble'].copy()
    pivot_mlp = ensemble_data.pivot_table(index='num_train_seqs', columns='model_id_or_k', values='prediction_time_s_std')
    if blast_available:
        blast_to_join = blast_data[['num_train_seqs', 'prediction_time_s_std']].rename(columns={'prediction_time_s_std': 'BLAST'}).set_index('num_train_seqs')
        pivot_table_combined = pivot_mlp.join(blast_to_join)
    else:
        pivot_table_combined = pivot_mlp
    pivot_table_combined.sort_index(inplace=True)

    print("\n  --- DIAGNOSTICS FOR FIGURE 5 ---")
    print("  Data Pivot Table for Time Stability Heatmap (Std Dev of Prediction Time):")
    print(pivot_table_combined.to_string())

    fig, ax = plt.subplots(figsize=(20, 12))
    cmap = sns.color_palette("Reds", as_cmap=True)
    min_val = max(pivot_table_combined[pivot_table_combined > 0].min().min(), 1e-3)
    max_val = pivot_table_combined.max().max()
    norm = LogNorm(vmin=min_val, vmax=max_val)
    sns.heatmap( pivot_table_combined, annot=True, fmt=".3f", linewidths=.5, cmap=cmap, norm=norm, ax=ax, cbar_kws={'label': 'Prediction Time Std Dev (s) - Log Scale'} )
    ax.set_title('Time Stability Landscape: ensemble vs. BLAST', fontsize=18, weight='bold')
    ax.set_xlabel('Number of Models in ensemble (k) / Method', fontsize=14)
    ax.set_ylabel('Number of Genomes in Training Set', fontsize=14)

    filename = output_dir / 'Figure_5_Time_Stability_Heatmap.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  - SUCCESS: Figure saved as '{filename}'")

# =============================================================================
# === 3. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Use pathlib to define input and output paths for cross-platform compatibility
    output_dir = Path('figure2-5')
    input_mlp_csv = Path('mlpSummaryResuts.csv')
    input_blast_csv = Path('blastSummary.csv')

    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    print(f"--- Ensuring output directory '{output_dir}' exists ---")

    setup_plot_style()
    mlp_data, blast_data, blast_available = load_and_clean_data(input_mlp_csv, input_blast_csv)

    print("\n--- GLOBAL DATA SUMMARY ---")
    print(f"  - Total MLP experiments loaded: {len(mlp_data['experiment'].unique())}")
    print(f"  - Total BLAST experiments loaded: {len(blast_data['experiment'].unique())}")
    print("---------------------------\n")

    # Pass the output directory Path object to each figure generation function
    generate_figure_1_blueprint(output_dir)
    generate_figure_2_condensed_facet_plot(mlp_data, blast_data, blast_available, output_dir)
    generate_figure_3_hybrid_crossover_plot(mlp_data, blast_data, blast_available, output_dir)
    generate_figure_4_speed_linechart(mlp_data, blast_data, blast_available, output_dir)
    generate_figure_5_time_stability_heatmap(mlp_data, blast_data, blast_available, output_dir)

    print("\n\n--- SCRIPT FINISHED ---")

    print(f"All figures have been generated and saved to the '{output_dir}' directory.")
