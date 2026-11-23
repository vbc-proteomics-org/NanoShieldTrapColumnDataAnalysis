"""
diann_main_report_filtering.py

Utility functions for diann_main_report_reading

"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from pathlib import Path



# filtering on protein group q value and proteotypic precursors
def filter_protein_group_confidence(
    df: pd.DataFrame,
    protein_group_q_max: float = 0.01,
    proteotypic_value: int = 1
) -> pd.DataFrame:
    """
    Filter a DIA-NN report dataframe by protein-group-level confidence and proteotypicity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    protein_grou_q_max : float, optional
        Maximum allowed PG.Q.Value (default: 0.01)
    proteotypic_value : int, optional
        Required value in Proteotypic column (default: 1)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """

    return (
        df[
            (df["PG.Q.Value"] <= protein_group_q_max) &
            (df["Proteotypic"] == proteotypic_value)
        ]
        .copy()
    )


# filtering on protein q value and proteotypic precursors
def filter_protein_confidence(
    df: pd.DataFrame,
    protein_q_max: float = 0.01,
    proteotypic_value: int = 1
) -> pd.DataFrame:
    """
    Filter a DIA-NN report dataframe by protein-level confidence and proteotypicity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    protein_q_max : float, optional
        Maximum allowed Protein.Q.Value (default: 0.01)
    proteotypic_value : int, optional
        Required value in Proteotypic column (default: 1)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """

    return (
        df[
            (df["Protein.Q.Value"] <= protein_q_max) &
            (df["Proteotypic"] == proteotypic_value)
        ]
        .copy()
    )


# filtering on precursor q value and proteotypic precursors
def filter_precursor_confidence(
    df: pd.DataFrame,
    q_value_max: float = 0.01,
    proteotypic_value: int = 1
) -> pd.DataFrame:
    """
    Filter a DIA-NN report dataframe by precursor-level confidence (Q.Value)
    and proteotypicity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    q_value_max : float, optional
        Maximum allowed Q.Value (default: 0.01)
    proteotypic_value : int, optional
        Required value in Proteotypic column (default: 1)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """

    return (
        df[
            (df["Q.Value"] <= q_value_max) &
            (df["Proteotypic"] == proteotypic_value)
        ]
        .copy()
    )


# plotting retention time distribution as bar charts for each file and exporting bin counts for each corresponding bar chart
def plot_rt_histogram(
    df: pd.DataFrame,
    run_name: str,
    rt_column: str = "RT",
    output_dir: Path | None = None
):
    """
    Plot raw and normalized RT histograms and export corresponding bin counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    run_name : str
        Run identifier (used for plot titles and filenames)
    rt_column : str, optional
        Column containing retention times (default: "RT")
    output_dir : Path or None, optional
        Directory for output files. If None, plots are shown but not saved.
    """

    # extract RTs and drop NaNs
    rt = df[rt_column].dropna().values

    if rt.size == 0:
        raise ValueError(f"No valid RT values found for run: {run_name}")

    # Freedman–Diaconis bin width
    q75, q25 = np.percentile(rt, [75, 25])
    iqr = q75 - q25

    bin_width = 2 * iqr / np.cbrt(rt.size) if iqr > 0 else None
    n_bins = int(np.ceil((rt.max() - rt.min()) / bin_width)) if bin_width else 50
    n_bins = max(n_bins, 10)  # safety floor

    # compute histogram once
    counts, bin_edges = np.histogram(rt, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # normalized counts
    total = counts.sum()
    norm_counts = counts / total if total > 0 else counts

    # build output dataframe
    hist_df = pd.DataFrame({
        "RT_bin_start": bin_edges[:-1],
        "RT_bin_end": bin_edges[1:],
        "RT_bin_center": bin_centers,
        "Count": counts,
        "Normalized.Count": norm_counts
    })

    # sanitize filename
    safe_name = run_name.replace("/", "_")

    # --- raw histogram ---
    plt.figure()
    plt.bar(bin_centers, counts, width=np.diff(bin_edges), align="center")
    plt.xlabel("Retention time [min]")
    plt.ylabel("Count")
    plt.title(run_name)

    if output_dir is not None:
        plt.savefig(output_dir / f"{safe_name}_rt_hist.png", dpi=300)
        hist_df.to_csv(output_dir / f"{safe_name}_rt_hist.csv", index=False)

    plt.close()

    # --- normalized histogram ---
    plt.figure()
    plt.bar(bin_centers, norm_counts, width=np.diff(bin_edges), align="center")
    plt.xlabel("Retention time [min]")
    plt.ylabel("Normalized count")
    plt.title(f"{run_name} (normalized)")

    if output_dir is not None:
        plt.savefig(
            output_dir / f"{safe_name}_rt_hist_normalized.png",
            dpi=300
        )
        hist_df.to_csv(
            output_dir / f"{safe_name}_rt_hist_normalized.csv",
            index=False
        )

    plt.close()


# plotting averaged retention time distribution as bar charts for each file and exporting bin counts for each corresponding bar chart
def plot_average_rt_histogram_by_string(
    run_dfs: dict[str, pd.DataFrame],
    match_string: str,
    rt_column: str = "RT",
    output_dir: Path | None = None
):
    """
    Average RT histograms across runs whose names contain a given string.

    Produces:
      - averaged absolute RT histogram (plot + CSV)
      - averaged normalized RT histogram (plot + CSV)

    Parameters
    ----------
    run_dfs : dict[str, pd.DataFrame]
        Dictionary mapping run names to dataframes
    match_string : str
        Substring used to select runs
    rt_column : str, optional
        Retention time column name (default: "RT")
    output_dir : Path or None, optional
        Directory for output files
    """

    # --- select matching runs ---
    selected = {
        name: df
        for name, df in run_dfs.items()
        if match_string in name
    }

    if len(selected) == 0:
        raise ValueError(f"No runs matched string: '{match_string}'")

    # --- pool RTs to define common bins ---
    pooled_rt = np.concatenate(
        [df[rt_column].dropna().values for df in selected.values()]
    )

    # Freedman–Diaconis binning
    q75, q25 = np.percentile(pooled_rt, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(pooled_rt.size) if iqr > 0 else None
    n_bins = int(np.ceil((pooled_rt.max() - pooled_rt.min()) / bin_width)) if bin_width else 50
    n_bins = max(n_bins, 10)

    # shared bins
    bin_edges = np.histogram_bin_edges(pooled_rt, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- compute per-run histograms ---
    abs_hists = []
    norm_hists = []

    for df in selected.values():
        rt = df[rt_column].dropna().values
        counts, _ = np.histogram(rt, bins=bin_edges)

        abs_hists.append(counts)

        total = counts.sum()
        norm_hists.append(counts / total if total > 0 else counts)

    # --- average ---
    mean_abs = np.mean(abs_hists, axis=0)
    mean_norm = np.mean(norm_hists, axis=0)

    # --- output dataframe ---
    out_df = pd.DataFrame({
        "RT_bin_start": bin_edges[:-1],
        "RT_bin_end": bin_edges[1:],
        "RT_bin_center": bin_centers,
        "Mean.Count": mean_abs,
        "Mean.Normalized.Count": mean_norm
    })

    safe_name = match_string.replace("/", "_")

    # --- absolute plot ---
    plt.figure()
    plt.bar(bin_centers, mean_abs, width=np.diff(bin_edges), align="center")
    plt.xlabel("Retention time [min]")
    plt.ylabel("Mean count")
    plt.title(f"Averaged RT histogram ({match_string})")

    if output_dir is not None:
        plt.savefig(output_dir / f"{safe_name}_avg_rt_hist.png", dpi=300)
        out_df.to_csv(output_dir / f"{safe_name}_avg_rt_hist.csv", index=False)

    plt.close()

    # --- normalized plot ---
    plt.figure()
    plt.bar(bin_centers, mean_norm, width=np.diff(bin_edges), align="center")
    plt.xlabel("Retention time [min]")
    plt.ylabel("Mean normalized count")
    plt.title(f"Averaged RT histogram ({match_string}, normalized)")

    if output_dir is not None:
        plt.savefig(
            output_dir / f"{safe_name}_avg_rt_hist_normalized.png",
            dpi=300
        )
        out_df.to_csv(
            output_dir / f"{safe_name}_avg_rt_hist_normalized.csv",
            index=False
        )

    plt.close()


# plotting histogram for full width half maximum (fwhm) of all identified precursors
def plot_fwhm_histogram_symmetric(df, title, output_dir, column_name="FWHM [second]"):
    """
    Plots a FWHM histogram centered on the median with a symmetric x-axis 
    range that captures at least 90% of the data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Data
    data = df[column_name].dropna().astype(float).values
    if len(data) == 0:
        return

    # 2. Calculate Statistics for the Box and Labels
    median_val = np.median(data)
    mean_val = np.mean(data)
    std_dev = np.std(data)
    
    # 3. Define Symmetric 90% Boundaries
    # We find the 5th and 95th percentiles (capturing 90% of the distribution)
    p05 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    
    # Determine the distance needed to cover both 5% and 95% symmetrically
    dist = max(abs(median_val - p05), abs(median_val - p95))
    
    lower_bound = max(0, median_val - dist) # FWHM cannot be negative
    upper_bound = median_val + dist

    # 4. Setup Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # 5. Create Histogram
    # We use a fixed number of bins (60) within the defined symmetric range
    ax = sns.histplot(
        data, 
        bins=60, 
        binrange=(lower_bound, upper_bound),
        color='midnightblue', 
        edgecolor='white',
        alpha=0.7,
        stat='count',
        kde=True,
        line_kws={'color': 'crimson', 'linewidth': 3}
    )

    # 6. Add Reference Lines
    plt.axvline(median_val, color='red', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}s')
    plt.axvline(mean_val, color='orange', linestyle='--', linewidth=1.2, label=f'Mean: {mean_val:.2f}s')

    # 7. Add Statistics Text Box
    stats_text = (
        f"N: {len(data)}\n"
        f"Median: {median_val:.3f} s\n"
        f"Mean: {mean_val:.3f} s\n"
        f"Std Dev (σ): {std_dev:.3f} s\n"
        f"{'-'*30}\n"
        f"Symmetric Window:\n"
        f"[{lower_bound:.2f}, {upper_bound:.2f}] s\n"
        f"(Captures ≥90% of data)"
    )
    
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 8. Labeling and Formatting
    plt.title(f"{title} - Symmetric FWHM Distribution (90% Coverage)", fontweight='bold', fontsize=14)
    plt.xlabel("FWHM [second]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim(lower_bound, upper_bound)
    
    plt.legend(loc='upper left', frameon=True, facecolor='white')
    plt.tight_layout()

    # 9. Save at 600 DPI
    safe_title = title.lower().replace(" ", "_")
    save_path = os.path.join(output_dir, f"{safe_title}_fwhm_symmetric_90.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()



# function to filter main report loaded in as pandas dataframe
def filter_dataframe(df: pd.DataFrame, 
                     column: str, 
                     cutoff: float, 
                     smaller_than: bool = True, 
                     filter_proteotypic: bool = False) -> pd.DataFrame:
    """
    Filters a DataFrame based on a column threshold and optionally restricts 
    to proteotypic peptides.

    Args:
        df: The pandas DataFrame to filter.
        column: The name of the column to apply the numerical threshold filter to.
        cutoff: The numerical threshold for filtering.
        smaller_than: If True, keeps rows < cutoff. If False, keeps rows > cutoff.
        filter_proteotypic: If True, only keeps rows where the 'Proteotypic' column is 1.

    Returns:
        pd.DataFrame: The filtered and potentially reduced DataFrame.
    """
    # Apply the numerical threshold filter
    if smaller_than:
        df = df[df[column] < cutoff]
    else:
        df = df[df[column] > cutoff]
        
    # Apply the Proteotypic filter if requested
    if filter_proteotypic:
        df = df[df["Proteotypic"] == 1]
        
    return df


# function to plot retention time stability plots
def generate_rt_stability_plots(df, output_dir, condition_filtering, reverse, mbr, quantity_metric):
    """
    Generates and saves RT stability plots for each condition, with 
    separate lines for each unique peptide sequence.
    
    Args:
        df (pd.DataFrame): Dataframe containing RT, RT_seconds, Condition, 
                           Run, and Stripped.Sequence.
        output_dir (str): Directory path to save the exported .png files.
        condiiton_filtering (boolean): Whether condition filtering for reference
                                       dataframe was used for finding list of peptides.
                                       This influences only naming scheme of saved .png files.
        reverse (boolean): Whether trap-and-elute was done in forward (False) or reverse flush (True).
                           This influences only naming scheme of saved .png files.
        mbr (boolean): Whether the DIANN search was performed with MBR or withouth MBR.
                       This influences only naming scheme of saved .png files.
    """
    # Extract replicate number for the X-axis
    df['Replicate'] = df['Run'].str.extract(r'rep_(\d+)').astype(int)
    
    conditions = df['Condition'].unique()
    
    for condition in conditions:
        # Filter for the specific condition and sort by Replicate
        cond_df = df[df['Condition'] == condition].sort_values('Replicate')
        
        # Define the two metrics (minutes and seconds)
        plot_configs = [
            {"col": "RT", "ylabel": "Retention time [min]", "suffix": "min"},
            {"col": "RT_seconds", "ylabel": "Retention time [seconds]", "suffix": "sec"}
        ]
        
        for config in plot_configs:
            plt.figure(figsize=(12, 7))
            
            # Using 'hue' to create a separate line for each peptide sequence
            sns.lineplot(
                data=cond_df, 
                x='Replicate', 
                y=config['col'], 
                hue='Stripped.Sequence', 
                marker='o',
                palette='tab10' # Good for up to 10 distinct colors
            )
            
            plt.title(f"RT Stability per Peptide - {condition} ({config['suffix']})\n mbr_{str(mbr)}_reverse_flush_{str(reverse)}_condition_filtering_{str(condition_filtering)}")
            plt.xlabel("Replicate")
            plt.ylabel(config['ylabel'])
            
            # Move legend outside the plot so it doesn't cover the lines
            plt.legend(title='Peptide Sequence', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.4)
            
            # Save file
            file_name = f"rt_stability_{condition}_{config['suffix']}_mbr_{str(mbr)}_reverse_flush_{str(reverse)}_condition_filtering_{str(condition_filtering)}_{quantity_metric}_.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
            plt.show()




# function to plot retention time stability plots - adjusted tick marks (top 5)
def generate_rt_stability_plots_top_5(df, output_dir, condition_filtering, reverse, mbr, quantity_metric):
    """
    Generates and saves RT stability plots for each condition, with 
    separate lines for each unique peptide sequence.
    
    Args:
        df (pd.DataFrame): Dataframe containing RT, RT_seconds, Condition, 
                           Run, and Stripped.Sequence.
        output_dir (str): Directory path to save the exported .png files.
        condition_filtering (boolean): Whether condition filtering for reference
                                       dataframe was used for finding list of peptides.
                                       This influences only naming scheme of saved .png files.
        reverse (boolean): Whether trap-and-elute was done in forward (False) or reverse flush (True).
                           This influences only naming scheme of saved .png files.
        mbr (boolean): Whether the DIANN search was performed with MBR or withouth MBR.
                       This influences only naming scheme of saved .png files.
    """
    # Extract replicate number for the X-axis
    df['Replicate'] = df['Run'].str.extract(r'rep_(\d+)').astype(int)
    
    conditions = df['Condition'].unique()
    
    for condition in conditions:
        # Filter for the specific condition and sort by Replicate
        cond_df = df[df['Condition'] == condition].sort_values('Replicate')
        
        # Define the two metrics (minutes and seconds)
        plot_configs = [
            {"col": "RT", "ylabel": "Retention time [min]", "suffix": "min"},
            {"col": "RT_seconds", "ylabel": "Retention time [seconds]", "suffix": "sec"}
        ]
        
        for config in plot_configs:
            plt.figure(figsize=(12, 7))
            
            # Using 'hue' to create a separate line for each peptide sequence
            # casting x to string forces discrete categorical axis handling
            sns.lineplot(
                data=cond_df, 
                x=cond_df['Replicate'].astype(str), 
                y=config['col'], 
                hue='Stripped.Sequence', 
                marker='o',
                palette='tab10' # Good for up to 10 distinct colors
            )
            
            plt.title(f"RT Stability per Peptide - {condition} ({config['suffix']})\n mbr_{str(mbr)}_reverse_flush_{str(reverse)}_condition_filtering_{str(condition_filtering)}")
            plt.xlabel("Replicate")
            plt.ylabel(config['ylabel'])
            
            # Move legend outside the plot so it doesn't cover the lines
            plt.legend(title='Peptide Sequence', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.4)
            
            # Save file
            file_name = f"rt_stability_{condition}_{config['suffix']}_mbr_{str(mbr)}_reverse_flush_{str(reverse)}_condition_filtering_{str(condition_filtering)}_{quantity_metric}_.png"
            save_path = os.path.join(output_dir, file_name)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
            plt.show()


# function to calculate all rt deltas - considering all possible pairs of combinations
def calculate_rt_deltas_by_condition(df: pd.DataFrame, absolute_value=True) -> pd.DataFrame:
    """
    Calculates all possible absolute pairwise retention time (RT) differences 
    for identical peptide sequences ('Stripped.Sequence') within each experimental 
    group ('Condition').

    This method systematically breaks down the dataset by Condition and Peptide, 
    gathers all matching RT observations across multiple replicates/precursors, 
    and computes the absolute delta or sign containing rt deltas for every combinatorial pair. 

    Args:
        df (pd.DataFrame): Input dataframe containing at minimum the columns:
                           'Stripped.Sequence', 'Condition', and 'RT'.

    Returns:
        pd.DataFrame: A long-format dataframe containing two columns:
                      - 'Condition': The experimental group.
                      - 'RT_Delta': The calculated absolute difference in minutes.
    """
    delta_records = []

    # Group by Condition first, then nested group by Peptide Sequence
    # observed=True keeps categorical groups efficient if 'Condition' is a category
    for condition, cond_group in df.groupby("Condition", observed=True):
        for peptide, pep_group in cond_group.groupby("Stripped.Sequence"):
            
            # Extract all RT values for this specific peptide in this condition
            rt_values = pep_group["RT"].dropna().tolist()
            
            # Pairwise combinations require at least two records to determine a delta
            if len(rt_values) < 2:
                continue
                
            # Generate all possible absolute differences between pairs, either absolute differences or sign containing
            # itertools.combinations(rt_values, 2) creates pairs without matching a value to itself
            
            if absolute_value:
                for rt_a, rt_b in itertools.combinations(rt_values, 2):
                    abs_delta = abs(rt_a - rt_b)
                    delta_records.append({
                        "Condition": condition,
                        "RT_Delta": abs_delta
                    })
            else:
                for rt_a, rt_b in itertools.combinations(rt_values, 2):
                    delta = rt_a - rt_b
                    delta_records.append({
                        "Condition": condition,
                        "RT_Delta": delta
                    }) 
                
    # Convert the collected list of dicts into a structured DataFrame
    deltas_df = pd.DataFrame(delta_records)
    return deltas_df


# function to calculate all rt deltas - considering only limited pairs with deltas between the first replicate and all other replicates, always to the first as the reference
def calculate_rt_deltas_from_first_replicate(df: pd.DataFrame, absolute_value=True) -> pd.DataFrame:
    """
    Calculates retention time (RT) differences for identical peptide sequences 
    ('Stripped.Sequence') within each 'Condition', always calculating the delta 
    relative to the first encountered replicate/precursor of that peptide.

    If a peptide has 5 replicates, it systematically computes:
    - rt_rep_1 - rt_rep_2
    - rt_rep_1 - rt_rep_3
    - rt_rep_1 - rt_rep_4
    - rt_rep_1 - rt_rep_5

    Args:
        df (pd.DataFrame): Input dataframe containing 'Stripped.Sequence', 'Condition', and 'RT'.
        absolute_value (bool): If True, returns absolute differences. If False, returns 
                               signed differences (rt_first - rt_current).

    Returns:
        pd.DataFrame: A long-format dataframe containing 'Condition' and 'RT_Delta'.
    """
    delta_records = []

    # Group by Condition first, then nested group by Peptide Sequence
    for condition, cond_group in df.groupby("Condition", observed=True):
        for peptide, pep_group in cond_group.groupby("Stripped.Sequence"):
            
            # Extract all RT values for this specific peptide in this condition
            rt_values = pep_group["RT"].dropna().tolist()
            
            # Need at least a first replicate and a second replicate to determine a delta
            if len(rt_values) < 2:
                continue
                
            # Define the reference point (the first replicate)
            rt_first = rt_values[0]

            # Loop through all subsequent replicates starting from the second one (index 1)
            for rt_current in rt_values[1:]:
                delta = rt_first - rt_current
                
                if absolute_value:
                    delta = abs(delta)
                    delta_records.append({
                        "Condition": condition,
                        "RT_Delta": delta
                    })
                else:
                    delta_records.append({
                        "Condition": condition,
                        "RT_Delta": delta
                    })

            
                
    # Convert the collected list of dicts into a structured DataFrame
    deltas_df = pd.DataFrame(delta_records)
    return deltas_df



























