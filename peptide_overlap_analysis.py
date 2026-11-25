# ══════════════════════════════════════════════════════════════════════════════
# PEPTIDE OVERLAP ANALYSIS — pepmap / nanoshield / direct_injection
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Compare peptide identification overlap between sample-introduction
#   conditions (pepmap, nanoshield_tht021_flip, direct_injection) at the
#   peptide level (Stripped.Sequence, charge states collapsed):
#     1. Pairwise overlap (pepmap vs direct_injection, nanoshield vs
#        direct_injection) with exported peptide lists for use in
#        notebook-drawn Venn diagrams.
#     2. Replicate-vs-replicate overlap within each condition, to gauge
#        the "baseline" overlap expected from technical/biological
#        replication alone.
#     3. GRAVY score distributions for peptides found exclusively in one
#        condition (three-way "only" regions), exported for external
#        plotting.
#
# Design notes:
#   - Fully standalone: no dependency on other project modules.
#   - GRAVY is computed with Biopython's Bio.SeqUtils.ProtParam.ProteinAnalysis.
#   - Venn diagrams are drawn here (matplotlib_venn) but carry NO text other
#     than the numeric subset counts — titles, legends, and labels are added
#     externally (PowerPoint). Figures are exported as 600 dpi PNG + SVG.
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib_venn import venn2, venn3

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ── font setup: Arial throughout, with graceful fallback ─────────────────────
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_chosen_font = next((f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in _available_fonts), "sans-serif")

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    [_chosen_font],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "axes.linewidth":     1.0,
})

# ──────────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH FOR CONDITION DISPLAY LABELS + COLORS
# ──────────────────────────────────────────────────────────────────────────────
condition_config = {
    "pepmap":                 ("PepMap Neo Trap Cartridge",          (213 / 255,  94 / 255,   0 / 255)),
    "nanoshield_tht021_flip": ("NanoShield 1.2 cm x 100 um, 300 A",  (  0 / 255,  90 / 255,  65 / 255)),
    "direct_injection":       ("Direct Injection",                   (  0 / 255, 114 / 255, 178 / 255)),
}

CONDITION_COLORS = {key: rgb for key, (label, rgb) in condition_config.items()}


def get_color(condition_key):
    """Look up the RGB color (0-1 matplotlib format) for a condition key."""
    return condition_config[condition_key][1]


def get_label(condition_key):
    """Look up the human-readable display label for a condition key."""
    return condition_config[condition_key][0]


# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE COMPLETENESS FILTER (peptide level, charge states collapsed)
# ══════════════════════════════════════════════════════════════════════════════

def filter_peptides_complete_within_condition(
    df,
    seq_col="Stripped.Sequence",
    condition_col="Condition",
    run_col="Run",
    quant_col=None,
):
    """
    Keep only (Condition, Stripped.Sequence) pairs observed in ALL replicate
    runs of their own condition (charge states collapsed).

    A peptide must have a valid (non-null, >0) quantity in `quant_col` in
    every replicate run of its condition to count as "observed" there —
    enforcing quantitative, not just qualitative, completeness.

    Parameters
    ----------
    df : pd.DataFrame
        Precursor-level data with at least [condition_col, run_col, seq_col]
        and, if quant_col is given, the quantity column.
    seq_col, condition_col, run_col : str
        Column names.
    quant_col : str or None
        If provided, only rows with a valid (notna & > 0) value in this
        column count toward "the peptide was seen in this run".

    Returns
    -------
    pd.DataFrame
        Subset of `df` restricted to complete (Condition, peptide) pairs.
    """
    work = df.copy()
    if quant_col is not None:
        work = work[work[quant_col].notna() & (work[quant_col] > 0)].copy()

    req_counts = work.groupby(condition_col)[run_col].nunique()

    seq_counts = (
        work.groupby([condition_col, seq_col])[run_col]
        .nunique()
        .reset_index(name="n_runs")
    )
    seq_counts["required"] = seq_counts[condition_col].map(req_counts)

    complete_pairs = seq_counts[seq_counts["n_runs"] == seq_counts["required"]]
    valid_pairs = set(zip(complete_pairs[condition_col], complete_pairs[seq_col]))

    mask = list(zip(work[condition_col], work[seq_col]))
    keep = [pair in valid_pairs for pair in mask]
    return work[keep].copy()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD PER-CONDITION PEPTIDE SETS
# ══════════════════════════════════════════════════════════════════════════════

def get_condition_peptide_sets(
    df,
    conditions,
    seq_col="Stripped.Sequence",
    condition_col="Condition",
    run_col="Run",
):
    """
    Apply the completeness filter per condition (presence across all
    replicate runs, no quantity requirement) and return the resulting set
    of peptides (stripped sequences) for each condition.

    Parameters
    ----------
    df : pd.DataFrame
        Precursor-level data, already restricted to the flush mode of
        interest, with a `condition_col` column assigned.
    conditions : list of str
        Condition labels to build sets for.
    seq_col, condition_col, run_col : str
        Column names.

    Returns
    -------
    dict of {condition: set of str}
        Peptide set per condition.
    """
    df_complete = filter_peptides_complete_within_condition(
        df, seq_col=seq_col, condition_col=condition_col, run_col=run_col, quant_col=None
    )
    peptide_sets = {}
    for cond in conditions:
        peps = set(df_complete.loc[df_complete[condition_col] == cond, seq_col].unique())
        peptide_sets[cond] = peps
        print(f"  {cond}: {len(peps)} complete peptides")
    return peptide_sets


def export_condition_peptide_lists(peptide_sets, out_dir, flush_label):
    """
    Export the raw peptide list (one .csv per condition) used as input to
    the overlap analysis.

    Parameters
    ----------
    peptide_sets : dict of {condition: set of str}
        Output of `get_condition_peptide_sets`.
    out_dir : Path
        Directory to write the .csv files into.
    flush_label : str
        'forward' or 'reverse' — included in filenames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond, peps in peptide_sets.items():
        fname = out_dir / f"peptide_list_{cond}_{flush_label}.csv"
        pd.DataFrame({"Stripped.Sequence": sorted(peps)}).to_csv(fname, index=False)
        print(f"  Exported {fname.name} ({len(peps)} peptides)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PAIRWISE OVERLAP (sets + export) AND THREE-WAY "ONLY" SETS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pairwise_overlap(peptide_sets, cond_a, cond_b, out_dir, flush_label):
    """
    Compute the 2-way overlap between two conditions and export peptide
    lists (shared + unique-to-each) as plain .csv files.

    Parameters
    ----------
    peptide_sets : dict of {condition: set of str}
        Output of `get_condition_peptide_sets`.
    cond_a, cond_b : str
        The two condition keys to compare.
    out_dir : Path
        Directory to write .csv outputs into.
    flush_label : str
        'forward' or 'reverse'.

    Returns
    -------
    dict of {region_name: set of str}
        f"only_{cond_a}", f"only_{cond_b}", f"shared_{cond_a}_{cond_b}".
    """
    set_a, set_b = peptide_sets[cond_a], peptide_sets[cond_b]
    only_a = set_a - set_b
    only_b = set_b - set_a
    shared = set_a & set_b

    tag = f"{cond_a}_vs_{cond_b}"
    print(f"\n=== Pairwise overlap [{flush_label}]: {cond_a} vs {cond_b} ===")
    print(f"  only {cond_a}: {len(only_a)} | only {cond_b}: {len(only_b)} | shared: {len(shared)}")

    regions = {
        f"only_{cond_a}": only_a,
        f"only_{cond_b}": only_b,
        f"shared_{cond_a}_{cond_b}": shared,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for region_name, peps in regions.items():
        fname = out_dir / f"overlap_{tag}_{region_name}_{flush_label}_peptides.csv"
        pd.DataFrame({"Stripped.Sequence": sorted(peps)}).to_csv(fname, index=False)
        print(f"  {region_name}: {len(peps)} peptides -> {fname.name}")

    return regions


def compute_three_way_only_sets(peptide_sets, cond_a, cond_b, cond_c):
    """
    Compute the three peptide sets exclusive to each single condition
    (i.e. found in that condition and in neither of the other two).

    Parameters
    ----------
    peptide_sets : dict of {condition: set of str}
        Output of `get_condition_peptide_sets`.
    cond_a, cond_b, cond_c : str
        The three condition keys.

    Returns
    -------
    dict of {condition: set of str}
        f"only_{cond_a}", f"only_{cond_b}", f"only_{cond_c}".
    """
    set_a, set_b, set_c = peptide_sets[cond_a], peptide_sets[cond_b], peptide_sets[cond_c]
    only_a = set_a - set_b - set_c
    only_b = set_b - set_a - set_c
    only_c = set_c - set_a - set_b

    print(f"\n=== Three-way exclusive sets: {cond_a}, {cond_b}, {cond_c} ===")
    print(f"  only {cond_a}: {len(only_a)} | only {cond_b}: {len(only_b)} | only {cond_c}: {len(only_c)}")

    return {cond_a: only_a, cond_b: only_b, cond_c: only_c}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: VENN DIAGRAM (numbers only — no titles/labels, design controlled here)
# ══════════════════════════════════════════════════════════════════════════════

def plot_venn2(peptide_sets, cond_a, cond_b, out_dir, flush_label, alpha=1.0):
    """
    Draw and export a 2-set Venn diagram for cond_a vs cond_b. Carries no
    text other than the automatically placed numeric subset counts — no
    set labels, no title (added externally in PowerPoint).

    Parameters
    ----------
    peptide_sets : dict of {condition: set of str}
        Output of `get_condition_peptide_sets`.
    cond_a, cond_b : str
        The two condition keys to compare.
    out_dir : Path
        Directory to write .png/.svg outputs into.
    flush_label : str
        'forward' or 'reverse'.
    alpha : float
        Fill transparency for the circles.

    Returns
    -------
    None
        Displays and exports the figure.
    """
    set_a, set_b = peptide_sets[cond_a], peptide_sets[cond_b]
    tag = f"{cond_a}_vs_{cond_b}"

    fig, ax = plt.subplots(figsize=(5, 5))
    v = venn2(
        subsets=(set_a, set_b),
        set_labels=("", ""),
        set_colors=(get_color(cond_a), get_color(cond_b)),
        alpha=alpha,
        ax=ax,
    )
    for label in (v.get_label_by_id("10"), v.get_label_by_id("01"), v.get_label_by_id("11")):
        if label is not None:
            label.set_fontsize(18)
            label.set_fontweight("bold")
    ax.set_axis_off()
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"venn_{tag}_{flush_label}"
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(f"{base}.svg", bbox_inches="tight", transparent=True)
    plt.show()
    print(f"  Exported {base.name}.png / .svg")


def plot_venn3(peptide_sets, cond_a, cond_b, cond_c, out_dir, flush_label, alpha=1.0):
    """
    Draw and export a 3-set Venn diagram for cond_a vs cond_b vs cond_c.
    Carries no text other than the automatically placed numeric subset
    counts — no set labels, no title (added externally in PowerPoint).

    Parameters
    ----------
    peptide_sets : dict of {condition: set of str}
        Output of `get_condition_peptide_sets`.
    cond_a, cond_b, cond_c : str
        The three condition keys to compare.
    out_dir : Path
        Directory to write .png/.svg outputs into.
    flush_label : str
        'forward' or 'reverse'.
    alpha : float
        Fill transparency for the circles.

    Returns
    -------
    None
        Displays and exports the figure.
    """
    set_a, set_b, set_c = peptide_sets[cond_a], peptide_sets[cond_b], peptide_sets[cond_c]
    tag = f"{cond_a}_vs_{cond_b}_vs_{cond_c}"

    fig, ax = plt.subplots(figsize=(6, 6))
    v = venn3(
        subsets=(set_a, set_b, set_c),
        set_labels=("", "", ""),
        set_colors=(get_color(cond_a), get_color(cond_b), get_color(cond_c)),
        alpha=alpha,
        ax=ax,
    )
    for region_id in ("100", "010", "001", "110", "101", "011", "111"):
        label = v.get_label_by_id(region_id)
        if label is not None:
            label.set_fontsize(18)
            label.set_fontweight("bold")
    ax.set_axis_off()
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"venn_{tag}_{flush_label}"
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(f"{base}.svg", bbox_inches="tight", transparent=True)
    plt.show()
    print(f"  Exported {base.name}.png / .svg")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: REPLICATE-VS-REPLICATE OVERLAP WITHIN A CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def get_replicate_peptide_sets(
    df,
    condition,
    condition_col="Condition",
    run_col="Run",
    seq_col="Stripped.Sequence",
):
    """
    Build one peptide set (charge states collapsed) per replicate run of a
    single condition, using all peptides identified in that run (no
    cross-replicate completeness requirement — this is the raw per-run
    identification set).

    Parameters
    ----------
    df : pd.DataFrame
        Precursor-level data, already restricted to the flush mode of
        interest, with a `condition_col` column assigned.
    condition : str
        Condition key to restrict to.
    condition_col, run_col, seq_col : str
        Column names.

    Returns
    -------
    dict of {run: set of str}
        Peptide set per replicate run.
    """
    sub = df[df[condition_col] == condition].copy()
    sets_by_run = {
        run: set(group[seq_col].unique())
        for run, group in sub.groupby(run_col)
    }
    for run, peps in sets_by_run.items():
        print(f"    {run}: {len(peps)} peptides")
    return sets_by_run


def compute_pairwise_replicate_overlap(sets_by_run, condition):
    """
    Compute the percentage overlap for every ordered pair of replicate runs
    within a condition. The percentage is calculated with each run as the
    reference in turn (|A ∩ B| / |A| and |A ∩ B| / |B| are reported as
    separate rows), so n replicates yield n*(n-1) rows.

    Parameters
    ----------
    sets_by_run : dict of {run: set of str}
        Output of `get_replicate_peptide_sets`.
    condition : str
        Condition key, recorded in the output table.

    Returns
    -------
    pd.DataFrame
        Columns: Condition, Reference_Run, Comparison_Run, N_Reference,
        N_Comparison, N_Shared, Pct_Overlap.
    """
    rows = []
    for ref_run, comp_run in permutations(sets_by_run.keys(), 2):
        set_ref, set_comp = sets_by_run[ref_run], sets_by_run[comp_run]
        n_shared = len(set_ref & set_comp)
        pct = 100.0 * n_shared / len(set_ref) if len(set_ref) > 0 else np.nan
        rows.append({
            "Condition": condition,
            "Reference_Run": ref_run,
            "Comparison_Run": comp_run,
            "N_Reference": len(set_ref),
            "N_Comparison": len(set_comp),
            "N_Shared": n_shared,
            "Pct_Overlap": pct,
        })
    return pd.DataFrame(rows)


def export_replicate_overlap_table(replicate_overlap_df, out_dir, flush_label):
    """
    Export the replicate-vs-replicate overlap table as a .csv for external
    plotting.

    Parameters
    ----------
    replicate_overlap_df : pd.DataFrame
        Output of `compute_pairwise_replicate_overlap` (optionally
        concatenated across conditions).
    out_dir : Path
    flush_label : str

    Returns
    -------
    None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"replicate_overlap_{flush_label}.csv"
    replicate_overlap_df.to_csv(fname, index=False)
    print(f"  Exported {fname.name} ({len(replicate_overlap_df)} rows)")


def plot_replicate_overlap_bar(replicate_overlap_df, out_dir, flush_label):
    """
    Bar plot (mean ± individual points) of replicate-vs-replicate overlap
    percentage per condition, for quick in-notebook inspection.

    Parameters
    ----------
    replicate_overlap_df : pd.DataFrame
        Output of `compute_pairwise_replicate_overlap` (concatenated across
        conditions), must contain Condition and Pct_Overlap.
    out_dir : Path
    flush_label : str

    Returns
    -------
    None
        Displays and exports the figure.
    """
    conditions_present = [c for c in condition_config if c in replicate_overlap_df["Condition"].unique()]
    palette = {c: get_color(c) for c in conditions_present}

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(conditions_present)), 5))
    sns.barplot(
        data=replicate_overlap_df, x="Condition", y="Pct_Overlap",
        order=conditions_present, hue="Condition", hue_order=conditions_present,
        palette=palette, legend=False, errorbar="sd", ax=ax,
    )
    sns.stripplot(
        data=replicate_overlap_df, x="Condition", y="Pct_Overlap",
        order=conditions_present, color="black", size=3, alpha=0.4, jitter=0.2, ax=ax,
    )
    ax.set_xticks(range(len(conditions_present)))
    ax.set_xticklabels([get_label(c) for c in conditions_present], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Replicate-vs-replicate overlap (%)")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"replicate_overlap_{flush_label}"
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    plt.show()
    print(f"  Exported {base.name}.png / .svg")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: GRAVY SCORE FOR CONDITION-EXCLUSIVE PEPTIDES
# ══════════════════════════════════════════════════════════════════════════════

def compute_gravy(seq):
    """
    Compute the GRAVY score for a single stripped peptide sequence using
    Biopython's ProteinAnalysis.

    Parameters
    ----------
    seq : str
        Stripped (unmodified) peptide sequence, standard 20 amino acid
        letters only.

    Returns
    -------
    float
        GRAVY score, or NaN if the sequence cannot be parsed.
    """
    try:
        return ProteinAnalysis(seq).gravy()
    except Exception:
        return np.nan


def build_gravy_table(only_sets, flush_label):
    """
    Compute GRAVY scores for each condition-exclusive peptide set and stack
    them into one long dataframe.

    Parameters
    ----------
    only_sets : dict of {condition: set of str}
        Output of `compute_three_way_only_sets`.
    flush_label : str
        'forward' or 'reverse', recorded in the output table.

    Returns
    -------
    pd.DataFrame
        Columns: Flush, Condition, Stripped.Sequence, GRAVY.
    """
    rows = []
    for cond, peps in only_sets.items():
        for seq in sorted(peps):
            rows.append({
                "Flush": flush_label,
                "Condition": cond,
                "Stripped.Sequence": seq,
                "GRAVY": compute_gravy(seq),
            })
    return pd.DataFrame(rows)


def export_gravy_table(gravy_df, out_dir, flush_label):
    """
    Export the long-format GRAVY table for external (GraphPad) plotting.

    Parameters
    ----------
    gravy_df : pd.DataFrame
        Output of `build_gravy_table`.
    out_dir : Path
    flush_label : str

    Returns
    -------
    None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"gravy_exclusive_peptides_{flush_label}.csv"
    gravy_df.to_csv(fname, index=False)
    print(f"  Exported {fname.name} ({len(gravy_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: QUICK IN-NOTEBOOK GRAVY DISTRIBUTION VIEW
# ══════════════════════════════════════════════════════════════════════════════

def plot_gravy_distribution(gravy_df, out_dir, flush_label):
    """
    Quick violin plot of the GRAVY score distribution for the three
    condition-exclusive peptide sets, for immediate in-notebook inspection.
    Final publication-ready violins are produced externally in GraphPad
    from the exported CSV — this is a sanity-check view only.

    Parameters
    ----------
    gravy_df : pd.DataFrame
        Output of `build_gravy_table`, columns: Flush, Condition,
        Stripped.Sequence, GRAVY.
    out_dir : Path
    flush_label : str

    Returns
    -------
    None
        Displays and exports the figure.
    """
    conditions_present = [c for c in condition_config if c in gravy_df["Condition"].unique()]
    palette = {c: get_color(c) for c in conditions_present}

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(conditions_present)), 5))
    sns.violinplot(
        data=gravy_df, x="Condition", y="GRAVY", order=conditions_present,
        hue="Condition", hue_order=conditions_present,
        palette=palette, legend=False, cut=0, ax=ax,
    )
    sns.stripplot(
        data=gravy_df, x="Condition", y="GRAVY", order=conditions_present,
        color="black", size=2, alpha=0.3, jitter=0.2, ax=ax,
    )
    ax.set_xticks(range(len(conditions_present)))
    ax.set_xticklabels([get_label(c) for c in conditions_present], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("GRAVY")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"gravy_exclusive_peptides_quickview_{flush_label}"
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    plt.show()
    print(f"  Exported {base.name}.png / .svg")


def print_gravy_summary_stats(gravy_df, flush_label):
    """
    Print descriptive statistics (n, mean, SD, median, Q1, Q3, min, max) of
    the GRAVY score per condition to the console.

    Parameters
    ----------
    gravy_df : pd.DataFrame
        Output of `build_gravy_table`, columns: Flush, Condition,
        Stripped.Sequence, GRAVY.
    flush_label : str
        Used only for the printed header.

    Returns
    -------
    pd.DataFrame
        Summary table, one row per condition, columns: Condition, N, Mean,
        SD, Median, Q1, Q3, Min, Max.
    """
    conditions_present = [c for c in condition_config if c in gravy_df["Condition"].unique()]

    rows = []
    for cond in conditions_present:
        vals = gravy_df.loc[gravy_df["Condition"] == cond, "GRAVY"].dropna()
        rows.append({
            "Condition": cond,
            "N": len(vals),
            "Mean": vals.mean(),
            "SD": vals.std(),
            "Median": vals.median(),
            "Q1": vals.quantile(0.25),
            "Q3": vals.quantile(0.75),
            "Min": vals.min(),
            "Max": vals.max(),
        })
    summary_df = pd.DataFrame(rows)

    print(f"\n  GRAVY summary statistics [{flush_label}]:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return summary_df


def run_gravy_pairwise_ttests(gravy_df, flush_label):
    """
    Run pairwise Welch's t-tests (unequal variance) between all condition
    pairs on the GRAVY score, printing n/mean/median/SD per condition and
    t/p per pairwise comparison.

    Parameters
    ----------
    gravy_df : pd.DataFrame
        Output of `build_gravy_table`, columns: Flush, Condition,
        Stripped.Sequence, GRAVY.
    flush_label : str
        Used only for the printed header.

    Returns
    -------
    pd.DataFrame
        One row per pairwise comparison, columns: Flush, Condition_A,
        Condition_B, t_stat, p_value.
    """
    from scipy import stats

    print(f"=== {flush_label.upper()} ===")
    groups = {}
    for cond in [c for c in condition_config if c in gravy_df["Condition"].unique()]:
        vals = gravy_df.loc[gravy_df["Condition"] == cond, "GRAVY"]
        groups[cond] = vals
        print(f"{cond}: n={len(vals)}, mean={vals.mean():.4f}, median={vals.median():.4f}, SD={vals.std():.4f}")

    conds = list(groups.keys())
    rows = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b = groups[conds[i]], groups[conds[j]]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            print(f"  {conds[i]} vs {conds[j]}: t={t:.3f}, p={p:.3e}")
            rows.append({
                "Flush": flush_label,
                "Condition_A": conds[i],
                "Condition_B": conds[j],
                "t_stat": t,
                "p_value": p,
            })
    print()

    return pd.DataFrame(rows)


    