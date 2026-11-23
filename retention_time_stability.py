# ══════════════════════════════════════════════════════════════════════════════
# RT STABILITY ANALYSIS & MATCHED VISUALIZATION — statistics & plotting module
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Determine whether retention time (RT) stability (variability) differs
#   across three sample-introduction conditions:
#     - pepmap                  (PepMap Neo Trap Cartridge)
#     - nanoshield_tht021_flip  (NanoShield 1.2 cm x 100 um, 300 A)
#     - direct_injection        (Direct Injection, no trap column)
#
# Statistical approach:
#   1. Collapse the pseudoreplicated pairwise RT_Delta values back down to ONE
#      independent summary statistic per precursor per condition:
#        - SD of the 6 raw RT values -> measures variability/stability.
#      This avoids pseudoreplication (using correlated pairwise differences as
#      if they were independent observations), which would inflate apparent
#      spread and distort p-values.
#   2. Because the same precursors are present in all 3 conditions (paired design),
#      use the Friedman test (non-parametric repeated-measures ANOVA) as the
#      omnibus test on the per-precursor SD distributions. Friedman/Wilcoxon are
#      rank-based and do not require normality, so no normality pre-check is
#      performed — this is doubly appropriate here since SD is a non-negative,
#      right-skewed statistic that is never Gaussian by construction.
#   3. If significant, run post-hoc pairwise Wilcoxon signed-rank tests with
#      Holm correction for multiple comparisons.
#   4. Report effect sizes: Kendall's W (omnibus concordance) and matched-pairs
#      rank-biserial correlation r (pairwise).
#   5. Plot the SAME per-precursor SD values used in the stats (NOT the pairwise
#      RT_Delta values) as violin plots, so the figure matches what the
#      statistics argue. Extreme outlier precursors are shown as individual
#      jittered scatter points rather than stretching the violin/KDE tail.
#
# Note on systematic RT bias:
#   An earlier version of this analysis also tested per-precursor centered mean
#   RT ("bias/shift") against zero via a one-sample Wilcoxon test, framed as a
#   Gaussian-around-zero question. That framing was a leftover from testing
#   pairwise RT_Delta values (pseudoreplicates) and no longer applies once the
#   stats operate on SD only — SD has no natural zero-symmetric reference, and
#   Friedman/Wilcoxon never required normality in the first place. This module
#   now focuses exclusively on RT variability (SD) across conditions.
#
# Color coding:
#   A single source of truth, `condition_config`, maps each internal condition
#   key to (display label, RGB color in 0-1 matplotlib format). ALL plots in
#   this script pull colors from this dictionary — no separate/duplicate
#   color dictionaries are used anywhere, so colors stay consistent across
#   every figure.
#
# Usage from the notebook:
#   import retention_time_stability_stats as rts
#   from retention_time_stability_stats import (
#       summarize_precursor_condition, friedman_and_posthoc,
#       print_stability_ranking, plot_sd_violin, plot_sd_violin_clean, ALPHA
#   )
#   rts.output_dir = output_dir  # point the module's export path at the
#                                 # output_dir already defined in the notebook
# ══════════════════════════════════════════════════════════════════════════════

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# ── font setup: Arial throughout, with graceful fallback ─────────────────────
# Picks the first available font from the preference list so the script does
# not crash on systems without Arial installed.
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
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
})

# ──────────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH FOR CONDITION DISPLAY LABELS + COLORS
# Every plotting function below pulls its palette from this dictionary via the
# helper functions `get_color()` / `get_label()` — never hardcode colors
# elsewhere, so a single edit here updates every figure consistently.
# RGB values are given 0-255 (as commonly reported by color pickers) and
# converted to the 0-1 range matplotlib expects.
# ──────────────────────────────────────────────────────────────────────────────
condition_config = {
    "pepmap":                 ("PepMap Neo Trap Cartridge",          (213 / 255,   94 / 255,   0 / 255)),
    "nanoshield_tht021_flip": ("NanoShield 1.2 cm x 100 um, 300 A",  (  0 / 255,  90 / 255,   65 / 255)),
    "direct_injection":       ("Direct Injection",                   ( 0 / 255, 114 / 255, 178 / 255)),
}

# Plain condition-key -> RGB tuple palette, derived from condition_config.
# This is what gets passed directly to seaborn's `palette=` argument whenever
# plots are keyed by the internal condition name (as opposed to the display label).
CONDITION_COLORS = {key: rgb for key, (label, rgb) in condition_config.items()}


def get_color(condition_key):
    """
    Look up the RGB color (0-1 matplotlib format) for a given internal
    condition key (e.g. 'pepmap'), using `condition_config` as the single
    source of truth.

    Parameters
    ----------
    condition_key : str
        Internal condition identifier, e.g. 'pepmap', 'nanoshield_tht021_flip',
        'direct_injection'.

    Returns
    -------
    tuple of float
        RGB color as a 3-tuple of floats in [0, 1].
    """
    return condition_config[condition_key][1]


def get_label(condition_key):
    """
    Look up the human-readable display label for a given internal condition
    key, using `condition_config` as the single source of truth.

    Parameters
    ----------
    condition_key : str
        Internal condition identifier, e.g. 'pepmap'.

    Returns
    -------
    str
        Display label, e.g. 'PepMap Neo Trap Cartridge'.
    """
    return condition_config[condition_key][0]


# significance threshold used throughout
ALPHA = 0.05

# default export directory for plot_sd_violin / plot_sd_violin_clean; the
# notebook should overwrite this right after import to point at its own
# already-defined output_dir, e.g.:
#   import retention_time_stability_stats as rts
#   rts.output_dir = output_dir
output_dir = Path(".")


def filter_precursors_present_in_all_runs(df, id_col="Precursor.Id"):
    """
    Filter a dataframe down to only those precursors (sequence + charge) that
    were identified in EVERY single run in the dataframe — i.e. present in
    all replicates of all conditions. This is the conservative "globally
    shared" filter, ensuring a fully paired design for downstream statistics
    (no missing values per precursor/condition combination).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least columns `id_col` and 'Run'.
    id_col : str, default 'Precursor.Id'
        Column identifying a unique precursor (sequence + charge state).

    Returns
    -------
    pd.DataFrame
        Subset of `df` containing only precursors present in every unique Run.
    """
    unique_runs = df["Run"].unique()
    total_run_count = len(unique_runs)
    # count how many unique runs each precursor is found in
    precursor_run_counts = df.groupby(id_col)["Run"].nunique()
    # keep only precursors present in every run
    cons_precursors = precursor_run_counts[precursor_run_counts == total_run_count].index
    return df[df[id_col].isin(cons_precursors)].copy()


def summarize_precursor_condition(df_raw, id_col="Precursor.Id"):
    """
    Collapse repeated RT measurements (6 replicates) per precursor per
    condition into one independent summary statistic, avoiding
    pseudoreplication.

    For each (precursor, condition) group computes:
      - SD: standard deviation of RT across replicates -> variability/stability.
      - N: number of replicate RT values used.

    Only the SD is retained: it is the single statistic of interest for this
    analysis (RT variability/stability across conditions). Systematic RT bias
    (shift) is no longer analyzed in this module — see the note at the top of
    this file.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Long-format dataframe with columns `id_col`, 'Condition', and 'RT'.
    id_col : str, default 'Precursor.Id'
        Column identifying a unique precursor (sequence + charge state).

    Returns
    -------
    pd.DataFrame
        One row per (precursor, condition), with columns SD, N.
    """
    g = df_raw.groupby([id_col, "Condition"])["RT"]
    summary = g.agg(SD="std", N="count").reset_index()
    return summary


def friedman_and_posthoc(wide_df, conditions, label):
    """
    Run a Friedman test (non-parametric repeated-measures ANOVA) across the
    given conditions on a paired summary statistic, and if significant,
    follow up with Holm-corrected pairwise Wilcoxon signed-rank tests.

    Friedman/Wilcoxon are rank-based tests and do not require normality, so
    no normality pre-check is performed anywhere in this module.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format dataframe, one column per condition, one row per precursor
        (paired design — same precursor present in every condition).
    conditions : list of str
        Ordered list of condition keys to compare.
    label : str
        Descriptive label used in printed output (should specify flush
        direction and statistic).

    Returns
    -------
    tuple
        (friedman_stat, friedman_p, kendalls_w, posthoc_df_or_None)
        posthoc_df is None if the omnibus test was not significant.
    """
    print(f"\n=== Friedman test: {label} ===")
    data = [wide_df[c].values for c in conditions]
    stat, p = stats.friedmanchisquare(*data)
    n = wide_df.shape[0]
    k = len(conditions)
    kendalls_w = stat / (n * (k - 1))
    print(f"Friedman chi2 = {stat:.4f}, df = {k-1}, p = {p:.4g}")
    print(f"Kendall's W (effect size) = {kendalls_w:.4f}")

    if p < ALPHA:
        print("\n--> Significant global difference detected. Running post-hoc pairwise "
              "Wilcoxon signed-rank tests (Holm-corrected):")
        pairs = list(combinations(conditions, 2))
        raw_pvals = []
        results = []
        for c1, c2 in pairs:
            w_stat, w_p = stats.wilcoxon(wide_df[c1], wide_df[c2])
            diffs = wide_df[c1].values - wide_df[c2].values
            n_eff = np.sum(diffs != 0)
            r_rb = w_stat / (n_eff * (n_eff + 1) / 2) * 2 - 1 if n_eff > 0 else np.nan
            raw_pvals.append(w_p)
            results.append((c1, c2, w_stat, w_p, r_rb))

        # Holm-Bonferroni step-down correction for multiple comparisons
        order = np.argsort(raw_pvals)
        m = len(raw_pvals)
        holm_pvals = np.empty(m)
        for rank, idx in enumerate(order):
            holm_pvals[idx] = min(raw_pvals[idx] * (m - rank), 1.0)
        running_max = 0
        for idx in order:
            running_max = max(running_max, holm_pvals[idx])
            holm_pvals[idx] = running_max

        posthoc_df = pd.DataFrame([
            {
                "Comparison": f"{c1} vs {c2}",
                "Wilcoxon_stat": w_stat,
                "p_raw": w_p,
                "p_holm": holm_p,
                "effect_size_r": r_rb,
                "significant_holm": holm_p < ALPHA,
            }
            for (c1, c2, w_stat, w_p, r_rb), holm_p in zip(results, holm_pvals)
        ])
        print(posthoc_df.to_string(index=False))
        return stat, p, kendalls_w, posthoc_df
    else:
        print("--> No significant global difference between conditions.")
        return stat, p, kendalls_w, None


def print_stability_ranking(wide_sd, conditions, flush_label):
    """
    Print an unambiguous, direction-of-effect report of RT variability (SD)
    across conditions, ranked from most to least stable, with pairwise
    comparisons and per-condition diagnostic percentiles.

    Parameters
    ----------
    wide_sd : pd.DataFrame
        Wide-format per-precursor SD values, one column per condition.
    conditions : list of str
        Ordered list of condition keys.
    flush_label : str
        'forward' or 'reverse' — used to label every line of printed output
        so results from the two datasets are never confused.

    Returns
    -------
    None
        Prints a formatted ranking report.
    """
    print("=" * 70)
    print(f"[{flush_label.upper()}] RT VARIABILITY (SD) — RANKED MOST STABLE -> LEAST STABLE")
    print("=" * 70)

    median_sd = wide_sd.median().sort_values()
    mean_sd = wide_sd.mean().sort_values()

    print(f"\n[{flush_label}] By median SD (minutes):")
    for rank, (cond, val) in enumerate(median_sd.items(), start=1):
        tag = "(most stable)" if rank == 1 else "(least stable)" if rank == len(median_sd) else ""
        print(f"  {rank}. {cond:<28s} {val:.6f} min  {tag}")

    print(f"\n[{flush_label}] By mean SD (minutes):")
    for rank, (cond, val) in enumerate(mean_sd.items(), start=1):
        tag = "(most stable)" if rank == 1 else "(least stable)" if rank == len(mean_sd) else ""
        print(f"  {rank}. {cond:<28s} {val:.6f} min  {tag}")

    print("\n" + "-" * 70)
    print(f"[{flush_label}] PAIRWISE COMPARISONS (based on median SD, median(diff) direction)")
    print("-" * 70)
    for c1, c2 in combinations(conditions, 2):
        diff_median = (wide_sd[c1] - wide_sd[c2]).median()
        if diff_median > 0:
            less_stable, more_stable = c1, c2
        else:
            less_stable, more_stable = c2, c1
        print(f"  {c1} vs {c2}:")
        print(f"      median(SD_{c1} - SD_{c2}) = {diff_median:+.6f} min")
        print(f"      --> {less_stable} is LESS stable than {more_stable}\n")

    print("-" * 70)
    print(f"[{flush_label}] OVERALL RANKING (median SD, most -> least stable):")
    print("  " + "  <  ".join(median_sd.index))
    print("-" * 70)

    print(f"\n[{flush_label}] Additional variability diagnostics per condition:\n")
    for c in conditions:
        vals = wide_sd[c]
        print(f"[{flush_label}] {c}:")
        print(f"  median SD   = {vals.median():.6f} min")
        print(f"  mean SD     = {vals.mean():.6f} min")
        print(f"  90th pct SD = {vals.quantile(0.90):.6f} min")
        print(f"  95th pct SD = {vals.quantile(0.95):.6f} min")
        print(f"  99th pct SD = {vals.quantile(0.99):.6f} min")
        print(f"  max SD      = {vals.max():.6f} min")
        print(f"  % precursors with SD > 0.05 min: {(vals > 0.05).mean()*100:.2f}%\n")


def plot_sd_violin(wide_sd, conditions, flush_label, export=True):
    """
    Plot the de-pseudoreplicated per-precursor SD distributions — i.e. EXACTLY
    the data the Friedman/Wilcoxon tests were run on — as GraphPad-style
    violins with an overlaid jittered strip plot and a dashed median line per
    condition. Colors are taken from `condition_config` (single source of
    truth) via `CONDITION_COLORS`.

    This is deliberately NOT the pairwise RT_Delta data (which is correlated
    across replicates and inflates apparent spread / p-values).

    Parameters
    ----------
    wide_sd : pd.DataFrame
        Wide-format per-precursor SD values, one column per condition.
    conditions : list of str
        Ordered list of condition keys to plot (x-axis order).
    flush_label : str
        'forward' or 'reverse' — used in the plot title and output filename.
    export : bool, default True
        If True, save the figure as both a 600 dpi .png and a vector .svg
        into the globally-defined `output_dir`.

    Returns
    -------
    None
        Displays the figure and (optionally) saves it to disk.
    """
    plot_df = wide_sd[conditions].melt(var_name="Condition", value_name="SD")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=plot_df, x="Condition", y="SD", order=conditions,
        hue="Condition", palette=CONDITION_COLORS, legend=False,
        inner=None, cut=0, linewidth=1.0, ax=ax
    )
    sns.stripplot(
        data=plot_df, x="Condition", y="SD", order=conditions,
        color="black", size=2, alpha=0.3, jitter=0.2, ax=ax
    )
    # dashed median line per condition
    for i, c in enumerate(conditions):
        med = wide_sd[c].median()
        ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=1.5,
                linestyle="--", zorder=5)

    ax.set_ylabel("Per-precursor RT SD (min)")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.4)
    ax.set_xticklabels([])  # external legend used, per house style
    ax.set_title(f"RT stability (per-precursor SD) — {flush_label}")
    sns.despine(ax=ax, top=True, right=True)

    # external legend built from condition_config display labels + colors
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=get_color(c),
                   markersize=10, label=get_label(c))
        for c in conditions
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.tight_layout()

    # ── export: 600 dpi PNG + vector SVG into output_dir ──────────────────────
    # Filename pattern: rt_stability_violin_<flush_label>.<ext>
    # bbox_inches="tight" ensures the external legend is not clipped off.
    if export:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"rt_stability_violin_{flush_label}"
        fig.savefig(output_dir / f"{base_name}.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / f"{base_name}.svg", bbox_inches="tight")

    plt.show()


def plot_sd_violin_clean(wide_sd, conditions, flush_label, clip=None, export=True):
    """
    Minimal violin plot (no overlaid full strip plot, no legend, no x-axis
    labels) of the per-precursor SD values used in the statistical tests,
    with a dashed median line per condition.

    Extreme outlier precursors (SD > `clip`) are excluded from the violin
    body/KDE and instead drawn as individual jittered scatter points, so a
    handful of rare extreme precursors don't stretch the violin tail and
    obscure the bulk of the distribution.

    Colors are taken from `condition_config` (single source of truth) via
    `CONDITION_COLORS`.

    Parameters
    ----------
    wide_sd : pd.DataFrame
        Wide-format per-precursor SD values, one column per condition.
    conditions : list of str
        Ordered list of condition keys to plot (x-axis order).
    flush_label : str
        'forward' or 'reverse' — used as the plot title and output filename.
    clip : float or None, default None
        SD threshold (minutes) above which points are drawn as outlier
        scatter points instead of being included in the violin. If None,
        automatically set to the 99th percentile of the pooled SD values
        across all given conditions.
    export : bool, default True
        If True, save the figure as both a 600 dpi .png and a vector .svg
        into the globally-defined `output_dir`.

    Returns
    -------
    None
        Displays the figure and (optionally) saves it to disk.
    """
    plot_df = wide_sd[conditions].melt(var_name="Condition", value_name="SD")

    if clip is None:
        clip = plot_df["SD"].quantile(0.99)

    mask_in = plot_df["SD"] <= clip
    inlier_df = plot_df[mask_in]
    outlier_df = plot_df[~mask_in]

    x_positions = {c: i for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=(5, 5))

    # violin on inliers only -> no long tail from rare extreme precursors
    sns.violinplot(
        data=inlier_df, x="Condition", y="SD", order=conditions,
        hue="Condition", palette=CONDITION_COLORS, legend=False,
        inner=None, cut=0, linewidth=1.0, ax=ax
    )

    # dashed median line — computed from the FULL (unclipped) data
    for i, c in enumerate(conditions):
        med = wide_sd[c].median()
        ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=1.5,
                linestyle="--", zorder=5)

    # outliers as jittered scatter points, colored per condition
    np.random.seed(42)
    if not outlier_df.empty:
        for c in conditions:
            pts = outlier_df.loc[outlier_df["Condition"] == c, "SD"].values
            if len(pts) == 0:
                continue
            jitter = np.random.uniform(-0.08, 0.08, size=len(pts))
            ax.scatter(
                x_positions[c] + jitter, pts,
                color=get_color(c), edgecolors="black", linewidths=0.5,
                s=25, alpha=0.85, zorder=6,
            )

    ax.set_ylabel("Per-precursor RT SD [min]")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.4)
    ax.set_xticks([])
    ax.set_title(flush_label.capitalize())
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    # ── export: 600 dpi PNG + vector SVG into output_dir ──────────────────────
    # Filename pattern: rt_stability_violin_clean_<flush_label>.<ext>
    if export:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"rt_stability_violin_clean_{flush_label}"
        fig.savefig(output_dir / f"{base_name}.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / f"{base_name}.svg", bbox_inches="tight")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PEPTIDE QUANTITY vs. RETENTION TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Visualize peptide-level quantity (Ms1.Apex.Area) as a function of retention
#   time, for a peptide panel shared across all conditions, aligned onto the
#   direct_injection RT scale.
#
# Approach (mirrors the identified-peptides-vs-RT pipeline):
#   1. Aggregate precursor-level rows to peptide level (Stripped.Sequence),
#      averaging across replicates and charge states, separately per condition.
#   2. Require each peptide to be quantified (non-null, >0 Ms1.Apex.Area) in
#      EVERY replicate of a condition before it is considered "complete" for
#      that condition.
#   3. Intersect the per-condition complete peptide sets across all conditions
#      -> a single shared panel, ensuring a fair, matched comparison.
#   4. RT-calibrate trap conditions onto the direct_injection scale using
#      isotonic regression anchored on peptides shared with direct_injection
#      (same method/function as the identified-peptide analysis).
#   5. Plot log10(Ms1.Apex.Area) vs. RT_Normalized for the shared panel, one
#      color per condition, with an optional smoothed trend line.
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

    If `quant_col` is given, a peptide must additionally have a valid
    (non-null, >0) quantity value in every one of those runs to count as
    "observed" there — this enforces quantitative, not just qualitative,
    completeness.

    Parameters
    ----------
    df : pd.DataFrame
        Precursor-level data with at least [condition_col, run_col, seq_col]
        and, if quant_col is given, the quantity column.
    seq_col : str
        Column holding the stripped peptide sequence.
    condition_col : str
        Column holding the condition label.
    run_col : str
        Column holding the run/file identifier.
    quant_col : str or None
        If provided, only rows with a valid (notna & > 0) value in this
        column count toward "the peptide was seen in this run".

    Returns
    -------
    pd.DataFrame
        Subset of `df` restricted to complete (Condition, peptide) pairs.
        If quant_col is given, rows with invalid quantity are also dropped.
    """
    work = df.copy()
    if quant_col is not None:
        work = work[work[quant_col].notna() & (work[quant_col] > 0)].copy()

    # required number of distinct runs per condition
    req_counts = work.groupby(condition_col)[run_col].nunique()

    # how many runs each (Condition, peptide) pair was actually seen in
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


def get_shared_peptides(peptide_df, conditions, seq_col="Stripped.Sequence", condition_col="Condition"):
    """
    Compute the intersection of peptides present across ALL given conditions.

    Parameters
    ----------
    peptide_df : pd.DataFrame
        Peptide-level table with at least [condition_col, seq_col].
    conditions : list of str
        Condition labels that must all contain a peptide for it to be kept.
    seq_col, condition_col : str
        Column names.

    Returns
    -------
    set
        Stripped sequences present in every condition listed.
    """
    sets_per_condition = [
        set(peptide_df.loc[peptide_df[condition_col] == cond, seq_col])
        for cond in conditions
    ]
    shared = set.intersection(*sets_per_condition)
    return shared


def fit_rt_calibration(peptide_rt_df, trap_condition, reference_condition,
                        condition_col="Condition", seq_col="Stripped.Sequence", rt_col="RT"):
    """
    Fit an isotonic (monotonic) calibration curve mapping a trap condition's
    peptide RTs onto the reference condition's RT scale, using peptides
    shared ("anchors") between the two.

    Parameters
    ----------
    peptide_rt_df : pd.DataFrame
        Peptide-level table with one row per (Condition, peptide), containing
        a representative RT per peptide per condition.
    trap_condition : str
        Condition to be calibrated (e.g. "pepmap").
    reference_condition : str
        Condition used as the RT reference (e.g. "direct_injection").
    condition_col, seq_col, rt_col : str
        Column names.

    Returns
    -------
    iso : sklearn.isotonic.IsotonicRegression
        Fitted calibration model (trap_RT -> reference_RT).
    n_anchors : int
        Number of anchor peptides used to fit the model.
    """
    from sklearn.isotonic import IsotonicRegression

    trap_rt = peptide_rt_df[peptide_rt_df[condition_col] == trap_condition].set_index(seq_col)[rt_col]
    ref_rt = peptide_rt_df[peptide_rt_df[condition_col] == reference_condition].set_index(seq_col)[rt_col]

    shared_peptides = trap_rt.index.intersection(ref_rt.index)
    x_trap = trap_rt.loc[shared_peptides].values
    y_ref = ref_rt.loc[shared_peptides].values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_trap, y_ref)
    return iso, len(shared_peptides)


def build_peptide_quantity_table(
    df,
    conditions,
    reference_condition,
    seq_col="Stripped.Sequence",
    condition_col="Condition",
    run_col="Run",
    rt_col="RT",
    quant_col="Ms1.Apex.Area",
):
    """
    Build the shared, RT-aligned peptide quantity table used for the
    quantity-vs-RT plot.

    Steps
    -----
    1. Filter to peptides quantitatively complete within each condition
       (seen + quantified in every replicate of that condition).
    2. Aggregate to one row per (Condition, peptide): mean RT (raw) and
       mean quantity, averaged across replicates and charge states.
    3. Intersect peptides across all `conditions` -> shared panel.
    4. RT-calibrate every non-reference condition onto `reference_condition`
       using isotonic regression on peptides shared with the reference.

    Parameters
    ----------
    df : pd.DataFrame
        Precursor-level data already restricted to the conditions/flush mode
        of interest, with a `condition_col` column assigned.
    conditions : list of str
        All condition labels to include (e.g. ["pepmap", "nanoshield_tht021_flip",
        "direct_injection"]).
    reference_condition : str
        Condition used as the RT reference (no delay correction applied).
    seq_col, condition_col, run_col, rt_col, quant_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: [condition_col, seq_col, rt_col, quant_col, "RT_Normalized"],
        restricted to the shared peptide panel across all conditions.
    """
    # 1. quantitative completeness filter, per condition
    df_complete = filter_peptides_complete_within_condition(
        df, seq_col=seq_col, condition_col=condition_col, run_col=run_col, quant_col=quant_col
    )

    # 2. aggregate to peptide level: mean RT and mean quantity per (Condition, peptide)
    peptide_quant = (
        df_complete.groupby([condition_col, seq_col])
        .agg(**{rt_col: (rt_col, "mean"), quant_col: (quant_col, "mean")})
        .reset_index()
    )

    # 3. intersect peptides across all conditions -> shared panel
    shared_peptides = get_shared_peptides(peptide_quant, conditions, seq_col=seq_col, condition_col=condition_col)
    peptide_quant = peptide_quant[peptide_quant[seq_col].isin(shared_peptides)].copy()
    print(f"  Shared peptide panel across {conditions}: {len(shared_peptides)} peptides")

    # 4. RT calibration onto the reference scale
    peptide_quant["RT_Normalized"] = peptide_quant[rt_col]  # reference condition stays unchanged
    for cond in conditions:
        if cond == reference_condition:
            continue
        iso_model, n_anchors = fit_rt_calibration(
            peptide_quant, trap_condition=cond, reference_condition=reference_condition,
            condition_col=condition_col, seq_col=seq_col, rt_col=rt_col,
        )
        print(f"  RT calibration '{cond}' -> '{reference_condition}': {n_anchors} anchor peptides used")
        cond_mask = peptide_quant[condition_col] == cond
        peptide_quant.loc[cond_mask, "RT_Normalized"] = iso_model.predict(
            peptide_quant.loc[cond_mask, rt_col].values
        )

    return peptide_quant


def plot_quantity_vs_rt(
    peptide_quant_df,
    conditions,
    quant_col="Ms1.Apex.Area",
    rt_col="RT_Normalized",
    title=None,
    smooth=True,
    smooth_frac=0.15,
):
    """
    Plot log10(quantity) vs. aligned retention time, one color per condition,
    using the shared peptide panel.

    Parameters
    ----------
    peptide_quant_df : pd.DataFrame
        Output of `build_peptide_quantity_table`.
    conditions : list of str
        Condition labels in desired legend/plot order.
    quant_col, rt_col : str
        Column names for quantity and (aligned) retention time.
    title : str or None
        Plot title.
    smooth : bool
        If True, overlay a LOWESS trend line per condition.
    smooth_frac : float
        LOWESS smoothing fraction (passed to statsmodels).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    df = peptide_quant_df.copy()
    df["log10_Quantity"] = np.log10(df[quant_col])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for cond in conditions:
        sub = df[df["Condition"] == cond].sort_values(rt_col)
        ax.scatter(
            sub[rt_col], sub["log10_Quantity"],
            s=12, alpha=0.45, color=get_color(cond), label=None, linewidths=0,
        )
        if smooth:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                sm = lowess(sub["log10_Quantity"], sub[rt_col], frac=smooth_frac, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color=get_color(cond), linewidth=2.2, label=get_label(cond))
            except ImportError:
                ax.plot([], [], color=get_color(cond), linewidth=2.2, label=get_label(cond))
        else:
            ax.plot([], [], color=get_color(cond), linewidth=2.2, label=get_label(cond))

    # GraphPad-like styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    ax.set_xlabel("Normalized Retention Time (min, direct-injection scale)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(r"log$_{10}$(Ms1.Apex.Area)", fontsize=12, fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    return fig



# ══════════════════════════════════════════════════════════════════════════════
# RT-BINNED PEPTIDE QUANTITY SUMMARY (mean ± SEM/SD trend view)
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Provide a simplified, trend-level alternative to the raw scatter of
#   log10(quantity) vs. RT_Normalized, WITHOUT relaxing the strict shared-
#   peptide-panel filtering already enforced upstream. Bins are computed on
#   the same shared/aligned peptide table — binning is purely a summarization
#   step on top of fair data, not a new filtering step.
#
# Approach:
#   1. Take the shared, RT-calibrated peptide table (output of
#      build_peptide_quantity_table) as-is.
#   2. Cut RT_Normalized into equal-width bins spanning the pooled range of
#      RT_Normalized across all conditions.
#   3. Within each (Condition, bin), compute mean log10(quantity), SD, SEM,
#      and n peptides.
#   4. Plot one line per condition (mean ± SEM or SD ribbon) across bin
#      centers.
# ══════════════════════════════════════════════════════════════════════════════


def bin_quantity_by_rt(
    peptide_quant_df,
    conditions,
    n_bins=20,
    quant_col="Ms1.Apex.Area",
    rt_col="RT_Normalized",
    condition_col="Condition",
):
    """
    Summarize log10(quantity) into equal-width RT bins per condition, using
    the already shared/RT-calibrated peptide panel (no additional peptide
    filtering is applied here).

    Parameters
    ----------
    peptide_quant_df : pd.DataFrame
        Output of `build_peptide_quantity_table` (shared peptide panel,
        RT-calibrated).
    conditions : list of str
        Condition labels to summarize.
    n_bins : int
        Number of equal-width RT bins spanning the pooled RT_Normalized range.
    quant_col, rt_col, condition_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: [condition_col, "RT_bin_center", "mean_log10", "sd_log10",
        "sem_log10", "n_peptides"].
    """
    df = peptide_quant_df.copy()
    df["log10_Quantity"] = np.log10(df[quant_col])

    # equal-width bins spanning the pooled RT_Normalized range across all conditions
    rt_min, rt_max = df[rt_col].min(), df[rt_col].max()
    bin_edges = np.linspace(rt_min, rt_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df["RT_bin"] = pd.cut(df[rt_col], bins=bin_edges, include_lowest=True, labels=False)

    summary_rows = []
    for cond in conditions:
        sub = df[df[condition_col] == cond]
        grouped = sub.groupby("RT_bin")["log10_Quantity"]
        means = grouped.mean()
        sds = grouped.std()
        ns = grouped.count()
        sems = sds / np.sqrt(ns)

        for bin_idx in range(n_bins):
            n = ns.get(bin_idx, 0)
            summary_rows.append({
                condition_col: cond,
                "RT_bin_center": bin_centers[bin_idx],
                "mean_log10": means.get(bin_idx, np.nan),
                "sd_log10": sds.get(bin_idx, np.nan),
                "sem_log10": sems.get(bin_idx, np.nan),
                "n_peptides": n,
            })

    return pd.DataFrame(summary_rows)


def plot_quantity_binned(
    binned_df,
    conditions,
    error_type="sem",
    title=None,
):
    """
    Plot mean log10(quantity) per RT bin as one line per condition, with a
    shaded ribbon showing SEM or SD.

    Parameters
    ----------
    binned_df : pd.DataFrame
        Output of `bin_quantity_by_rt`.
    conditions : list of str
        Condition labels in desired legend/plot order.
    error_type : {"sem", "sd"}
        Which spread measure to shade.
    title : str or None
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    err_col = "sem_log10" if error_type == "sem" else "sd_log10"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for cond in conditions:
        sub = binned_df[binned_df["Condition"] == cond].sort_values("RT_bin_center")
        sub = sub.dropna(subset=["mean_log10"])
        x = sub["RT_bin_center"].values
        y = sub["mean_log10"].values
        err = sub[err_col].values

        ax.plot(x, y, color=get_color(cond), linewidth=2.2, label=get_label(cond))
        ax.fill_between(x, y - err, y + err, color=get_color(cond), alpha=0.2, linewidth=0)

    # GraphPad-like styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    ax.set_xlabel("Normalized Retention Time (min, direct-injection scale)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(r"Mean log$_{10}$(Ms1.Apex.Area) $\pm$ " + error_type.upper(), fontsize=12, fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    return fig



