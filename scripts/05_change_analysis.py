"""
05_change_analysis.py
======================
New "change" analysis: comparing popularity of prior works vs destination.

Analyses:
  A. Transition type regressions (9 categories: prior_trend x current_trend)
  B. Continuous popularity_change regressions
  C. Author fixed effects models
  D. Transition heatmap visualization

Input:   data/author_paper_transitions.csv
Output:  results_v2/table10-13_*.csv, results_v2/figure8-9_*.pdf

Usage:   nohup python 05_change_analysis.py > logs_v2/05_change_analysis.log 2>&1 &
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import seaborn as sns

# -- Paths --
DATA_DIR    = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
RESULTS_DIR = Path("/gpfs/kellogg/proj/dashun/abbey/thesis/results_v2")
TRANSITIONS = DATA_DIR / "author_paper_transitions.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ====================================================================
# Data loading
# ====================================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRANSITIONS)
    log.info(f"Loaded {len(df):,} author-paper pairs")
    log.info(f"  Unique authors: {df['author_id'].nunique():,}")
    log.info(f"  Unique papers: {df['work_id'].nunique():,}")
    log.info(f"  Year range: {df['publication_year'].min()}-{df['publication_year'].max()}")

    # Create FE variables
    df["year_fe"]   = df["publication_year"].astype(str)
    df["topic_fe"]  = df["level1_topic"].astype(str)
    df["author_fe"] = df["author_id"].astype(str)

    # Create transition dummies
    # Reference category: stable_to_stable (most common)
    transitions = df["transition_type"].unique()
    log.info(f"\nTransition type distribution:")
    log.info(f"{df['transition_type'].value_counts()}")

    for t in transitions:
        if t != "stable_to_stable" and t != "unknown":
            safe_name = t.replace("_to_", "_TO_")
            df[f"T_{safe_name}"] = (df["transition_type"] == t).astype(int)

    # Prior trend dummies (reference: stable)
    df["prior_ascend"]  = (df["prior_trend_label"] == "ascend").astype(int)
    df["prior_descend"] = (df["prior_trend_label"] == "descend").astype(int)

    # Current trend dummies
    df["D_ascend"]  = (df["trend_label"] == "ascend").astype(int)
    df["D_descend"] = (df["trend_label"] == "descend").astype(int)

    return df


def run_model(formula: str, data: pd.DataFrame, label: str, vcov=None):
    log.info(f"  Running {label}: {formula}")
    try:
        if vcov is None:
            vcov = {"CRV1": "topic_fe"}
        m = pf.feols(formula, data=data, vcov=vcov)
        log.info(f"    N={m._N:,}  R2={m._r2:.4f}")
        return m
    except Exception as e:
        log.error(f"    FAILED: {e}")
        return None


def save_model_table(models: dict, path: Path):
    """Save models to CSV in wide format."""
    wide_rows = []
    for name, m in models.items():
        if m is None:
            continue
        tidy = m.tidy().reset_index()
        if "index" in tidy.columns:
            tidy = tidy.rename(columns={"index": "Coefficient"})
        for _, row in tidy.iterrows():
            wide_rows.append({
                "variable": row["Coefficient"],
                f"{name}_coef": row["Estimate"],
                f"{name}_se": row["Std. Error"],
                f"{name}_pval": row["Pr(>|t|)"],
            })

    wide = pd.DataFrame(wide_rows)
    if not wide.empty:
        wide = wide.groupby("variable").first().reset_index()

    footer = {"variable": "N"}
    footer_r2 = {"variable": "R2"}
    for name, m in models.items():
        if m is None:
            continue
        footer[f"{name}_coef"] = m._N
        footer_r2[f"{name}_coef"] = f"{m._r2:.4f}"

    wide = pd.concat([wide, pd.DataFrame([footer, footer_r2])], ignore_index=True)
    wide.to_csv(path, index=False, float_format="%.6f")
    log.info(f"  Saved {path.name}")


# ====================================================================
# A. Transition type descriptives
# ====================================================================
def transition_descriptives(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Transition type descriptives")
    log.info(f"{'='*60}")

    # Filter out unknown transitions
    valid = df[df["transition_type"] != "unknown"].copy()
    log.info(f"  Valid transitions: {len(valid):,}")

    # Table 10: Hit rate by transition type
    trans_stats = valid.groupby("transition_type").agg(
        N=("hit", "count"),
        mean_pivot=("avg_pivot", "mean"),
        mean_hit=("hit", "mean"),
        mean_rehit=("re_hit", "mean"),
        mean_pop_change=("popularity_change", "mean"),
        mean_current_slope=("topic_log_slope", "mean"),
        mean_prior_slope=("prior_avg_slope", "mean"),
    ).reset_index()

    trans_stats = trans_stats.sort_values("N", ascending=False)
    trans_stats.to_csv(RESULTS_DIR / "table10_transition_stats.csv",
                       index=False, float_format="%.4f")
    log.info("  Saved table10_transition_stats.csv")

    # Figure 8: Transition heatmap (hit rate)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dv, title in zip(axes, ["hit", "re_hit"],
                              ["(a) Raw Hit Rate", "(b) Refined Hit Rate (re_hit)"]):
        # Create 3x3 matrix: prior_trend (rows) x current_trend (cols)
        heatmap_data = valid.groupby(["prior_trend_label", "trend_label"])[dv].mean().unstack()

        # Reorder
        row_order = ["ascend", "stable", "descend"]
        col_order = ["ascend", "stable", "descend"]
        row_order = [r for r in row_order if r in heatmap_data.index]
        col_order = [c for c in col_order if c in heatmap_data.columns]
        heatmap_data = heatmap_data.loc[row_order, col_order]

        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=1, cbar_kws={"label": f"Mean {dv}"})
        ax.set_xlabel("Current Topic Trend")
        ax.set_ylabel("Prior Topic Trend")
        ax.set_title(title)

    fig.suptitle("Hit Rate by Prior and Current Topic Popularity", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure8_transition_heatmap.pdf")
    fig.savefig(RESULTS_DIR / "figure8_transition_heatmap.png")
    plt.close(fig)
    log.info("  Saved figure8")

    # Figure 8b: Transition heatmap (count, log scale)
    fig, ax = plt.subplots(figsize=(7, 5))
    count_data = valid.groupby(["prior_trend_label", "trend_label"]).size().unstack()
    count_data = count_data.loc[
        [r for r in row_order if r in count_data.index],
        [c for c in col_order if c in count_data.columns]
    ]
    sns.heatmap(count_data, annot=True, fmt=",d", cmap="Blues", ax=ax, linewidths=1)
    ax.set_xlabel("Current Topic Trend")
    ax.set_ylabel("Prior Topic Trend")
    ax.set_title("Number of Author-Paper Pairs by Transition Type")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure8b_transition_counts.pdf")
    fig.savefig(RESULTS_DIR / "figure8b_transition_counts.png")
    plt.close(fig)
    log.info("  Saved figure8b")


# ====================================================================
# B. Transition type regressions
# ====================================================================
def transition_regressions(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Transition type regressions")
    log.info(f"{'='*60}")

    valid = df[
        (df["transition_type"] != "unknown")
        & df["avg_pivot"].notna()
        & df["topic_log_slope"].notna()
        & df["num_authors"].notna()
        & df["year_fe"].notna()
        & df["topic_fe"].notna()
    ].copy()
    log.info(f"  Regression sample: {len(valid):,}")

    FE = "year_fe + topic_fe"
    CONTROLS = "num_authors"

    # Get transition dummy columns
    t_cols = [c for c in valid.columns if c.startswith("T_")]
    t_str = " + ".join(t_cols)

    models = {}

    # T1: Transition type dummies (no pivot interaction)
    for dv in ["re_hit", "hit"]:
        dv_valid = valid.dropna(subset=[dv])
        label = f"T1_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + {t_str} + {CONTROLS} | {FE}",
            dv_valid, label
        )

    save_model_table(models, RESULTS_DIR / "table11_transition_models.csv")

    # T2: Transition type with pivot interactions
    models2 = {}
    for dv in ["re_hit", "hit"]:
        dv_valid = valid.dropna(subset=[dv])
        # Interact pivot with key transitions
        key_transitions = [c for c in t_cols
                          if "ascend_TO_descend" in c
                          or "descend_TO_ascend" in c
                          or "stable_TO_ascend" in c
                          or "stable_TO_descend" in c]
        interactions = " + ".join([f"avg_pivot:{c}" for c in key_transitions])

        label = f"T2_{dv}"
        models2[label] = run_model(
            f"{dv} ~ avg_pivot + {t_str} + {interactions} + {CONTROLS} | {FE}",
            dv_valid, label
        )

    save_model_table(models2, RESULTS_DIR / "table11b_transition_interactions.csv")


# ====================================================================
# C. Continuous popularity change regressions
# ====================================================================
def popularity_change_regressions(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Popularity change regressions")
    log.info(f"{'='*60}")

    valid = df[
        df["popularity_change"].notna()
        & df["avg_pivot"].notna()
        & df["prior_avg_slope"].notna()
        & df["num_authors"].notna()
        & df["year_fe"].notna()
        & df["topic_fe"].notna()
    ].copy()
    log.info(f"  Regression sample: {len(valid):,}")

    FE = "year_fe + topic_fe"
    CONTROLS = "num_authors"

    models = {}

    for dv in ["re_hit", "hit"]:
        dv_valid = valid.dropna(subset=[dv])

        # C1: popularity_change + pivot
        label = f"C1_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + popularity_change + {CONTROLS} | {FE}",
            dv_valid, label
        )

        # C2: popularity_change x pivot interaction
        label = f"C2_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + popularity_change + avg_pivot:popularity_change + {CONTROLS} | {FE}",
            dv_valid, label
        )

        # C3: prior_slope + current_slope + change (decomposition)
        label = f"C3_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + prior_avg_slope + topic_log_slope + {CONTROLS} | {FE}",
            dv_valid, label
        )

    save_model_table(models, RESULTS_DIR / "table12_popchange_models.csv")


# ====================================================================
# D. Author fixed effects
# ====================================================================
def author_fe_regressions(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Author fixed effects regressions")
    log.info(f"{'='*60}")

    # Filter to authors with >= 5 papers in the sample for meaningful FE
    author_counts = df.groupby("author_id").size()
    authors_ge5 = set(author_counts[author_counts >= 5].index)
    valid = df[
        df["author_id"].isin(authors_ge5)
        & df["avg_pivot"].notna()
        & df["topic_log_slope"].notna()
        & df["num_authors"].notna()
        & df["year_fe"].notna()
        & df["topic_fe"].notna()
    ].copy()
    log.info(f"  Authors with >= 5 papers: {len(authors_ge5):,}")
    log.info(f"  Regression sample: {len(valid):,}")

    CONTROLS = "num_authors"

    models = {}

    for dv in ["re_hit", "hit"]:
        dv_valid = valid.dropna(subset=[dv])

        # A1: Year + Topic FE (baseline, same as main but on author-paper data)
        label = f"A1_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | year_fe + topic_fe",
            dv_valid, label
        )

        # A2: Year + Topic + Author FE (key specification)
        label = f"A2_{dv}"
        models[label] = run_model(
            f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | year_fe + topic_fe + author_fe",
            dv_valid, label,
            vcov={"CRV1": "author_fe"}  # Cluster by author
        )

        # A3: Author FE + popularity change
        label = f"A3_{dv}"
        pop_valid = dv_valid.dropna(subset=["popularity_change"])
        models[label] = run_model(
            f"{dv} ~ avg_pivot + popularity_change + avg_pivot:popularity_change + {CONTROLS} | year_fe + topic_fe + author_fe",
            pop_valid, label,
            vcov={"CRV1": "author_fe"}
        )

    save_model_table(models, RESULTS_DIR / "table13_author_fe_models.csv")


# ====================================================================
# E. Marginal effects visualization
# ====================================================================
def figure9_change_effects(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Figure 9: Popularity change effects visualization")
    log.info(f"{'='*60}")

    valid = df[
        (df["transition_type"] != "unknown")
        & df["hit"].notna()
        & df["re_hit"].notna()
    ].copy()

    # Bar chart: hit rate by transition type (sorted)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, dv, title in zip(axes, ["hit", "re_hit"],
                              ["(a) Raw Hit Rate", "(b) Refined Hit Rate"]):
        trans_means = valid.groupby("transition_type")[dv].agg(["mean", "count", "sem"]).reset_index()
        trans_means = trans_means[trans_means["count"] >= 100]  # min 100 observations
        trans_means = trans_means.sort_values("mean", ascending=True)

        # Color by transition direction
        colors = []
        for t in trans_means["transition_type"]:
            if "ascend" in t.split("_to_")[1] if "_to_" in t else "":
                colors.append("#e74c3c")
            elif "descend" in t.split("_to_")[1] if "_to_" in t else "":
                colors.append("#3498db")
            else:
                colors.append("#95a5a6")

        bars = ax.barh(range(len(trans_means)), trans_means["mean"],
                       xerr=1.96 * trans_means["sem"], color=colors, edgecolor="white")

        # Format labels nicely
        labels = [t.replace("_to_", " -> ").replace("_", " ").title()
                  for t in trans_means["transition_type"]]
        ax.set_yticks(range(len(trans_means)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f"Mean {dv} Rate")
        ax.set_title(title)

        # Add count annotations
        for i, (_, row) in enumerate(trans_means.iterrows()):
            ax.annotate(f"n={int(row['count']):,}", xy=(row["mean"], i),
                       xytext=(5, 0), textcoords="offset points",
                       fontsize=7, va="center")

    fig.suptitle("Hit Rate by Transition Type (Prior -> Current Topic Trend)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure9_transition_hit_rates.pdf")
    fig.savefig(RESULTS_DIR / "figure9_transition_hit_rates.png")
    plt.close(fig)
    log.info("  Saved figure9")

    # Figure 9b: Scatter of popularity_change vs hit rate (binned)
    valid_pc = valid.dropna(subset=["popularity_change"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, dv, title in zip(axes, ["hit", "re_hit"],
                              ["(a) Raw Hit", "(b) Refined Hit (re_hit)"]):
        n_bins = 30
        valid_pc["pc_bin"] = pd.qcut(valid_pc["popularity_change"], q=n_bins,
                                      labels=False, duplicates="drop")
        binned = valid_pc.groupby("pc_bin").agg(
            x=("popularity_change", "mean"),
            y=(dv, "mean"),
        )
        ax.scatter(binned["x"], binned["y"], s=30, color="#2c3e50", alpha=0.7)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Popularity Change (current - prior)")
        ax.set_ylabel(f"Mean {dv} Rate")
        ax.set_title(title)

    fig.suptitle("Popularity Change vs Hit Rate (Binned Scatter)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure9b_popchange_scatter.pdf")
    fig.savefig(RESULTS_DIR / "figure9b_popchange_scatter.png")
    plt.close(fig)
    log.info("  Saved figure9b")


# ====================================================================
# Main
# ====================================================================
def main():
    df = load_data()

    transition_descriptives(df)
    transition_regressions(df)
    popularity_change_regressions(df)
    author_fe_regressions(df)
    figure9_change_effects(df)

    log.info("\nAll change analysis outputs saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
