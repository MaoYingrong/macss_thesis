"""
02_regression_v2.py
====================
Phase 2 (v2): Regression analysis with updated controls.

Changes from v1:
  - Controls: only num_authors (dropped num_references and num_funds)
  - Reads from analysis_sample_v2.csv
  - Outputs to results_v2/
  - Also re-runs matching (CEM) and descriptive tables

Usage:   nohup python 02_regression_v2.py > logs_v2/02_regression_v2.log 2>&1 &
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
from scipy import stats
import seaborn as sns

# -- Paths --
DATA_DIR    = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
RESULTS_DIR = Path("/gpfs/kellogg/proj/dashun/abbey/thesis/results_v2")
SAMPLE      = DATA_DIR / "analysis_sample_v2.csv"

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
COLORS = {"ascend": "#e74c3c", "stable": "#95a5a6", "descend": "#3498db"}


# ====================================================================
# Data loading
# ====================================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE)
    log.info(f"Loaded {len(df):,} papers")

    df["D_ascend"]  = (df["trend_label"] == "ascend").astype(int)
    df["D_descend"] = (df["trend_label"] == "descend").astype(int)

    for q in range(2, 6):
        df[f"pivot_Q{q}"] = (df["pivot_quintile"] == q).astype(int)

    df["year_fe"]  = df["publication_year"].astype(str)
    df["topic_fe"] = df["level1_topic"].astype(str)

    return df


def regression_sample(df: pd.DataFrame, extra_cols: list = None) -> pd.DataFrame:
    cols = ["avg_pivot", "topic_log_slope", "num_authors", "year_fe", "topic_fe"]
    if extra_cols:
        cols += extra_cols
    existing = [c for c in cols if c in df.columns]
    sub = df.dropna(subset=existing).copy()
    log.info(f"  Regression sample: {len(sub):,} papers")
    return sub


# ====================================================================
# Helpers
# ====================================================================
def run_model(formula: str, data: pd.DataFrame, label: str):
    log.info(f"  Running {label}: {formula}")
    try:
        m = pf.feols(formula, data=data, vcov={"CRV1": "topic_fe"})
        log.info(f"    N={m._N:,}  R2={m._r2:.4f}")
        return m
    except Exception as e:
        log.error(f"    FAILED: {e}")
        return None


def save_table(models: dict, path: Path):
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
# KEY CHANGE: Controls now only num_authors
# ====================================================================
CONTROLS = "num_authors"
FE       = "year_fe + topic_fe"


def run_main_models(df: pd.DataFrame, dv: str) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"Running M1-M4 with DV = {dv}")
    log.info(f"{'='*60}")

    data = regression_sample(df, extra_cols=[dv])
    models = {}

    models["M1"] = run_model(f"{dv} ~ avg_pivot | {FE}", data, "M1")
    models["M2"] = run_model(
        f"{dv} ~ avg_pivot + topic_log_slope + {CONTROLS} | {FE}", data, "M2")
    models["M3"] = run_model(
        f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
        data, "M3")
    models["M4"] = run_model(
        f"{dv} ~ avg_pivot + D_ascend + D_descend + avg_pivot:D_ascend + avg_pivot:D_descend + {CONTROLS} | {FE}",
        data, "M4")

    return models


def run_continuous_models(df: pd.DataFrame) -> dict:
    log.info(f"\n{'='*60}")
    log.info("Running M5-M6 with DV = log_citations")
    log.info(f"{'='*60}")

    data = regression_sample(df, extra_cols=["log_citations"])
    models = {}

    models["M5"] = run_model(
        f"log_citations ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
        data, "M5")
    models["M6"] = run_model(
        f"log_citations ~ avg_pivot + D_ascend + D_descend + avg_pivot:D_ascend + avg_pivot:D_descend + {CONTROLS} | {FE}",
        data, "M6")

    return models


def run_nonlinear_models(df: pd.DataFrame) -> dict:
    log.info(f"\n{'='*60}")
    log.info("Running M7-M8 (nonlinearity checks) with DV = re_hit")
    log.info(f"{'='*60}")

    data = regression_sample(df, extra_cols=["re_hit"])
    models = {}

    models["M7"] = run_model(
        f"re_hit ~ avg_pivot + pivot_sq + topic_log_slope "
        f"+ avg_pivot:topic_log_slope + pivot_sq:topic_log_slope "
        f"+ {CONTROLS} | {FE}",
        data, "M7")

    models["M8"] = run_model(
        f"re_hit ~ pivot_Q2 + pivot_Q3 + pivot_Q4 + pivot_Q5 + topic_log_slope "
        f"+ pivot_Q2:topic_log_slope + pivot_Q3:topic_log_slope "
        f"+ pivot_Q4:topic_log_slope + pivot_Q5:topic_log_slope "
        f"+ {CONTROLS} | {FE}",
        data, "M8")

    return models


# ====================================================================
# Figure 7: Predicted re_hit by quintile x temperature
# ====================================================================
def figure7(df: pd.DataFrame, m8):
    if m8 is None:
        log.warning("  Skipping figure7: M8 model not available")
        return

    log.info("Figure 7: Predicted re_hit by pivot quintile x temperature")

    data = regression_sample(df, extra_cols=["re_hit"])

    slope_vals = {
        "Low temp (P10)":  data["topic_log_slope"].quantile(0.10),
        "Med temp (P50)":  data["topic_log_slope"].quantile(0.50),
        "High temp (P90)": data["topic_log_slope"].quantile(0.90),
    }

    base_rate = data["re_hit"].mean()
    coefs = m8.coef()

    quintiles = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#95a5a6", "#e74c3c"]

    for (temp_label, slope_val), color in zip(slope_vals.items(), colors):
        preds = []
        for q in quintiles:
            pred = base_rate
            if q >= 2:
                key = f"pivot_Q{q}"
                if key in coefs:
                    pred += coefs[key]
            if "topic_log_slope" in coefs:
                pred += coefs["topic_log_slope"] * slope_val
            if q >= 2:
                int_key = f"pivot_Q{q}:topic_log_slope"
                if int_key in coefs:
                    pred += coefs[int_key] * slope_val
            preds.append(pred)

        ax.plot(quintiles, preds, marker="o", color=color,
                label=f"{temp_label} ({slope_val:.3f})", linewidth=2)

    ax.set_xlabel("Pivot Quintile (1 = lowest, 5 = highest)")
    ax.set_ylabel("Predicted re_hit Probability")
    ax.set_title("Predicted Hit Probability by Pivot Level and Topic Temperature")
    ax.set_xticks(quintiles)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure7_predicted_rehit.pdf")
    fig.savefig(RESULTS_DIR / "figure7_predicted_rehit.png")
    plt.close(fig)
    log.info("  Saved figure7")


# ====================================================================
# Descriptive tables (re-run for new sample)
# ====================================================================
def run_descriptives(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Running descriptive tables and figures")
    log.info(f"{'='*60}")

    # Table 1: Summary stats
    vars_ = [
        "avg_pivot", "topic_log_slope", "cited_by_count", "log_citations",
        "hit", "re_hit", "CINF", "num_authors", "num_references", "num_funds",
    ]
    existing = [v for v in vars_ if v in df.columns]
    full = df[existing].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    full.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
    full.to_csv(RESULTS_DIR / "table1_summary_full.csv", float_format="%.4f")
    log.info("  Saved table1")

    # Table 2: Correlations
    corr_vars = ["avg_pivot", "topic_log_slope", "log_citations", "hit", "re_hit"]
    existing = [v for v in corr_vars if v in df.columns]
    corr = df[existing].corr()
    corr.to_csv(RESULTS_DIR / "table2_correlation.csv", float_format="%.4f")
    log.info("  Saved table2")

    # Table 3: Pivot by trend group
    valid = df.dropna(subset=["trend_label", "avg_pivot"])
    rows = []
    for label in ["ascend", "stable", "descend"]:
        sub = valid.loc[valid["trend_label"] == label, "avg_pivot"]
        rows.append({
            "trend_group": label, "N": len(sub),
            "mean_pivot": sub.mean(), "sd_pivot": sub.std(), "median_pivot": sub.median(),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "table3_pivot_by_trend.csv", index=False, float_format="%.4f")
    log.info("  Saved table3")

    # Figure 1: Papers per year
    counts = df["publication_year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color="#2c3e50", width=0.8)
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Biology Papers in Analysis Sample by Year")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(RESULTS_DIR / "figure1_papers_per_year.pdf")
    fig.savefig(RESULTS_DIR / "figure1_papers_per_year.png")
    plt.close(fig)
    log.info("  Saved figure1")

    # Figure 3: Pivot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(df["avg_pivot"].dropna(), bins=80, color="#2c3e50",
                 edgecolor="white", linewidth=0.3, density=True)
    axes[0].set_xlabel("Average Pivot Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) Full Sample")
    valid = df.dropna(subset=["trend_label"])
    for label in ["ascend", "stable", "descend"]:
        sub = valid.loc[valid["trend_label"] == label, "avg_pivot"]
        axes[1].hist(sub, bins=80, alpha=0.45, color=COLORS[label],
                     label=label.capitalize(), density=True)
    axes[1].set_xlabel("Average Pivot Score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) By Topic Trend Group")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3_pivot_distribution.pdf")
    fig.savefig(RESULTS_DIR / "figure3_pivot_distribution.png")
    plt.close(fig)
    log.info("  Saved figure3")

    # Figure 5: Binned scatter
    valid = df.dropna(subset=["trend_label", "avg_pivot", "hit", "re_hit"])
    n_bins = 20
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, dv, title in zip(axes, ["hit", "re_hit"],
                              ["(a) Raw Hit", "(b) Field-Adjusted Hit (re_hit)"]):
        for label in ["ascend", "stable", "descend"]:
            sub = valid[valid["trend_label"] == label].copy()
            sub["pivot_bin"] = pd.qcut(sub["avg_pivot"], q=n_bins,
                                       labels=False, duplicates="drop")
            binned = sub.groupby("pivot_bin").agg(x=("avg_pivot", "mean"), y=(dv, "mean"))
            ax.plot(binned["x"], binned["y"], marker="o", markersize=4,
                    color=COLORS[label], label=label.capitalize(), linewidth=1.5)
        ax.set_xlabel("Average Pivot Score (binned mean)")
        ax.set_ylabel(f"{dv} Rate")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Pivot Size vs Hit Rate by Topic Trend Group", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure5_pivot_vs_hit_binned.pdf")
    fig.savefig(RESULTS_DIR / "figure5_pivot_vs_hit_binned.png")
    plt.close(fig)
    log.info("  Saved figure5")

    # Figure 6: Heatmap
    valid = df.dropna(subset=["trend_label", "pivot_quintile", "hit", "re_hit"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dv, title in zip(axes, ["hit", "re_hit"],
                              ["(a) Raw Hit", "(b) Field-Adjusted Hit (re_hit)"]):
        pivot = valid.pivot_table(
            index="pivot_quintile", columns="trend_label", values=dv, aggfunc="mean")
        col_order = [c for c in ["descend", "stable", "ascend"] if c in pivot.columns]
        pivot = pivot[col_order]
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": f"Mean {dv}"})
        ax.set_xlabel("Trend Group")
        ax.set_ylabel("Pivot Quintile")
        ax.set_title(title)
    fig.suptitle("Mean Hit Rate by Pivot Quintile and Topic Trend", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure6_heatmap_pivot_trend.pdf")
    fig.savefig(RESULTS_DIR / "figure6_heatmap_pivot_trend.png")
    plt.close(fig)
    log.info("  Saved figure6")


# ====================================================================
# CEM Matching (re-run for new sample)
# ====================================================================
def run_matching(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("Running CEM Matching")
    log.info(f"{'='*60}")

    valid = df.dropna(subset=["trend_label", "avg_pivot", "hit", "re_hit",
                              "num_authors", "num_references"]).copy()
    treat = valid[valid["trend_label"].isin(["ascend", "descend"])].copy()
    treat["treatment"] = (treat["trend_label"] == "ascend").astype(int)
    log.info(f"  Treatment sample: {len(treat):,} (ascend: {treat['treatment'].sum():,}, "
             f"descend: {(1-treat['treatment']).sum():,})")

    # Coarsening
    treat["year_bin"] = treat["publication_year"]
    treat["author_bin"] = pd.cut(treat["num_authors"], bins=[0, 2, 5, 10, 50, 10000],
                                  labels=False, right=True)
    treat["ref_bin"] = pd.cut(treat["num_references"], bins=[0, 15, 30, 50, 100, 10000],
                               labels=False, right=True)
    treat["pivot_bin"] = pd.qcut(treat["avg_pivot"], q=5, labels=False, duplicates="drop")

    match_vars = ["year_bin", "author_bin", "ref_bin", "pivot_bin"]
    treat["stratum"] = treat[match_vars].astype(str).agg("_".join, axis=1)

    # Keep only strata with both treatment and control
    stratum_counts = treat.groupby("stratum")["treatment"].agg(["sum", "count"])
    stratum_counts["n_control"] = stratum_counts["count"] - stratum_counts["sum"]
    valid_strata = stratum_counts[(stratum_counts["sum"] > 0) & (stratum_counts["n_control"] > 0)].index
    matched = treat[treat["stratum"].isin(valid_strata)].copy()
    log.info(f"  Matched sample: {len(matched):,}")
    log.info(f"    Ascend: {matched['treatment'].sum():,}")
    log.info(f"    Descend: {(1-matched['treatment']).sum():,}")

    # Balance table
    balance_vars = ["avg_pivot", "num_authors", "num_references", "num_funds",
                    "publication_year", "cited_by_count"]
    balance_rows = []
    for var in balance_vars:
        for sample_name, sample_df in [("Pre-match", treat), ("Post-match", matched)]:
            t_mean = sample_df.loc[sample_df["treatment"] == 1, var].mean()
            c_mean = sample_df.loc[sample_df["treatment"] == 0, var].mean()
            pooled_std = sample_df[var].std()
            smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
            balance_rows.append({
                "variable": var, "sample": sample_name,
                "mean_ascend": t_mean, "mean_descend": c_mean,
                "diff": t_mean - c_mean, "std_mean_diff": smd,
            })
    pd.DataFrame(balance_rows).to_csv(
        RESULTS_DIR / "table8_cem_balance.csv", index=False, float_format="%.4f")
    log.info("  Saved table8")

    # ATT
    att_rows = []
    for dv in ["hit", "re_hit"]:
        sub = matched.dropna(subset=[dv])
        t_mean = sub.loc[sub["treatment"] == 1, dv].mean()
        c_mean = sub.loc[sub["treatment"] == 0, dv].mean()
        att = t_mean - c_mean
        t_vals = sub.loc[sub["treatment"] == 1, dv]
        c_vals = sub.loc[sub["treatment"] == 0, dv]
        t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        att_rows.append({
            "DV": dv, "mean_ascend": t_mean, "mean_descend": c_mean,
            "ATT": att, "t_stat": t_stat, "p_value": p_val,
            "N_ascend": len(t_vals), "N_descend": len(c_vals),
        })
    pd.DataFrame(att_rows).to_csv(
        RESULTS_DIR / "table9_cem_att.csv", index=False, float_format="%.6f")
    log.info("  Saved table9")


# ====================================================================
# Table 7: hit vs re_hit comparison
# ====================================================================
def table7(hit_models: dict, rehit_models: dict):
    log.info("Table 7: hit vs re_hit comparison")
    key_vars = ["avg_pivot", "topic_log_slope", "avg_pivot:topic_log_slope",
                "D_ascend", "D_descend", "avg_pivot:D_ascend", "avg_pivot:D_descend"]
    rows = []
    for model_name in ["M1", "M2", "M3", "M4"]:
        m_hit   = hit_models.get(model_name)
        m_rehit = rehit_models.get(model_name)
        for var in key_vars:
            row = {"model": model_name, "variable": var}
            for prefix, m in [("hit", m_hit), ("rehit", m_rehit)]:
                if m is not None:
                    tidy = m.tidy().reset_index()
                    if "index" in tidy.columns:
                        tidy = tidy.rename(columns={"index": "Coefficient"})
                    match = tidy[tidy["Coefficient"] == var]
                    if not match.empty:
                        row[f"{prefix}_coef"] = match["Estimate"].values[0]
                        row[f"{prefix}_se"]   = match["Std. Error"].values[0]
                        row[f"{prefix}_pval"] = match["Pr(>|t|)"].values[0]
            if len(row) > 2:
                rows.append(row)
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "table7_hit_vs_rehit.csv", index=False, float_format="%.6f")
    log.info("  Saved table7")


# ====================================================================
# Logit robustness
# ====================================================================
def run_logit_robustness(df: pd.DataFrame):
    log.info("\nLogit robustness (M3 equivalent, no FE)")
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit

        data = regression_sample(df, extra_cols=["hit", "re_hit"])
        x_vars = ["avg_pivot", "topic_log_slope", "num_authors"]
        data["pivot_x_slope"] = data["avg_pivot"] * data["topic_log_slope"]
        x_vars_all = x_vars + ["pivot_x_slope"]

        X = sm.add_constant(data[x_vars_all])

        for dv in ["hit", "re_hit"]:
            log.info(f"  Logit M3 with DV={dv}")
            try:
                model = Logit(data[dv], X.astype(float)).fit(disp=0, maxiter=100)
                summary_df = pd.DataFrame({
                    "coef": model.params[x_vars_all],
                    "se": model.bse[x_vars_all],
                    "pval": model.pvalues[x_vars_all],
                })
                summary_df.to_csv(
                    RESULTS_DIR / f"logit_m3_{dv}.csv", float_format="%.6f")
                log.info(f"    Saved logit_m3_{dv}.csv")
            except Exception as e:
                log.error(f"    Logit failed for {dv}: {e}")
    except ImportError:
        log.warning("  statsmodels not available, skipping logit")


# ====================================================================
# Main
# ====================================================================
def main():
    df = load_data()

    # Descriptive tables and figures
    run_descriptives(df)

    # Main binary models
    hit_models   = run_main_models(df, "hit")
    rehit_models = run_main_models(df, "re_hit")
    save_table(hit_models,   RESULTS_DIR / "table4_hit_models.csv")
    save_table(rehit_models, RESULTS_DIR / "table5_rehit_models.csv")

    # Continuous DV
    cont_models = run_continuous_models(df)
    save_table(cont_models, RESULTS_DIR / "table6_logcitations_models.csv")

    # Nonlinearity
    nl_models = run_nonlinear_models(df)
    save_table(nl_models, RESULTS_DIR / "table6b_nonlinear_models.csv")

    # Figure 7
    figure7(df, nl_models.get("M8"))

    # Table 7
    table7(hit_models, rehit_models)

    # Matching
    run_matching(df)

    # Logit
    run_logit_robustness(df)

    log.info("\nAll v2 regression outputs saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
