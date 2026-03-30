"""
06_si_robustness.py
====================
Supplementary Information robustness checks:
  A. Multiple ascending/descending thresholds (eps = 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)
  B. Different concept trend time windows (3yr, 7yr, 10yr)
  C. Re-run M3/M4 with each configuration

Input:
  data/analysis_sample_v2.csv
  data/final_biology_l2concepts_l1topic.csv  (for recomputing trends)
  data/concept_trends_level2.csv

Output:  results_v2/si_*.csv

Usage:   nohup python 06_si_robustness.py > logs_v2/06_si_robustness.log 2>&1 &
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from tqdm import tqdm

# -- Paths --
DATA_DIR    = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
RESULTS_DIR = Path("/gpfs/kellogg/proj/dashun/abbey/thesis/results_v2")
SAMPLE      = DATA_DIR / "analysis_sample_v2.csv"
L2_CONCEPTS = DATA_DIR / "final_biology_l2concepts_l1topic.csv"
TRENDS      = DATA_DIR / "concept_trends_level2.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1980, 2020
L2_CHUNK = 5_000_000

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ====================================================================
# Helpers
# ====================================================================
def run_model(formula, data, label):
    log.info(f"  {label}: {formula}")
    try:
        m = pf.feols(formula, data=data, vcov={"CRV1": "topic_fe"})
        log.info(f"    N={m._N:,}  R2={m._r2:.4f}")
        return m
    except Exception as e:
        log.error(f"    FAILED: {e}")
        return None


def extract_key_coefs(m, model_label):
    """Extract key coefficients from a model."""
    if m is None:
        return {}
    tidy = m.tidy().reset_index()
    if "index" in tidy.columns:
        tidy = tidy.rename(columns={"index": "Coefficient"})
    result = {"model": model_label, "N": m._N, "R2": m._r2}
    for _, row in tidy.iterrows():
        var = row["Coefficient"]
        result[f"{var}_coef"] = row["Estimate"]
        result[f"{var}_se"]   = row["Std. Error"]
        result[f"{var}_pval"] = row["Pr(>|t|)"]
    return result


# ====================================================================
# A. Multiple ascending/descending thresholds
# ====================================================================
def robustness_thresholds(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("SI-A: Multiple ascending/descending thresholds")
    log.info(f"{'='*60}")

    CONTROLS = "num_authors"
    FE = "year_fe + topic_fe"
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

    results = []

    for eps in thresholds:
        log.info(f"\n--- Threshold eps = {eps} ---")

        # Reclassify trend labels
        df_t = df.copy()
        df_t["trend_label_alt"] = "stable"
        df_t.loc[df_t["topic_log_slope"] > eps, "trend_label_alt"] = "ascend"
        df_t.loc[df_t["topic_log_slope"] < -eps, "trend_label_alt"] = "descend"
        df_t.loc[df_t["topic_log_slope"].isna(), "trend_label_alt"] = np.nan

        df_t["D_ascend_alt"]  = (df_t["trend_label_alt"] == "ascend").astype(int)
        df_t["D_descend_alt"] = (df_t["trend_label_alt"] == "descend").astype(int)

        n_asc  = df_t["D_ascend_alt"].sum()
        n_desc = df_t["D_descend_alt"].sum()
        n_stab = len(df_t) - n_asc - n_desc
        log.info(f"  Ascend: {n_asc:,}, Descend: {n_desc:,}, Stable: {n_stab:,}")

        for dv in ["re_hit", "hit"]:
            data = df_t.dropna(subset=[dv, "avg_pivot", "topic_log_slope",
                                       "num_authors", "year_fe", "topic_fe"])

            # M3 equivalent with continuous interaction
            label = f"eps{eps}_{dv}_M3"
            m3 = run_model(
                f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
                data, label
            )
            row = extract_key_coefs(m3, label)
            row["eps"] = eps
            row["dv"] = dv
            row["model_type"] = "M3"
            row["n_ascend"] = n_asc
            row["n_descend"] = n_desc
            results.append(row)

            # M4 equivalent with categorical
            label = f"eps{eps}_{dv}_M4"
            m4 = run_model(
                f"{dv} ~ avg_pivot + D_ascend_alt + D_descend_alt + avg_pivot:D_ascend_alt + avg_pivot:D_descend_alt + {CONTROLS} | {FE}",
                data, label
            )
            row = extract_key_coefs(m4, label)
            row["eps"] = eps
            row["dv"] = dv
            row["model_type"] = "M4"
            row["n_ascend"] = n_asc
            row["n_descend"] = n_desc
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "si_threshold_sensitivity.csv",
                      index=False, float_format="%.6f")
    log.info("  Saved si_threshold_sensitivity.csv")

    return results_df


# ====================================================================
# B. Different concept trend time windows
# ====================================================================
def recompute_trends_with_window(window: int) -> pd.DataFrame:
    """
    Recompute concept popularity trends using a different rolling window size.
    Returns DataFrame with concept_id, end_year, log_slope.
    """
    log.info(f"\n  Recomputing concept trends with {window}-year window ...")

    # Load paper-concept data to get concept-year counts
    # Use the analysis sample v2 to know which papers are in scope
    sample = pd.read_csv(SAMPLE, usecols=["work_id", "publication_year"])
    valid_wids = set(sample["work_id"])
    paper_years = dict(zip(sample["work_id"], sample["publication_year"]))

    # Count papers per concept per year
    concept_year_counts = {}

    reader = pd.read_csv(
        L2_CONCEPTS,
        chunksize=L2_CHUNK,
        usecols=["work_id", "concept_id"],
    )

    for chunk in tqdm(reader, desc=f"Counting concepts ({window}yr)"):
        chunk = chunk[chunk["work_id"].isin(valid_wids)].copy()
        if chunk.empty:
            continue
        chunk["year"] = chunk["work_id"].map(paper_years)
        chunk = chunk.dropna(subset=["year"])
        chunk["year"] = chunk["year"].astype(int)

        counts = chunk.groupby(["concept_id", "year"]).size().reset_index(name="count")
        for _, row in counts.iterrows():
            key = (row["concept_id"], row["year"])
            concept_year_counts[key] = concept_year_counts.get(key, 0) + row["count"]

    # Convert to DataFrame
    records = [{"concept_id": k[0], "year": k[1], "count": v}
               for k, v in concept_year_counts.items()]
    cy = pd.DataFrame(records)
    log.info(f"  Total concept-year pairs: {len(cy):,}")

    # Total papers per year (for computing shares)
    yearly_totals = cy.groupby("year")["count"].sum().reset_index(name="total")
    cy = cy.merge(yearly_totals, on="year")
    cy["share"] = cy["count"] / cy["total"]

    # Compute rolling log-slopes
    trends = []
    for concept_id, group in tqdm(cy.groupby("concept_id"), desc=f"Computing slopes ({window}yr)"):
        group = group.sort_values("year")
        years = group["year"].values
        shares = group["share"].values

        for i in range(len(years)):
            end_year = years[i]
            start_year = end_year - window + 1

            mask = (years >= start_year) & (years <= end_year)
            if mask.sum() < max(3, window // 2):  # need at least half the window
                continue

            y = years[mask]
            s = shares[mask]

            # Log-slope: regress log(share) on year
            s_safe = np.maximum(s, 1e-10)
            log_s = np.log(s_safe)

            if len(y) < 2:
                continue

            # Simple OLS
            y_centered = y - y.mean()
            slope = np.sum(y_centered * log_s) / np.sum(y_centered ** 2)

            trends.append({
                "concept_id": concept_id,
                "start_year": start_year,
                "end_year": end_year,
                "log_slope": slope,
            })

    trends_df = pd.DataFrame(trends)
    log.info(f"  Computed {len(trends_df):,} concept-year trend values")

    return trends_df


def recompute_paper_slopes(trends_df: pd.DataFrame, window_label: str) -> pd.DataFrame:
    """
    Given new concept trends, recompute paper-level topic_log_slope.
    Returns DataFrame with work_id, topic_log_slope_{window_label}.
    """
    log.info(f"  Recomputing paper-level slopes for {window_label} ...")

    sample = pd.read_csv(SAMPLE, usecols=["work_id", "publication_year"])
    valid_wids = set(sample["work_id"])
    paper_years = dict(zip(sample["work_id"], sample["publication_year"]))

    slope_parts = []

    reader = pd.read_csv(
        L2_CONCEPTS,
        chunksize=L2_CHUNK,
        usecols=["work_id", "concept_id", "score"],
        dtype={"score": "float32"},
    )

    for chunk in tqdm(reader, desc=f"Merging slopes ({window_label})"):
        chunk = chunk[chunk["work_id"].isin(valid_wids)].copy()
        if chunk.empty:
            continue

        chunk["publication_year"] = chunk["work_id"].map(paper_years)
        chunk = chunk.dropna(subset=["publication_year"])
        chunk["publication_year"] = chunk["publication_year"].astype(int)

        chunk_t = chunk.merge(
            trends_df[["concept_id", "end_year", "log_slope"]],
            left_on=["concept_id", "publication_year"],
            right_on=["concept_id", "end_year"],
            how="inner",
        )
        if chunk_t.empty:
            continue

        chunk_t["w_slope"] = chunk_t["score"].astype("float64") * chunk_t["log_slope"]
        agg = (
            chunk_t.groupby("work_id")
            .agg(w_slope_sum=("w_slope", "sum"), score_sum=("score", "sum"))
            .reset_index()
        )
        slope_parts.append(agg)

    if not slope_parts:
        return pd.DataFrame(columns=["work_id", f"topic_log_slope_{window_label}"])

    all_slopes = pd.concat(slope_parts, ignore_index=True)
    final = (
        all_slopes.groupby("work_id")
        .agg(w_slope_sum=("w_slope_sum", "sum"), score_sum=("score_sum", "sum"))
        .reset_index()
    )
    final[f"topic_log_slope_{window_label}"] = final["w_slope_sum"] / final["score_sum"]

    return final[["work_id", f"topic_log_slope_{window_label}"]]


def robustness_windows(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("SI-B: Different concept trend time windows")
    log.info(f"{'='*60}")

    CONTROLS = "num_authors"
    FE = "year_fe + topic_fe"

    results = []

    for window in [3, 7, 10]:
        log.info(f"\n===== Window = {window} years =====")

        # Recompute concept trends
        trends = recompute_trends_with_window(window)

        # Save concept trends for reference
        trends.to_csv(RESULTS_DIR / f"concept_trends_{window}yr.csv",
                      index=False, float_format="%.6f")

        # Recompute paper-level slopes
        label = f"{window}yr"
        paper_slopes = recompute_paper_slopes(trends, label)
        log.info(f"  Papers with new slopes: {len(paper_slopes):,}")

        # Merge with analysis sample
        df_w = df.merge(paper_slopes, on="work_id", how="left")
        slope_col = f"topic_log_slope_{label}"

        n_valid = df_w[slope_col].notna().sum()
        log.info(f"  Papers with valid {slope_col}: {n_valid:,}")

        if n_valid < 100000:
            log.warning(f"  Too few valid papers for {window}yr window, skipping")
            continue

        # Run M3 with new slopes
        for dv in ["re_hit", "hit"]:
            data = df_w.dropna(subset=[dv, "avg_pivot", slope_col,
                                       "num_authors", "year_fe", "topic_fe"]).copy()

            model_label = f"w{window}_{dv}_M3"
            m = run_model(
                f"{dv} ~ avg_pivot + {slope_col} + avg_pivot:{slope_col} + {CONTROLS} | {FE}",
                data, model_label
            )
            row = extract_key_coefs(m, model_label)
            row["window"] = window
            row["dv"] = dv
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "si_window_sensitivity.csv",
                      index=False, float_format="%.6f")
    log.info("  Saved si_window_sensitivity.csv")


# ====================================================================
# C. Other robustness from v1 (re-run with new sample/controls)
# ====================================================================
def robustness_other(df: pd.DataFrame):
    log.info(f"\n{'='*60}")
    log.info("SI-C: Other robustness checks")
    log.info(f"{'='*60}")

    CONTROLS = "num_authors"
    FE = "year_fe + topic_fe"

    results = []

    for dv in ["re_hit", "hit"]:
        data = df.dropna(subset=[dv, "avg_pivot", "topic_log_slope",
                                 "num_authors", "year_fe", "topic_fe"]).copy()

        # A1: Alternative hit thresholds
        # (re_hit is already 95th pctl; for 90th and 99th we'd need to recompute)
        # Skip if not applicable

        # A5: Min references >= 10
        data_10 = data[data["num_references"] >= 10]
        label = f"minref10_{dv}"
        m = run_model(
            f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
            data_10, label
        )
        row = extract_key_coefs(m, label)
        row["check"] = "min_refs_10"
        row["dv"] = dv
        results.append(row)

        # A6: Sub-periods
        for period_name, (y1, y2) in [("1980-2000", (1980, 2000)), ("2001-2020", (2001, 2020))]:
            data_p = data[data["publication_year"].between(y1, y2)]
            label = f"period_{period_name}_{dv}"
            m = run_model(
                f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
                data_p, label
            )
            row = extract_key_coefs(m, label)
            row["check"] = f"period_{period_name}"
            row["dv"] = dv
            results.append(row)

        # A7: Exclude extreme pivots (top/bottom 1%)
        p01, p99 = data["avg_pivot"].quantile([0.01, 0.99])
        data_trim = data[data["avg_pivot"].between(p01, p99)]
        label = f"trim_pivot_{dv}"
        m = run_model(
            f"{dv} ~ avg_pivot + topic_log_slope + avg_pivot:topic_log_slope + {CONTROLS} | {FE}",
            data_trim, label
        )
        row = extract_key_coefs(m, label)
        row["check"] = "trim_pivot_1pct"
        row["dv"] = dv
        results.append(row)

        # A4: Top-concept trend instead of score-weighted average
        data_top = data.dropna(subset=["top_concept_trend"])
        label = f"top_concept_{dv}"
        m = run_model(
            f"{dv} ~ avg_pivot + top_concept_trend + avg_pivot:top_concept_trend + {CONTROLS} | {FE}",
            data_top, label
        )
        row = extract_key_coefs(m, label)
        row["check"] = "top_concept_trend"
        row["dv"] = dv
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "si_other_robustness.csv",
                      index=False, float_format="%.6f")
    log.info("  Saved si_other_robustness.csv")


# ====================================================================
# Main
# ====================================================================
def main():
    log.info("Loading analysis sample v2 ...")
    df = pd.read_csv(SAMPLE)
    log.info(f"Loaded {len(df):,} papers")

    df["year_fe"]  = df["publication_year"].astype(str)
    df["topic_fe"] = df["level1_topic"].astype(str)
    df["D_ascend"]  = (df["trend_label"] == "ascend").astype(int)
    df["D_descend"] = (df["trend_label"] == "descend").astype(int)

    # A: Threshold sensitivity
    robustness_thresholds(df)

    # B: Window sensitivity (computationally intensive)
    robustness_windows(df)

    # C: Other robustness checks
    robustness_other(df)

    log.info("\nAll SI robustness outputs saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
