"""
build_analysis_sample_v2.py
============================
Phase 0 (v2): Build clean paper-level analysis dataset for thesis.

Changes from v1:
  - Year range: 1980-2020 (was 1985-2018)
  - Includes Dimensions paper ID ('id') for author linkage
  - Output: analysis_sample_v2.csv

Input
-----
  data/merged_biology_popularity_pivot.csv   (96 M rows, paper x concept)
  data/final_biology_l2concepts_l1topic.csv  (39 M rows)
  data/concept_trends_level2.csv             (998 K rows)

Output
------
  data/analysis_sample_v2.csv

Usage
-----
  nohup python build_analysis_sample_v2.py > logs_v2/build_analysis_sample_v2.log 2>&1 &
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -- Paths --
DATA_DIR    = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
MERGED      = DATA_DIR / "merged_biology_popularity_pivot.csv"
L2_CONCEPTS = DATA_DIR / "final_biology_l2concepts_l1topic.csv"
TRENDS      = DATA_DIR / "concept_trends_level2.csv"
OUTPUT      = DATA_DIR / "analysis_sample_v2.csv"

# -- Constants --
YEAR_MIN, YEAR_MAX = 1980, 2020
MIN_REFS   = 5
CHUNKSIZE  = 1_000_000
L2_CHUNK   = 5_000_000
TREND_EPS  = 0.05

# Columns to extract at paper level from the merged file
# Added "id" (Dimensions paper ID) for author linkage
PAPER_COLS = [
    "work_id", "id", "publication_year", "cited_by_count",
    "num_authors", "num_references", "num_funds",
    "field_l1", "avg_pivot", "CINF", "hit", "type",
]

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ====================================================================
# Step 0a: Extract unique paper-level rows
# ====================================================================
def extract_papers() -> pd.DataFrame:
    log.info("Step 0a: Extracting unique papers from merged file ...")

    seen: set = set()
    chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(
        MERGED,
        usecols=PAPER_COLS,
        chunksize=CHUNKSIZE,
        dtype={"type": "str", "field_l1": "str", "id": "str"},
    )

    for chunk in tqdm(reader, desc="Reading merged CSV"):
        dedup = chunk.drop_duplicates(subset="work_id")
        new = dedup[~dedup["work_id"].isin(seen)]
        seen.update(new["work_id"].tolist())
        chunks.append(new)

    papers = pd.concat(chunks, ignore_index=True)
    log.info(f"  Total unique papers: {len(papers):,}")

    # -- Filters --
    n0 = len(papers)
    papers = papers[papers["avg_pivot"].notna()]
    log.info(f"  After avg_pivot not-null: {len(papers):,}  (dropped {n0-len(papers):,})")

    n0 = len(papers)
    papers = papers[papers["publication_year"].between(YEAR_MIN, YEAR_MAX)]
    log.info(f"  After year [{YEAR_MIN}-{YEAR_MAX}]: {len(papers):,}  (dropped {n0-len(papers):,})")

    n0 = len(papers)
    papers = papers[papers["num_references"] >= MIN_REFS]
    log.info(f"  After min refs >= {MIN_REFS}: {len(papers):,}  (dropped {n0-len(papers):,})")

    n0 = len(papers)
    papers = papers[papers["type"] != "preprint"]
    log.info(f"  After preprint exclusion: {len(papers):,}  (dropped {n0-len(papers):,})")

    papers = papers.drop(columns=["type"]).reset_index(drop=True)
    papers["publication_year"] = papers["publication_year"].astype(int)
    log.info(f"  Final paper count: {len(papers):,}")
    return papers


# ====================================================================
# Steps 0b + 0d: L2 concepts -> topic temperature + level1_topic
# ====================================================================
def process_concepts(papers: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 0b/0d: Processing L2 concepts and concept trends ...")

    trends = pd.read_csv(TRENDS)
    trends = trends[trends["end_year"].between(YEAR_MIN, YEAR_MAX)]
    log.info(f"  Concept trends loaded: {len(trends):,} rows (filtered to years)")

    paper_years = papers[["work_id", "publication_year"]].copy()
    valid_wids  = set(paper_years["work_id"])

    topic_parts:     list[pd.DataFrame] = []
    slope_parts:     list[pd.DataFrame] = []
    top_trend_parts: list[pd.DataFrame] = []

    reader = pd.read_csv(
        L2_CONCEPTS,
        chunksize=L2_CHUNK,
        usecols=["work_id", "concept_id", "score", "level1_topic"],
        dtype={"score": "float32"},
    )

    for chunk in tqdm(reader, desc="Processing L2 concepts"):
        chunk = chunk[chunk["work_id"].isin(valid_wids)].copy()
        if chunk.empty:
            continue

        idx = chunk.groupby("work_id")["score"].idxmax()
        best = chunk.loc[idx, ["work_id", "score", "level1_topic"]].copy()
        best.rename(columns={"score": "_best_score"}, inplace=True)
        topic_parts.append(best)

        chunk = chunk.merge(paper_years, on="work_id", how="inner")
        chunk_t = chunk.merge(
            trends[["concept_id", "end_year", "log_slope"]],
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

        idx2 = chunk_t.groupby("work_id")["score"].idxmax()
        top  = chunk_t.loc[idx2, ["work_id", "score", "log_slope"]].copy()
        top.rename(columns={"score": "_top_score", "log_slope": "top_concept_trend"}, inplace=True)
        top_trend_parts.append(top)

    # Combine level1_topic across chunks
    all_topics = pd.concat(topic_parts, ignore_index=True)
    idx = all_topics.groupby("work_id")["_best_score"].idxmax()
    paper_topics = all_topics.loc[idx, ["work_id", "level1_topic"]].copy()
    log.info(f"  Papers with level1_topic: {len(paper_topics):,}")

    # Combine topic_log_slope across chunks
    if slope_parts:
        all_slopes = pd.concat(slope_parts, ignore_index=True)
        final_slopes = (
            all_slopes.groupby("work_id")
            .agg(w_slope_sum=("w_slope_sum", "sum"), score_sum=("score_sum", "sum"))
            .reset_index()
        )
        final_slopes["topic_log_slope"] = final_slopes["w_slope_sum"] / final_slopes["score_sum"]
        final_slopes = final_slopes[["work_id", "topic_log_slope"]]
        log.info(f"  Papers with topic_log_slope: {len(final_slopes):,}")
    else:
        final_slopes = pd.DataFrame(columns=["work_id", "topic_log_slope"])

    # Combine top_concept_trend across chunks
    if top_trend_parts:
        all_top = pd.concat(top_trend_parts, ignore_index=True)
        idx = all_top.groupby("work_id")["_top_score"].idxmax()
        paper_top_trend = all_top.loc[idx, ["work_id", "top_concept_trend"]].copy()
        log.info(f"  Papers with top_concept_trend: {len(paper_top_trend):,}")
    else:
        paper_top_trend = pd.DataFrame(columns=["work_id", "top_concept_trend"])

    # Merge into papers
    papers = papers.merge(paper_topics,    on="work_id", how="left")
    papers = papers.merge(final_slopes,    on="work_id", how="left")
    papers = papers.merge(paper_top_trend, on="work_id", how="left")

    # Derive trend_label
    papers["trend_label"] = "stable"
    papers.loc[papers["topic_log_slope"] > TREND_EPS, "trend_label"] = "ascend"
    papers.loc[papers["topic_log_slope"] < -TREND_EPS, "trend_label"] = "descend"
    papers.loc[papers["topic_log_slope"].isna(), "trend_label"] = np.nan

    log.info(f"  Trend label distribution:\n{papers['trend_label'].value_counts(dropna=False)}")
    return papers


# ====================================================================
# Step 0c: Compute re_hit
# ====================================================================
def compute_re_hit(papers: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 0c: Computing re_hit ...")

    mask = (
        papers["CINF"].notna()
        & papers["publication_year"].notna()
        & papers["level1_topic"].notna()
    )
    valid = papers.loc[mask].copy()
    log.info(f"  Valid rows for re_hit: {len(valid):,} / {len(papers):,}")

    thresholds = (
        valid.groupby(["publication_year", "level1_topic"])["CINF"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"CINF": "threshold_95"})
    )

    valid = valid.merge(thresholds, on=["publication_year", "level1_topic"], how="left")
    valid["re_hit"] = (valid["CINF"] >= valid["threshold_95"]).astype(int)

    papers = papers.merge(valid[["work_id", "re_hit"]], on="work_id", how="left")
    log.info(f"  re_hit distribution:\n{papers['re_hit'].value_counts(dropna=False)}")
    return papers


# ====================================================================
# Step 0e: Derived variables
# ====================================================================
def create_derived(papers: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 0e: Creating derived variables ...")

    papers["log_citations"] = np.log1p(papers["cited_by_count"].fillna(0))
    papers["pivot_sq"]      = papers["avg_pivot"] ** 2

    papers["pivot_quintile"] = pd.qcut(
        papers["avg_pivot"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype("float")

    log.info(f"  pivot_quintile distribution:\n{papers['pivot_quintile'].value_counts().sort_index()}")
    return papers


# ====================================================================
# Main
# ====================================================================
def main():
    papers = extract_papers()
    papers = process_concepts(papers)
    papers = compute_re_hit(papers)
    papers = create_derived(papers)

    # Final column selection & save
    out_cols = [
        "work_id", "id", "publication_year", "cited_by_count", "log_citations",
        "avg_pivot", "pivot_quintile", "pivot_sq",
        "topic_log_slope", "trend_label", "top_concept_trend",
        "hit", "re_hit", "CINF",
        "num_authors", "num_references", "num_funds",
        "field_l1", "level1_topic",
    ]
    out_cols = [c for c in out_cols if c in papers.columns]
    papers = papers[out_cols]

    log.info(f"\n{'='*60}")
    log.info(f"Final analysis sample v2: {len(papers):,} papers")
    log.info(f"Year range: {papers['publication_year'].min()}-{papers['publication_year'].max()}")
    log.info(f"Columns: {list(papers.columns)}")
    log.info(f"Null counts:\n{papers.isnull().sum()}")
    log.info(f"{'='*60}")

    papers.to_csv(OUTPUT, index=False)
    log.info(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
