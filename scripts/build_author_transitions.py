"""
build_author_transitions.py
============================
Build author-paper level dataset with prior topic temperature
and transition categories for the "change" analysis.

For each author-paper pair, computes:
  - prior_avg_slope: mean topic_log_slope of the author's prior papers
  - prior_trend_label: categorical label of prior slope
  - popularity_change: current topic_log_slope - prior_avg_slope
  - transition_type: prior_trend x current_trend (9 categories)

Input
-----
  data/analysis_sample_v2.csv                      (paper-level, includes Dimensions 'id')
  pivot_penalty/data/pivot_with_avg_hits.csv        (20 GB, author-paper level, all fields)

Output
------
  data/author_paper_transitions.csv

Usage
-----
  nohup python build_author_transitions.py > logs_v2/build_author_transitions.log 2>&1 &
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -- Paths --
DATA_DIR      = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
PIVOT_HITS    = Path("/gpfs/kellogg/proj/dashun/abbey/pivot_penalty/data/pivot_with_avg_hits.csv")
SAMPLE        = DATA_DIR / "analysis_sample_v2.csv"
OUTPUT        = DATA_DIR / "author_paper_transitions.csv"

TREND_EPS  = 0.05
CHUNKSIZE  = 2_000_000

# Biology field_l1 codes: 3000-3299 range in Dimensions
# field_l1 can have pipe-separated values like "3201|3211"
BIO_PREFIXES = [str(i) for i in range(30, 33)]  # "30", "31", "32"

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def is_biology(field_l1: str) -> bool:
    """Check if any field code starts with 30xx, 31xx, or 32xx."""
    if pd.isna(field_l1):
        return False
    for code in str(field_l1).split("|"):
        code = code.strip()
        if len(code) >= 2 and code[:2] in BIO_PREFIXES:
            return True
    return False


def classify_trend(slope: float, eps: float = TREND_EPS) -> str:
    if np.isnan(slope):
        return "unknown"
    if slope > eps:
        return "ascend"
    elif slope < -eps:
        return "descend"
    else:
        return "stable"


def main():
    # ================================================================
    # 1. Load paper-level data from analysis sample v2
    # ================================================================
    log.info("Loading analysis sample v2 ...")
    papers = pd.read_csv(SAMPLE)
    log.info(f"  Loaded {len(papers):,} papers")

    # Build lookup: dimensions_id -> paper data
    papers_with_id = papers.dropna(subset=["id"]).copy()
    log.info(f"  Papers with Dimensions ID: {len(papers_with_id):,}")

    paper_lookup = {}
    for _, row in tqdm(papers_with_id.iterrows(), total=len(papers_with_id),
                       desc="Building paper lookup"):
        paper_lookup[row["id"]] = {
            "work_id": row["work_id"],
            "publication_year": int(row["publication_year"]),
            "cited_by_count": row["cited_by_count"],
            "log_citations": row["log_citations"],
            "avg_pivot": row["avg_pivot"],
            "pivot_quintile": row["pivot_quintile"],
            "topic_log_slope": row["topic_log_slope"],
            "trend_label": row["trend_label"],
            "hit": row["hit"],
            "re_hit": row["re_hit"],
            "CINF": row["CINF"],
            "num_authors": row["num_authors"],
            "field_l1": row["field_l1"],
            "level1_topic": row["level1_topic"],
        }

    sample_dim_ids = set(paper_lookup.keys())
    log.info(f"  Paper lookup size: {len(sample_dim_ids):,}")

    # ================================================================
    # 2. Extract biology author-paper mappings from pivot_with_avg_hits.csv
    # ================================================================
    log.info(f"\nReading author-paper mappings from {PIVOT_HITS} ...")
    log.info("  Filtering to biology papers that are in our analysis sample ...")

    # Build author -> [(year, paper_id), ...] from the large file
    author_papers: dict[str, list] = {}
    n_total_rows = 0
    n_bio_rows = 0
    n_matched_rows = 0

    reader = pd.read_csv(
        PIVOT_HITS,
        usecols=["researcher_id", "id", "paper_year", "field_l1"],
        chunksize=CHUNKSIZE,
        dtype={"researcher_id": "str", "id": "str", "field_l1": "str"},
    )

    for chunk in tqdm(reader, desc="Reading pivot_with_avg_hits"):
        n_total_rows += len(chunk)

        # Filter to biology
        bio_mask = chunk["field_l1"].apply(is_biology)
        chunk = chunk[bio_mask]
        n_bio_rows += len(chunk)

        # Filter to papers in our sample
        chunk = chunk[chunk["id"].isin(sample_dim_ids)]
        n_matched_rows += len(chunk)

        # Deduplicate (same author-paper can appear with different field_l1)
        chunk = chunk.drop_duplicates(subset=["researcher_id", "id"])

        # Add to author_papers dict
        for _, row in chunk.iterrows():
            rid = row["researcher_id"]
            if rid not in author_papers:
                author_papers[rid] = []
            author_papers[rid].append((int(row["paper_year"]), row["id"]))

    log.info(f"  Total rows read: {n_total_rows:,}")
    log.info(f"  Biology rows: {n_bio_rows:,}")
    log.info(f"  Matched to sample: {n_matched_rows:,}")
    log.info(f"  Unique authors: {len(author_papers):,}")

    # Count papers per author
    author_counts = {k: len(v) for k, v in author_papers.items()}
    ge2 = sum(1 for c in author_counts.values() if c >= 2)
    ge5 = sum(1 for c in author_counts.values() if c >= 5)
    log.info(f"  Authors with >= 2 papers in sample: {ge2:,}")
    log.info(f"  Authors with >= 5 papers in sample: {ge5:,}")

    # ================================================================
    # 3. Build author-paper transitions
    # ================================================================
    log.info("\nComputing author-paper transitions ...")

    rows = []
    n_authors_processed = 0
    n_authors_skipped = 0
    n_pairs = 0

    for author_id, paper_list in tqdm(author_papers.items(), desc="Processing authors"):
        if len(paper_list) < 2:
            n_authors_skipped += 1
            continue

        # Deduplicate within author (same paper may appear if multiple field_l1 values)
        seen_pids = set()
        deduped = []
        for year, pid in paper_list:
            if pid not in seen_pids:
                seen_pids.add(pid)
                deduped.append((year, pid))

        if len(deduped) < 2:
            n_authors_skipped += 1
            continue

        # Sort by year (ascending)
        deduped.sort(key=lambda x: x[0])
        n_authors_processed += 1

        # Compute prior topic temperature for each paper after the first
        prior_slopes = []
        for i, (year, pid) in enumerate(deduped):
            pdata = paper_lookup[pid]
            current_slope = pdata["topic_log_slope"]

            if i == 0:
                # First paper: no prior, record slope for future
                if pd.notna(current_slope):
                    prior_slopes.append(current_slope)
                continue

            # Compute prior average slope
            if len(prior_slopes) > 0:
                prior_avg = np.nanmean(prior_slopes)
            else:
                prior_avg = np.nan

            # Classify prior and current trends
            prior_trend = classify_trend(prior_avg)
            current_trend = pdata["trend_label"] if pd.notna(pdata["trend_label"]) else "unknown"

            # Transition type
            if prior_trend != "unknown" and current_trend != "unknown":
                transition = f"{prior_trend}_to_{current_trend}"
            else:
                transition = "unknown"

            # Popularity change
            if pd.notna(prior_avg) and pd.notna(current_slope):
                pop_change = current_slope - prior_avg
            else:
                pop_change = np.nan

            rows.append({
                "author_id": author_id,
                "work_id": pdata["work_id"],
                "dim_id": pid,
                "publication_year": pdata["publication_year"],
                "avg_pivot": pdata["avg_pivot"],
                "cited_by_count": pdata["cited_by_count"],
                "log_citations": pdata["log_citations"],
                "pivot_quintile": pdata["pivot_quintile"],
                "hit": pdata["hit"],
                "re_hit": pdata["re_hit"],
                "CINF": pdata["CINF"],
                "num_authors": pdata["num_authors"],
                "field_l1": pdata["field_l1"],
                "level1_topic": pdata["level1_topic"],
                "topic_log_slope": current_slope,
                "trend_label": current_trend,
                "prior_avg_slope": prior_avg,
                "prior_trend_label": prior_trend,
                "popularity_change": pop_change,
                "transition_type": transition,
                "author_paper_seq": i + 1,
                "author_n_prior": i,
            })
            n_pairs += 1

            # Update prior slopes
            if pd.notna(current_slope):
                prior_slopes.append(current_slope)

    log.info(f"  Authors processed: {n_authors_processed:,}")
    log.info(f"  Authors skipped (<2 papers in sample): {n_authors_skipped:,}")
    log.info(f"  Total author-paper pairs: {n_pairs:,}")

    # ================================================================
    # 4. Save
    # ================================================================
    df = pd.DataFrame(rows)

    log.info(f"\n{'='*60}")
    log.info(f"Author-paper transitions dataset: {len(df):,} rows")
    log.info(f"Unique authors: {df['author_id'].nunique():,}")
    log.info(f"Unique papers: {df['work_id'].nunique():,}")
    log.info(f"\nTransition type distribution:")
    log.info(f"{df['transition_type'].value_counts(dropna=False)}")
    log.info(f"\nPrior trend distribution:")
    log.info(f"{df['prior_trend_label'].value_counts(dropna=False)}")
    log.info(f"\nPopularity change stats:")
    log.info(f"{df['popularity_change'].describe()}")
    log.info(f"{'='*60}")

    df.to_csv(OUTPUT, index=False)
    log.info(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
