"""
07_openalex_pivot.py
====================
OpenAlex-only pivot experiment (SI robustness check).

Computes pivot scores using ONLY OpenAlex reference lists
(instead of Dimensions references), then re-runs key regressions
to check whether results are robust to the reference data source.

Pipeline:
  1. Extract OpenAlex work IDs for our biology sample
  2. Build reference lists from works_referenced_works.csv (188 GB)
  3. Build author → paper sequences from works_authorships.csv (71 GB)
  4. Compute pivot scores (1 - cosine_sim of reference sets)
  5. Merge with analysis sample and re-run M3 regressions

Input
-----
  data/analysis_sample_v2.csv
  OpenAlex20260202/csv-files/works_referenced_works.csv  (188 GB)
  OpenAlex20260202/csv-files/works_authorships.csv       (71 GB)
  OpenAlex20260202/csv-files/works.csv                   (515 GB, for pub year)

Output
------
  results_v2/si_openalex_pivot.csv
  results_v2/si_openalex_pivot_comparison.csv

Usage
-----
  nohup python 07_openalex_pivot.py > logs_v2/phase5_openalex_pivot.log 2>&1 &
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -- Paths --
DATA_DIR = Path("/gpfs/kellogg/proj/dashun/abbey/interaction_env/data")
OA_DIR = Path("/gpfs/kellogg/proj/dashun/OpenAlex20260202/csv-files")
RESULTS = Path("/gpfs/kellogg/proj/dashun/abbey/thesis/results_v2")
SAMPLE = DATA_DIR / "analysis_sample_v2.csv"

REF_FILE = OA_DIR / "works_referenced_works.csv"
AUTH_FILE = OA_DIR / "works_authorships.csv"
WORKS_FILE = OA_DIR / "works.csv"

CHUNKSIZE = 5_000_000
MIN_REFS = 5
MIN_PAPERS = 5

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def cosine_sim_sets(set_a: set, set_b: set) -> float:
    """Cosine similarity between two sets (binary vectors)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / (len(set_a) ** 0.5 * len(set_b) ** 0.5)


def main():
    # ================================================================
    # 1. Load sample and get OpenAlex work IDs
    # ================================================================
    log.info("Loading analysis sample ...")
    sample = pd.read_csv(SAMPLE)
    log.info(f"  {len(sample):,} papers")

    # Set of OpenAlex work IDs in our sample
    sample_oa_ids = set(sample["work_id"].dropna())
    log.info(f"  OpenAlex IDs in sample: {len(sample_oa_ids):,}")

    # Also need all referenced works (not just our sample) for pivot calculation
    # but we only need references FROM our sample papers

    # ================================================================
    # 2. Build reference lists from works_referenced_works.csv
    # ================================================================
    log.info(f"\nReading reference lists from {REF_FILE} ...")
    paper_refs: dict[str, set] = {}
    n_total = 0
    n_matched = 0

    reader = pd.read_csv(
        REF_FILE,
        chunksize=CHUNKSIZE,
        dtype={"work_id": "str", "referenced_work_id": "str"},
    )

    for chunk in tqdm(reader, desc="Reading references"):
        n_total += len(chunk)
        # Keep only references FROM papers in our sample
        chunk = chunk[chunk["work_id"].isin(sample_oa_ids)]
        n_matched += len(chunk)

        for _, row in chunk.iterrows():
            wid = row["work_id"]
            rid = row["referenced_work_id"]
            if wid not in paper_refs:
                paper_refs[wid] = set()
            paper_refs[wid].add(rid)

    log.info(f"  Total reference rows: {n_total:,}")
    log.info(f"  Matched to sample: {n_matched:,}")
    log.info(f"  Papers with references: {len(paper_refs):,}")

    # Filter to papers with >= MIN_REFS references
    paper_refs = {k: v for k, v in paper_refs.items() if len(v) >= MIN_REFS}
    log.info(f"  Papers with >= {MIN_REFS} references: {len(paper_refs):,}")

    # ================================================================
    # 3. Build author -> paper sequences from works_authorships.csv
    # ================================================================
    log.info(f"\nReading author-paper mappings from {AUTH_FILE} ...")

    # We need publication year for ordering — get from our sample
    year_lookup = dict(zip(sample["work_id"], sample["publication_year"]))

    # Only consider papers that have references
    papers_with_refs = set(paper_refs.keys())

    author_papers: dict[str, list] = defaultdict(list)
    n_total_auth = 0
    n_matched_auth = 0

    reader = pd.read_csv(
        AUTH_FILE,
        usecols=["work_id", "author_id"],
        chunksize=CHUNKSIZE,
        dtype={"work_id": "str", "author_id": "str"},
    )

    for chunk in tqdm(reader, desc="Reading authorships"):
        n_total_auth += len(chunk)
        # Keep only papers in our sample that have references
        chunk = chunk[chunk["work_id"].isin(papers_with_refs)]
        chunk = chunk.dropna(subset=["author_id"])
        # Deduplicate
        chunk = chunk.drop_duplicates(subset=["work_id", "author_id"])
        n_matched_auth += len(chunk)

        for _, row in chunk.iterrows():
            wid = row["work_id"]
            aid = row["author_id"]
            year = year_lookup.get(wid)
            if year is not None:
                author_papers[aid].append((int(year), wid))

    log.info(f"  Total authorship rows: {n_total_auth:,}")
    log.info(f"  Matched to sample+refs: {n_matched_auth:,}")
    log.info(f"  Unique authors: {len(author_papers):,}")

    # Filter to authors with >= MIN_PAPERS papers
    author_papers = {k: v for k, v in author_papers.items() if len(v) >= MIN_PAPERS}
    log.info(f"  Authors with >= {MIN_PAPERS} papers: {len(author_papers):,}")

    # ================================================================
    # 4. Compute OpenAlex-based pivot scores
    # ================================================================
    log.info("\nComputing OpenAlex pivot scores ...")

    pivot_records = []
    n_computed = 0
    n_skipped = 0

    for author_id, paper_list in tqdm(author_papers.items(), desc="Computing pivots"):
        # Deduplicate and sort by year
        seen = set()
        deduped = []
        for year, wid in paper_list:
            if wid not in seen:
                seen.add(wid)
                deduped.append((year, wid))
        deduped.sort(key=lambda x: x[0])

        if len(deduped) < 2:
            continue

        # Accumulate prior references
        prior_refs: set = set()
        for i, (year, wid) in enumerate(deduped):
            current_refs = paper_refs.get(wid, set())

            if i == 0:
                prior_refs = prior_refs | current_refs
                continue

            if len(prior_refs) == 0 or len(current_refs) < MIN_REFS:
                prior_refs = prior_refs | current_refs
                n_skipped += 1
                continue

            # Compute pivot = 1 - cosine_similarity
            sim = cosine_sim_sets(prior_refs, current_refs)
            oa_pivot = 1.0 - sim

            pivot_records.append({
                "work_id": wid,
                "author_id": author_id,
                "oa_pivot": oa_pivot,
                "n_prior_refs": len(prior_refs),
                "n_current_refs": len(current_refs),
            })
            n_computed += 1

            # Update prior refs
            prior_refs = prior_refs | current_refs

    log.info(f"  Pivot scores computed: {n_computed:,}")
    log.info(f"  Skipped (insufficient refs): {n_skipped:,}")

    # ================================================================
    # 5. Aggregate to paper level (avg across authors)
    # ================================================================
    log.info("\nAggregating to paper level ...")
    pivots_df = pd.DataFrame(pivot_records)

    if len(pivots_df) == 0:
        log.error("No pivot scores computed! Aborting.")
        return

    # Average pivot across all authors of each paper
    paper_pivots = (
        pivots_df.groupby("work_id")["oa_pivot"]
        .mean()
        .reset_index()
        .rename(columns={"oa_pivot": "avg_oa_pivot"})
    )
    log.info(f"  Papers with OA pivot: {len(paper_pivots):,}")

    # ================================================================
    # 6. Merge with analysis sample and compare
    # ================================================================
    log.info("\nMerging with analysis sample ...")
    merged = sample.merge(paper_pivots, on="work_id", how="inner")
    log.info(f"  Merged sample: {len(merged):,}")

    # Compare Dimensions pivot vs OpenAlex pivot
    corr = merged["avg_pivot"].corr(merged["avg_oa_pivot"])
    log.info(f"\n  Correlation(Dimensions pivot, OA pivot): {corr:.4f}")
    log.info(f"  Dimensions avg_pivot: mean={merged['avg_pivot'].mean():.4f}, "
             f"std={merged['avg_pivot'].std():.4f}")
    log.info(f"  OpenAlex avg_oa_pivot: mean={merged['avg_oa_pivot'].mean():.4f}, "
             f"std={merged['avg_oa_pivot'].std():.4f}")

    # ================================================================
    # 7. Run key regressions with OpenAlex pivot
    # ================================================================
    log.info("\nRunning regressions with OpenAlex pivot ...")
    import pyfixest as pf

    # Prepare variables
    merged["year_fe"] = merged["publication_year"].astype(str)
    merged["topic_fe"] = merged["level1_topic"].astype(str)
    merged["oa_pivot_sq"] = merged["avg_oa_pivot"] ** 2

    results_rows = []

    for dv in ["re_hit", "hit", "log_citations"]:
        for pivot_var, label in [("avg_pivot", "Dimensions"), ("avg_oa_pivot", "OpenAlex")]:
            spec = f"{dv} ~ {pivot_var} + topic_log_slope + {pivot_var}:topic_log_slope + num_authors | year_fe + topic_fe"
            try:
                model = pf.feols(spec, data=merged, vcov={"CRV1": "topic_fe"})
                coefs = model.coef()
                ses = model.se()
                pvals = model.pvalue()

                row = {
                    "DV": dv,
                    "pivot_source": label,
                    "N": model._N,
                    "R2": round(model._r2, 4),
                }

                for var in coefs.index:
                    clean = var.replace(pivot_var, "pivot")
                    row[f"{clean}_coef"] = round(coefs[var], 6)
                    row[f"{clean}_se"] = round(ses[var], 6)
                    row[f"{clean}_pval"] = round(pvals[var], 6)

                results_rows.append(row)
                log.info(f"  {label} {dv}: N={model._N:,}, R2={model._r2:.4f}, "
                         f"interaction={coefs.iloc[2]:.4f} (p={pvals.iloc[2]:.4f})")
            except Exception as e:
                log.warning(f"  Failed {label} {dv}: {e}")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS / "si_openalex_pivot.csv", index=False)
    log.info(f"\n  Saved si_openalex_pivot.csv")

    # Also save the comparison stats
    comparison = pd.DataFrame([{
        "metric": "correlation",
        "value": corr,
    }, {
        "metric": "dim_pivot_mean",
        "value": merged["avg_pivot"].mean(),
    }, {
        "metric": "dim_pivot_std",
        "value": merged["avg_pivot"].std(),
    }, {
        "metric": "oa_pivot_mean",
        "value": merged["avg_oa_pivot"].mean(),
    }, {
        "metric": "oa_pivot_std",
        "value": merged["avg_oa_pivot"].std(),
    }, {
        "metric": "n_papers_with_both",
        "value": len(merged),
    }])
    comparison.to_csv(RESULTS / "si_openalex_pivot_comparison.csv", index=False)
    log.info(f"  Saved si_openalex_pivot_comparison.csv")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
