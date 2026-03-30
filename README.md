# When Scientists Chase the Heat: Linking Topic Popularity to the Research Pivot Penalty

MA Thesis, MACSS Program, University of Chicago (June 2026)

**Author:** Yingrong Mao
**Faculty Advisor:** Bernard Koch
**Preceptor:** Sabrina Nardin

## Overview

This study examines whether the popularity trajectory of research topics moderates the pivot penalty in science. Using 4.2 million biology papers (1980-2020) from Dimensions and OpenAlex, I find that hot topics amplify the pivot penalty: researchers who make large pivots into ascending topics face a steeper penalty than those who pivot into stable or declining areas.

## Repository Structure

```
scripts/                          # Analysis pipeline
  build_analysis_sample_v2.py     # Phase 0: Build paper-level analysis dataset
  build_author_transitions.py     # Phase 1: Build author-paper transition dataset
  02_regression_v2.py             # Phase 2: Main regressions, CEM matching, logit
  05_change_analysis.py           # Phase 3: Individual-level change analysis with author FE
  06_si_robustness.py             # Phase 4: SI robustness (thresholds, time windows, etc.)
  07_openalex_pivot.py            # Phase 5: OpenAlex-only pivot experiment (SI)
  run_all_v2.sh                   # Master pipeline runner

draft/                            # Thesis manuscript
  main.tex                        # LaTeX source
  references.bib                  # Bibliography
  images/                         # Figures
```

## Data Sources

- **Dimensions** (institutional license): Paper metadata, reference lists, citation counts, field classifications
- **OpenAlex** (open access): Concept taxonomy, topic classifications, reference lists

## Pipeline

Run `scripts/run_all_v2.sh` to execute the full pipeline. Phases 0 and 1 run sequentially; Phases 2, 3, and 4 run in parallel.

### Dependencies

- Python 3.12+
- pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, tqdm
- pyfixest (high-dimensional fixed effects estimation)

## Key Findings

1. The interaction between pivot distance and topic temperature is negative and significant, indicating that hot topics amplify the pivot penalty
2. The composition effect: ~53% of the raw interaction is driven by subfield composition (ascending topics cluster in high-citation subfields)
3. Author fixed effects confirm the amplification effect operates within individual careers

## License

This project is for academic purposes. Data access requires institutional licenses for Dimensions.
