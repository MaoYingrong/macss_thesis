#!/bin/bash
# run_all_v2.sh
# ==============
# Master runner for thesis v2 pipeline.
# Sends email notification after each phase.
#
# Usage:
#   nohup bash run_all_v2.sh > logs_v2/run_all_v2.log 2>&1 &

set -e

EMAIL="yingrong.mao@kellogg.northwestern.edu"
THESIS_DIR="/gpfs/kellogg/proj/dashun/abbey/thesis"
LOG_DIR="$THESIS_DIR/logs_v2"

cd "$THESIS_DIR"
mkdir -p "$LOG_DIR"

send_email() {
    local subject="$1"
    local body="$2"
    echo "$body" | mail -s "$subject" "$EMAIL"
}

echo "======================================"
echo "Thesis v2 Pipeline Start: $(date)"
echo "======================================"

# ============================================================
# Phase 0: Build analysis sample v2 (1980-2020)
# ============================================================
echo ""
echo ">>> Phase 0: Building analysis sample v2 ..."
echo "    Started at $(date)"

python build_analysis_sample_v2.py > "$LOG_DIR/phase0_build_sample.log" 2>&1
PHASE0_STATUS=$?

if [ $PHASE0_STATUS -ne 0 ]; then
    send_email "[Thesis v2] Phase 0 FAILED" \
        "Phase 0 (build analysis sample v2) failed at $(date). Check log: $LOG_DIR/phase0_build_sample.log"
    echo "Phase 0 FAILED. Aborting pipeline."
    exit 1
fi

SAMPLE_SIZE=$(wc -l < /gpfs/kellogg/proj/dashun/abbey/interaction_env/data/analysis_sample_v2.csv)
send_email "[Thesis v2] Phase 0 Complete" \
    "Phase 0 (build analysis sample v2) completed at $(date).
Sample size: $SAMPLE_SIZE rows (including header).
Year range: 1980-2020.
Log: $LOG_DIR/phase0_build_sample.log"
echo "    Phase 0 complete at $(date). Sample: $SAMPLE_SIZE rows."

# ============================================================
# Phase 1: Build author transitions dataset
# ============================================================
echo ""
echo ">>> Phase 1: Building author transitions dataset ..."
echo "    Started at $(date)"

python build_author_transitions.py > "$LOG_DIR/phase1_author_transitions.log" 2>&1
PHASE1_STATUS=$?

if [ $PHASE1_STATUS -ne 0 ]; then
    send_email "[Thesis v2] Phase 1 FAILED" \
        "Phase 1 (author transitions) failed at $(date). Check log: $LOG_DIR/phase1_author_transitions.log"
    echo "Phase 1 FAILED."
else
    TRANS_SIZE=$(wc -l < /gpfs/kellogg/proj/dashun/abbey/interaction_env/data/author_paper_transitions.csv)
    send_email "[Thesis v2] Phase 1 Complete" \
        "Phase 1 (author transitions) completed at $(date).
Transitions dataset: $TRANS_SIZE rows.
Log: $LOG_DIR/phase1_author_transitions.log"
    echo "    Phase 1 complete at $(date). Transitions: $TRANS_SIZE rows."
fi

# ============================================================
# Phase 2: Regressions v2 (parallel with Phase 3)
# Phase 3: Change analysis (parallel with Phase 2)
# Phase 4: SI robustness
# ============================================================
echo ""
echo ">>> Phase 2: Regressions v2 (starting in background) ..."
echo "    Started at $(date)"
python 02_regression_v2.py > "$LOG_DIR/phase2_regression.log" 2>&1 &
PID_REG=$!

# Only start Phase 3 if Phase 1 succeeded
if [ $PHASE1_STATUS -eq 0 ]; then
    echo ">>> Phase 3: Change analysis (starting in background) ..."
    echo "    Started at $(date)"
    python 05_change_analysis.py > "$LOG_DIR/phase3_change_analysis.log" 2>&1 &
    PID_CHANGE=$!
fi

echo ">>> Phase 4: SI robustness (starting in background) ..."
echo "    Started at $(date)"
python 06_si_robustness.py > "$LOG_DIR/phase4_si_robustness.log" 2>&1 &
PID_SI=$!

# Wait for Phase 2
wait $PID_REG
PHASE2_STATUS=$?
if [ $PHASE2_STATUS -ne 0 ]; then
    send_email "[Thesis v2] Phase 2 FAILED" \
        "Phase 2 (regressions v2) failed at $(date). Check log: $LOG_DIR/phase2_regression.log"
    echo "    Phase 2 FAILED at $(date)."
else
    N_FILES=$(ls -1 /gpfs/kellogg/proj/dashun/abbey/thesis/results_v2/ | wc -l)
    send_email "[Thesis v2] Phase 2 Complete" \
        "Phase 2 (regressions v2) completed at $(date).
Controls: num_authors only (dropped num_references, num_funds).
$N_FILES output files in results_v2/.
Log: $LOG_DIR/phase2_regression.log"
    echo "    Phase 2 complete at $(date). $N_FILES output files."
fi

# Wait for Phase 3
if [ $PHASE1_STATUS -eq 0 ]; then
    wait $PID_CHANGE
    PHASE3_STATUS=$?
    if [ $PHASE3_STATUS -ne 0 ]; then
        send_email "[Thesis v2] Phase 3 FAILED" \
            "Phase 3 (change analysis) failed at $(date). Check log: $LOG_DIR/phase3_change_analysis.log"
        echo "    Phase 3 FAILED at $(date)."
    else
        send_email "[Thesis v2] Phase 3 Complete" \
            "Phase 3 (change analysis) completed at $(date).
Includes: transition regressions, popularity change models, author FE.
Log: $LOG_DIR/phase3_change_analysis.log"
        echo "    Phase 3 complete at $(date)."
    fi
fi

# Wait for Phase 4
wait $PID_SI
PHASE4_STATUS=$?
if [ $PHASE4_STATUS -ne 0 ]; then
    send_email "[Thesis v2] Phase 4 FAILED" \
        "Phase 4 (SI robustness) failed at $(date). Check log: $LOG_DIR/phase4_si_robustness.log"
    echo "    Phase 4 FAILED at $(date)."
else
    send_email "[Thesis v2] Phase 4 Complete" \
        "Phase 4 (SI robustness) completed at $(date).
Includes: 7 threshold variations, 3 window sizes, sub-period analysis.
Log: $LOG_DIR/phase4_si_robustness.log"
    echo "    Phase 4 complete at $(date)."
fi

# ============================================================
# Final summary
# ============================================================
echo ""
echo "======================================"
echo "Pipeline finished at $(date)"
echo "======================================"

N_TOTAL=$(ls -1 /gpfs/kellogg/proj/dashun/abbey/thesis/results_v2/ | wc -l)
send_email "[Thesis v2] ALL PHASES COMPLETE" \
    "Full thesis v2 pipeline completed at $(date).

Summary:
  Phase 0: Build sample v2 (1980-2020) - $([ $PHASE0_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED')
  Phase 1: Author transitions - $([ $PHASE1_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED')
  Phase 2: Regressions v2 - $([ $PHASE2_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED')
  Phase 3: Change analysis - $([ $PHASE1_STATUS -eq 0 ] && ([ $PHASE3_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED') || echo 'SKIPPED')
  Phase 4: SI robustness - $([ $PHASE4_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED')

Total output files: $N_TOTAL in results_v2/
Logs: $LOG_DIR/"
