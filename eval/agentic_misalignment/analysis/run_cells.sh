#!/usr/bin/env bash
# Runs all 4 cells of the action-history experiment.
# Usage: bash analysis/run_cells.sh
# Run from the repository root (main-notebooks/).
#
# A3 PILOT NOTE: Before running EPOCHS=60, pilot A3 at EPOCHS=5 and inspect
# outputs for confusion patterns (model referencing its "own" prior reasoning,
# refusing to continue, sycophantic persistence). If patterns appear, remove
# A3 from the LEVELS list and set EPOCHS=80 for a 3-level design.
set -euo pipefail

MODEL="anthropic/claude-opus-4-1-20250805"
EPOCHS=60
SCENARIO="blackmail"
URGENCY="restriction"
PRIME="dont"
TEMP=1.0
LOG_BASE="logs/action_history_experiment"

mkdir -p "$LOG_BASE"

for level in A_raw A0 A1 A2 A3; do
  echo ""
  echo "=== Running cell $level (history_type=$level) ==="
  uv run inspect eval inspect_evals/agentic_misalignment \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --temperature "$TEMP" \
    --max-connections 20 \
    -T scenario="$SCENARIO" \
    -T urgency_type="$URGENCY" \
    -T extra_system_instructions="$PRIME" \
    -T history_type="$level" \
    -T feedback=false \
    -T cf=false \
    -T manipulation_check=true \
    --log-dir "$LOG_BASE/cell_$level"
  echo "=== Cell $level complete ==="
done

echo ""
echo "All 4 cells complete. Logs in $LOG_BASE/"
