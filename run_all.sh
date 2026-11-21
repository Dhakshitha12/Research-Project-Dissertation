#!/usr/bin/env bash
# Full protocol: 40k steps x 5 seeds x 5 PPO lambdas, plus 5-seed DQN.
# Run AWS calibration explicitly first; it creates billable resources.
# Usage: WORKLOAD=image_resize bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

RUNTIME="${RUNTIME:-python3.12}"
WORKLOAD="${WORKLOAD:-image_resize}"
CALIBRATION="${CALIBRATION:-data/aws_calibration.csv}"
MEMORY_GRID="${MEMORY_GRID:-128 256 512 1024}"

echo "== 1. Dependencies =="
pip install -q --no-input torch --index-url https://download.pytorch.org/whl/cpu
pip install -q --no-input stable-baselines3 gymnasium pandas scipy matplotlib

echo "== 2. Verify calibration is present =="
if [ ! -f "$CALIBRATION" ]; then
  echo "Missing calibration CSV: $CALIBRATION" >&2
  echo "Run the explicit, billable calibration command in README.md first." >&2
  exit 1
fi

echo "== 3. Training: PPO across the lambda sweep, 5 seeds each; DQN at lambda=0.02 =="
for LAM in 0.005 0.01 0.02 0.05 0.1; do
  # shellcheck disable=SC2086
  python3 train.py --algo ppo --lam "$LAM" --seeds 0 1 2 3 4 --steps 40000 \
    --calibration "$CALIBRATION" --runtime "$RUNTIME" --workload "$WORKLOAD" \
    --memory-grid-mb $MEMORY_GRID --outdir ../models
done
# shellcheck disable=SC2086
python3 train.py --algo dqn --lam 0.02 --seeds 0 1 2 3 4 --steps 40000 \
  --calibration "$CALIBRATION" --runtime "$RUNTIME" --workload "$WORKLOAD" \
  --memory-grid-mb $MEMORY_GRID --outdir ../models

echo "== 4. Paired evaluation on held-out functions + statistics + frontier =="
python3 evaluate.py --models ../models --out ../results --require-full-protocol \
  --calibration "$CALIBRATION" --runtime "$RUNTIME" --workload "$WORKLOAD" \
  --memory-grid-mb $MEMORY_GRID

echo "Done. See ../results: eval_rows.csv, frontier_points.csv (95% CIs), frontier.png, stats.csv, regime_shift.csv, summary.json"
