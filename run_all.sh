#!/usr/bin/env bash
# ==============================================================================
#  Single-command reproduction of all manuscript results.
# ==============================================================================
set -euo pipefail

MODE="${1:-full}"
echo "==========================================================================="
echo "  QLSTM Multi-Asset Benchmark — running in mode: $MODE"
echo "==========================================================================="

mkdir -p results/raw results/tables results/figures

# 1) Data + Table 1
echo ""; echo "[1/5]  fetch data and build Table 1"
python -m src.data_loader --start 2020-01-01 --end 2024-12-31

# 2) Experiments
echo ""; echo "[2/5]  run experiments"
if [[ "$MODE" == "--quick" ]]; then
  python -m src.run_experiments --quick
elif [[ "$MODE" == "--classical-only" ]]; then
  python -m src.run_experiments --classical-only
else
  python -m src.run_experiments
fi

# 3) Aggregate -> Tables 2-8
echo ""; echo "[3/5]  aggregate -> Tables 2-8"
python -m src.aggregate

# 4) Figures 1-4
echo ""; echo "[4/5]  build figures"
python -m src.make_figures

# 5) Fill manuscript
echo ""; echo "[5/5]  fill manuscript"
if [[ -f QLSTM_Multi_Asset_Benchmark_JoSC.docx ]]; then
  python -m src.fill_manuscript \
    --in  QLSTM_Multi_Asset_Benchmark_JoSC.docx \
    --out results/manuscript_filled.docx
else
  echo "  (manuscript template not found — skipping auto-fill)"
fi

echo ""
echo "==========================================================================="
echo "  All done. See results/manuscript_filled.docx"
echo "==========================================================================="
