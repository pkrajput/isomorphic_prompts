#!/bin/bash
#
# run_metrics.sh -- Opcode analysis: extract distributions + JSD (Original vs ISO).
#
# Usage:
#   ./opcode_analysis/run_metrics.sh --model gemini-2.0-flash
#   ./opcode_analysis/run_metrics.sh --model codestral-22b --datasets mbpp

set -e

MODEL=""
DATASETS="bigobench,effibench,mbpp"
RESULTS_ROOT="./results"
METRICS_ROOT="./metrics"
DATASETS_ROOT="./datasets"

if command -v python3 &> /dev/null; then PYTHON="python3"; else PYTHON="python"; fi

usage() {
    cat <<EOF
Usage: $0 --model <model> [OPTIONS]

Required:
  --model           Model name (e.g. gemini-2.0-flash)

Options:
  --datasets        Comma-separated (default: bigobench,effibench,mbpp)
  --results_root    Results directory (default: ./results)
  --metrics_root    Output directory (default: ./metrics)
  --datasets_root   Datasets directory (default: ./datasets)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL="$2";        shift 2 ;;
        --datasets)      DATASETS="$2";     shift 2 ;;
        --results_root)  RESULTS_ROOT="$2"; shift 2 ;;
        --metrics_root)  METRICS_ROOT="$2"; shift 2 ;;
        --datasets_root) DATASETS_ROOT="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "ERROR: --model is required"; usage; exit 1; }

IFS=',' read -ra DS_ARRAY <<< "$DATASETS"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

echo "=== Opcode Analysis ==="
echo "Model: $MODEL | Datasets: $DATASETS"

echo ""
echo "[1/3] Preparing metric-ready data..."
$PYTHON opcode_analysis/metric_ready.py \
    --results_root "$RESULTS_ROOT" \
    --out_root "$METRICS_ROOT/ready" \
    --datasets_root "$DATASETS_ROOT" \
    --datasets "$DATASETS" \
    --models "$MODEL" \
    --rewrite_marker

echo ""
echo "[2/3] Extracting opcode distributions..."
for DS in "${DS_ARRAY[@]}"; do
    mkdir -p "$METRICS_ROOT/opcodes/$MODEL/$DS"
    for COND in original iso; do
        READY="$METRICS_ROOT/ready/$MODEL/$DS/${COND}.ready.jsonl"
        [[ -f "$READY" ]] || continue
        echo "  $DS/$COND"
        $PYTHON opcode_analysis/extract_opcodes.py \
            --ready_jsonl "$READY" \
            --out_jsonl "$METRICS_ROOT/opcodes/$MODEL/$DS/${COND}.jsonl" \
            --store_centroid
    done
done

echo ""
echo "[3/3] Comparing conditions (orig vs iso)..."
for DS in "${DS_ARRAY[@]}"; do
    ORIG="$METRICS_ROOT/opcodes/$MODEL/$DS/original.jsonl"
    ISO="$METRICS_ROOT/opcodes/$MODEL/$DS/iso.jsonl"
    if [[ -f "$ORIG" && -f "$ISO" ]]; then
        echo "  $DS"
        $PYTHON opcode_analysis/compare_conditions.py \
            --orig_jsonl "$ORIG" \
            --iso_jsonl "$ISO" \
            --out_jsonl "$METRICS_ROOT/opcodes/$MODEL/$DS/orig_vs_iso.jsonl"
    fi
done

echo ""
echo "Done. Output: $METRICS_ROOT/opcodes/$MODEL/"
