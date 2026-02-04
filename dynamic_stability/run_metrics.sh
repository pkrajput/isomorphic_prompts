#!/bin/bash
#
# run_metrics.sh - Compute SCTD/DCTD metrics and energy shift analysis
#
# Part 3 of the I/O Isomorphism Pipeline.
# Computes Static Code Trace Divergence (SCTD) and Dynamic Code Trace Divergence (DCTD)
# to measure implementation variance between Original and ISO conditions.
#
# Usage:
#   ./dynamic_stability/run_metrics.sh --model <model> [OPTIONS]
#
# Example:
#   ./dynamic_stability/run_metrics.sh --model gemini-2.0-flash --datasets bigobench,effibench,mbpp
#   ./dynamic_stability/run_metrics.sh --model starcoder-15b --datasets mbpp --dctd_only
#

set -e

# Default values
MODEL=""
DATASETS="bigobench,effibench,mbpp"
WORKERS=16
TIMEOUT=15
RESULTS_ROOT="./results"
METRICS_ROOT="./metrics"
DATASETS_ROOT="./datasets"
SCTD_ONLY=false
DCTD_ONLY=false
SKIP_PLOTS=false

# Detect Python
if command -v python3.10 &> /dev/null; then
    PYTHON="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

usage() {
    cat <<EOF
Usage: $0 --model <model> [OPTIONS]

Compute SCTD/DCTD metrics and energy shift analysis for Original vs ISO conditions.

Required:
  --model           Model name (e.g., gemini-2.0-flash, starcoder-15b)

Options:
  --datasets        Comma-separated datasets (default: bigobench,effibench,mbpp)
  --workers         Parallel workers for DCTD (default: 16)
  --timeout         Timeout per execution in seconds (default: 15)
  --results_root    Results directory (default: ./results)
  --metrics_root    Metrics output directory (default: ./metrics)
  --datasets_root   Datasets directory (default: ./datasets)
  --sctd_only       Only compute SCTD (skip DCTD)
  --dctd_only       Only compute DCTD (skip SCTD)
  --skip_plots      Skip plot generation
  -h, --help        Show this help

Examples:
  # Full analysis for Gemini 2.0 Flash on all datasets
  $0 --model gemini-2.0-flash

  # Only DCTD for a specific dataset
  $0 --model starcoder-15b --datasets mbpp --dctd_only

  # Quick SCTD-only analysis
  $0 --model codestral-22b --sctd_only --skip_plots
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --results_root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --metrics_root)
            METRICS_ROOT="$2"
            shift 2
            ;;
        --datasets_root)
            DATASETS_ROOT="$2"
            shift 2
            ;;
        --sctd_only)
            SCTD_ONLY=true
            shift
            ;;
        --dctd_only)
            DCTD_ONLY=true
            shift
            ;;
        --skip_plots)
            SKIP_PLOTS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required args
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    usage
    exit 1
fi

# Convert comma-separated datasets to array
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

# Get script directory (for relative imports)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Dynamic Stability Metrics Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "Datasets: $DATASETS"
echo "Workers: $WORKERS"
echo "Timeout: ${TIMEOUT}s"
echo "SCTD only: $SCTD_ONLY"
echo "DCTD only: $DCTD_ONLY"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: Prepare Metric-Ready Data
# =============================================================================
echo "=== Step 1: Preparing Metric-Ready Data ==="
$PYTHON dynamic_stability/metric_ready.py \
    --results_root "$RESULTS_ROOT" \
    --out_root "$METRICS_ROOT/ready" \
    --datasets_root "$DATASETS_ROOT" \
    --datasets "$DATASETS" \
    --models "$MODEL" \
    --rewrite_marker

# =============================================================================
# Step 2: Compute SCTD (Static Code Trace Divergence)
# =============================================================================
if [[ "$DCTD_ONLY" != true ]]; then
    echo ""
    echo "=== Step 2: Computing SCTD ==="
    
    for DATASET in "${DATASET_ARRAY[@]}"; do
        echo "  Processing $DATASET..."
        mkdir -p "$METRICS_ROOT/sctd/$MODEL/$DATASET"
        
        # Original
        ORIG_READY="$METRICS_ROOT/ready/$MODEL/$DATASET/original.ready.jsonl"
        if [[ -f "$ORIG_READY" ]]; then
            echo "    - Original..."
            $PYTHON dynamic_stability/calculate_sctd.py \
                --ready_jsonl "$ORIG_READY" \
                --out_jsonl "$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_original.jsonl" \
                --store_centroid
        else
            echo "    - Original: SKIPPED (file not found)"
        fi
        
        # ISO
        ISO_READY="$METRICS_ROOT/ready/$MODEL/$DATASET/iso.ready.jsonl"
        if [[ -f "$ISO_READY" ]]; then
            echo "    - ISO..."
            $PYTHON dynamic_stability/calculate_sctd.py \
                --ready_jsonl "$ISO_READY" \
                --out_jsonl "$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_iso.jsonl" \
                --store_centroid
        else
            echo "    - ISO: SKIPPED (file not found)"
        fi
    done
fi

# =============================================================================
# Step 3: Compute DCTD (Dynamic Code Trace Divergence)
# =============================================================================
if [[ "$SCTD_ONLY" != true ]]; then
    echo ""
    echo "=== Step 3: Computing DCTD ==="
    
    for DATASET in "${DATASET_ARRAY[@]}"; do
        echo "  Processing $DATASET..."
        mkdir -p "$METRICS_ROOT/dctd/$MODEL/$DATASET"
        
        # Original
        ORIG_READY="$METRICS_ROOT/ready/$MODEL/$DATASET/original.ready.jsonl"
        if [[ -f "$ORIG_READY" ]]; then
            echo "    - Original..."
            $PYTHON dynamic_stability/calculate_dctd.py \
                --dataset "$DATASET" \
                --ready_jsonl "$ORIG_READY" \
                --out_jsonl "$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_original.jsonl" \
                --workers "$WORKERS" \
                --timeout "$TIMEOUT"
        else
            echo "    - Original: SKIPPED (file not found)"
        fi
        
        # ISO
        ISO_READY="$METRICS_ROOT/ready/$MODEL/$DATASET/iso.ready.jsonl"
        if [[ -f "$ISO_READY" ]]; then
            echo "    - ISO..."
            $PYTHON dynamic_stability/calculate_dctd.py \
                --dataset "$DATASET" \
                --ready_jsonl "$ISO_READY" \
                --out_jsonl "$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_iso.jsonl" \
                --workers "$WORKERS" \
                --timeout "$TIMEOUT"
        else
            echo "    - ISO: SKIPPED (file not found)"
        fi
    done
fi

# =============================================================================
# Step 4: Compute Energy Shift (Merge Original vs ISO)
# =============================================================================
echo ""
echo "=== Step 4: Computing Energy Shift ==="

for DATASET in "${DATASET_ARRAY[@]}"; do
    echo "  Processing $DATASET..."
    
    # SCTD Shift
    if [[ "$DCTD_ONLY" != true ]]; then
        SCTD_ORIG="$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_original.jsonl"
        SCTD_ISO="$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_iso.jsonl"
        if [[ -f "$SCTD_ORIG" && -f "$SCTD_ISO" ]]; then
            echo "    - SCTD shift..."
            $PYTHON dynamic_stability/merge_shift.py \
                --metric_prefix sctd \
                --orig_jsonl "$SCTD_ORIG" \
                --iso_jsonl "$SCTD_ISO" \
                --out_jsonl "$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_shift_original_vs_iso.jsonl"
        fi
    fi
    
    # DCTD Shift
    if [[ "$SCTD_ONLY" != true ]]; then
        DCTD_ORIG="$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_original.jsonl"
        DCTD_ISO="$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_iso.jsonl"
        if [[ -f "$DCTD_ORIG" && -f "$DCTD_ISO" ]]; then
            echo "    - DCTD shift..."
            $PYTHON dynamic_stability/merge_shift.py \
                --metric_prefix dctd \
                --orig_jsonl "$DCTD_ORIG" \
                --iso_jsonl "$DCTD_ISO" \
                --out_jsonl "$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_shift_original_vs_iso.jsonl"
        fi
    fi
done

# =============================================================================
# Step 5: Generate Plots
# =============================================================================
if [[ "$SKIP_PLOTS" != true ]]; then
    echo ""
    echo "=== Step 5: Generating Plots ==="
    mkdir -p "$METRICS_ROOT/plots"
    
    for DATASET in "${DATASET_ARRAY[@]}"; do
        echo "  Plotting $DATASET..."
        
        # SCTD plot
        if [[ "$DCTD_ONLY" != true ]]; then
            SCTD_SHIFT="$METRICS_ROOT/sctd/$MODEL/$DATASET/sctd_shift_original_vs_iso.jsonl"
            if [[ -f "$SCTD_SHIFT" ]]; then
                echo "    - SCTD plot..."
                $PYTHON metrics/plot_metric_shift.py \
                    --in_jsonl "$SCTD_SHIFT" \
                    --out_png "$METRICS_ROOT/plots/${MODEL}_${DATASET}_sctd_shift.png" \
                    --y_key sctd_jsd
            fi
        fi
        
        # DCTD plot
        if [[ "$SCTD_ONLY" != true ]]; then
            DCTD_SHIFT="$METRICS_ROOT/dctd/$MODEL/$DATASET/dctd_shift_original_vs_iso.jsonl"
            if [[ -f "$DCTD_SHIFT" ]]; then
                echo "    - DCTD plot..."
                $PYTHON metrics/plot_metric_shift.py \
                    --in_jsonl "$DCTD_SHIFT" \
                    --out_png "$METRICS_ROOT/plots/${MODEL}_${DATASET}_dctd_shift.png" \
                    --y_key dctd_jsd
            fi
        fi
    done
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Metrics Pipeline Complete!"
echo "=============================================="
echo ""
echo "Output locations:"
echo "  - Metric-ready data: $METRICS_ROOT/ready/$MODEL/<dataset>/"
if [[ "$DCTD_ONLY" != true ]]; then
    echo "  - SCTD results:      $METRICS_ROOT/sctd/$MODEL/<dataset>/"
fi
if [[ "$SCTD_ONLY" != true ]]; then
    echo "  - DCTD results:      $METRICS_ROOT/dctd/$MODEL/<dataset>/"
fi
if [[ "$SKIP_PLOTS" != true ]]; then
    echo "  - Plots:             $METRICS_ROOT/plots/"
fi
echo ""

# Print Pass@k summary if compute_pass_at_k.py exists
if [[ -f "./generation_and_testing/compute_pass_at_k.py" ]]; then
    echo "=== Pass@k Summary ==="
    for DATASET in "${DATASET_ARRAY[@]}"; do
        echo ""
        echo "--- $DATASET ---"
        
        # Original
        ORIG_UT="$RESULTS_ROOT/$MODEL/$DATASET/unittests"
        if [[ -d "$ORIG_UT" ]]; then
            for f in "$ORIG_UT"/*.jsonl; do
                if [[ -f "$f" ]] && [[ ! "$f" == *".OLD."* ]]; then
                    echo "Original:"
                    $PYTHON ./generation_and_testing/compute_pass_at_k.py "$f" 2>/dev/null | grep -E "Pass@|Problems" || true
                    break
                fi
            done
        fi
        
        # ISO
        ISO_UT="$RESULTS_ROOT/$MODEL/${DATASET}_iso/unittests"
        if [[ -d "$ISO_UT" ]]; then
            for f in "$ISO_UT"/*.jsonl; do
                if [[ -f "$f" ]] && [[ ! "$f" == *".OLD."* ]]; then
                    echo "ISO:"
                    $PYTHON ./generation_and_testing/compute_pass_at_k.py "$f" 2>/dev/null | grep -E "Pass@|Problems" || true
                    break
                fi
            done
        fi
    done
fi

echo ""
echo "Done!"
