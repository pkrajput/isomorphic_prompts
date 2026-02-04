#!/usr/bin/env bash
#
# pipeline_generate_and_test.sh
#
# Convenience wrapper to run generation and unittests for code models.
# Part 2 of the I/O Isomorphism Pipeline.
#
# Supports:
#   - Gemini models (API-based): gemini-2.0-flash, gemini-2.5-pro
#   - HuggingFace models (local GPU): starcoder-15b, codestral-22b
#

set -e

# Default values
DATASET=""
MODEL=""
TEMPERATURE="0.0"
N_GENERATIONS="5"
WORKERS="32"
BATCH_SIZE=""
MAX_PROBLEMS=""
CONFIG="config.yaml"
RESUME_FLAG=""
DATASETS_ROOT="./datasets"
OUT_DIR="./results"
ORIGINAL_FLAG=""

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
Usage: $0 --dataset <dataset> --model <model> [OPTIONS]

Run generation and unittests for code models on original or ISO datasets.

Required:
  --dataset         Dataset to process (bigobench, effibench, or mbpp)
  --model           Model to use for generation:
                    - Gemini (API): gemini-2.0-flash, gemini-2.5-pro
                    - HuggingFace (GPU): starcoder-15b, codestral-22b

Options:
  --temperature     LLM temperature (default: 0.0)
  --n_generations   Completions per problem (default: 5)
  --workers         Parallel workers for Gemini (default: 32)
  --batch_size      Batch size for HuggingFace (default: model-specific)
  --max_problems    Max problems to process (default: all)
  --config          Config YAML path (default: config.yaml)
  --resume          Resume from existing output
  --original        Use original datasets (default: ISO)
  --datasets_root   Datasets directory (default: ./datasets)
  --out_dir         Output directory (default: ./results)
  -h, --help        Show this help

Examples:
  # Generate with Gemini on BigOBench ISO
  $0 --dataset bigobench --model gemini-2.0-flash --resume

  # Generate with StarCoder on MBPP original
  $0 --dataset mbpp --model starcoder-15b --original --batch_size 4

  # Generate with Codestral on EffiBench ISO
  $0 --dataset effibench --model codestral-22b --n_generations 5
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --n_generations)
            N_GENERATIONS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_problems)
            MAX_PROBLEMS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --original)
            ORIGINAL_FLAG="--original"
            shift
            ;;
        --datasets_root)
            DATASETS_ROOT="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
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
if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required"
    usage
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    usage
    exit 1
fi

if [[ "$DATASET" != "bigobench" && "$DATASET" != "effibench" && "$DATASET" != "mbpp" ]]; then
    echo "ERROR: dataset must be 'bigobench', 'effibench', or 'mbpp'"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build generation args
GEN_ARGS="--dataset $DATASET --model $MODEL --datasets_root $DATASETS_ROOT --out_dir $OUT_DIR"
GEN_ARGS="$GEN_ARGS --temperature $TEMPERATURE --n_generations $N_GENERATIONS"
GEN_ARGS="$GEN_ARGS --workers $WORKERS --config $CONFIG"

if [[ -n "$BATCH_SIZE" ]]; then
    GEN_ARGS="$GEN_ARGS --batch_size $BATCH_SIZE"
fi

if [[ -n "$MAX_PROBLEMS" ]]; then
    GEN_ARGS="$GEN_ARGS --max_problems $MAX_PROBLEMS"
fi

if [[ -n "$RESUME_FLAG" ]]; then
    GEN_ARGS="$GEN_ARGS $RESUME_FLAG"
fi

if [[ -n "$ORIGINAL_FLAG" ]]; then
    GEN_ARGS="$GEN_ARGS $ORIGINAL_FLAG"
fi

# Determine dataset subdirectory
if [[ -n "$ORIGINAL_FLAG" ]]; then
    # Original datasets use different naming
    case "$DATASET" in
        bigobench) ISO_DIR="bigobench" ;;
        effibench) ISO_DIR="effibench" ;;
        mbpp) ISO_DIR="mbpp" ;;
    esac
else
    ISO_DIR="${DATASET}_iso"
fi

echo "=========================================="
echo "Part 2: Generation + Unittests Pipeline"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Mode: ${ORIGINAL_FLAG:-ISO}"
echo "Temperature: $TEMPERATURE"
echo "N generations: $N_GENERATIONS"
echo "Max problems: ${MAX_PROBLEMS:-all}"
echo ""

# Step 1: Generate
echo "[1/2] Running generation..."
$PYTHON "$SCRIPT_DIR/gen_models.py" $GEN_ARGS

# Step 2: Find generated file and run unittests
GEN_DIR="$OUT_DIR/$MODEL/$ISO_DIR/generations"

if [[ -d "$GEN_DIR" ]]; then
    for GEN_FILE in "$GEN_DIR"/*.jsonl; do
        if [[ -f "$GEN_FILE" ]]; then
            BASENAME=$(basename "$GEN_FILE" .jsonl)
            UNITTEST_FILE="$OUT_DIR/$MODEL/$ISO_DIR/unittests/${BASENAME}__unittest.jsonl"
            
            echo ""
            echo "[2/2] Running unittests for: $(basename "$GEN_FILE")"
            
            mkdir -p "$(dirname "$UNITTEST_FILE")"
            
            UNITTEST_ARGS="--dataset $DATASET --generations_jsonl $GEN_FILE"
            UNITTEST_ARGS="$UNITTEST_ARGS --out_jsonl $UNITTEST_FILE"
            UNITTEST_ARGS="$UNITTEST_ARGS --datasets_root $DATASETS_ROOT"
            UNITTEST_ARGS="$UNITTEST_ARGS --workers $WORKERS --timeout 10"
            
            if [[ -n "$RESUME_FLAG" ]]; then
                UNITTEST_ARGS="$UNITTEST_ARGS $RESUME_FLAG"
            fi
            
            $PYTHON "$SCRIPT_DIR/run_unittests.py" $UNITTEST_ARGS
        fi
    done
else
    echo "WARNING: No generation output found at $GEN_DIR"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Generations: $OUT_DIR/$MODEL/$ISO_DIR/generations/"
echo "  - Unittests:   $OUT_DIR/$MODEL/$ISO_DIR/unittests/"
