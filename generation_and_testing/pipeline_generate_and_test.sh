#!/usr/bin/env bash
#
# pipeline_generate_and_test.sh
#
# Wrapper: generation + unittests for all configured models.
# Supports Gemini API, OpenRouter, and HuggingFace backends.
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
ISO_DEC_ONLY_FLAG=""
ISO_ENC_ONLY_FLAG=""
FUZZ_TAG=""
ISO_FAMILY="affine_int"

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
  --model           Model key from gen_models.py MODEL_CONFIGS

Options:
  --temperature     LLM temperature (default: 0.0)
  --n_generations   Completions per problem (default: 5)
  --workers         Parallel workers for Gemini (default: 32)
  --batch_size      Batch size for HuggingFace (default: model-specific)
  --max_problems    Max problems to process (default: all)
  --config          Config YAML path (default: config.yaml)
  --resume          Resume from existing output
  --original        Use original datasets (default: ISO)
  --iso_dec_only    ISO (dec only): oracle inputs, encoded outputs
  --iso_enc_only    ISO (enc only): encoded inputs, original outputs
  --fuzz TAG        Fuzzed variant: synonym_20, synonym_40, deadcode
  --iso_family      Isomorphism family: affine_int, base_conv, cubic_int (default: affine_int)
  --datasets_root   Datasets directory (default: ./datasets)
  --out_dir         Output directory (default: ./results)
  -h, --help        Show this help

Examples:
  # Generate with Gemini on BigOBench ISO (affine, default)
  $0 --dataset bigobench --model gemini-2.0-flash --resume

  # Generate with Gemini on BigOBench ISO (base conversion)
  $0 --dataset bigobench --model gemini-2.0-flash --iso_family base_conv --resume

  # Generate with Gemini on BigOBench ISO (cubic)
  $0 --dataset bigobench --model gemini-2.0-flash --iso_family cubic_int --resume

  # Generate with StarCoder on MBPP original
  $0 --dataset mbpp --model starcoder2-15b --original --batch_size 4

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
        --iso_dec_only)
            ISO_DEC_ONLY_FLAG="--iso_dec_only"
            shift
            ;;
        --iso_enc_only)
            ISO_ENC_ONLY_FLAG="--iso_enc_only"
            shift
            ;;
        --fuzz)
            FUZZ_TAG="$2"
            shift 2
            ;;
        --iso_family)
            ISO_FAMILY="$2"
            shift 2
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

MODE_COUNT=0
[[ -n "$ORIGINAL_FLAG" ]] && ((MODE_COUNT++))
[[ -n "$ISO_DEC_ONLY_FLAG" ]] && ((MODE_COUNT++))
[[ -n "$ISO_ENC_ONLY_FLAG" ]] && ((MODE_COUNT++))
[[ -n "$FUZZ_TAG" ]] && ((MODE_COUNT++))
if [[ "$MODE_COUNT" -gt 1 ]]; then
    echo "ERROR: --original, --iso_dec_only, --iso_enc_only, and --fuzz are mutually exclusive"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build generation args
GEN_ARGS="--dataset $DATASET --model $MODEL --datasets_root $DATASETS_ROOT --out_dir $OUT_DIR"
GEN_ARGS="$GEN_ARGS --temperature $TEMPERATURE --n_generations $N_GENERATIONS"
GEN_ARGS="$GEN_ARGS --workers $WORKERS --config $CONFIG --iso_family $ISO_FAMILY"

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

if [[ -n "$ISO_DEC_ONLY_FLAG" ]]; then
    GEN_ARGS="$GEN_ARGS $ISO_DEC_ONLY_FLAG"
fi

if [[ -n "$ISO_ENC_ONLY_FLAG" ]]; then
    GEN_ARGS="$GEN_ARGS $ISO_ENC_ONLY_FLAG"
fi

if [[ -n "$FUZZ_TAG" ]]; then
    GEN_ARGS="$GEN_ARGS --fuzz $FUZZ_TAG"
fi

# Map dataset to its canonical directory name
case "$DATASET" in
    bigobench) DS_DIR="BigOBench" ;;
    effibench) DS_DIR="Effibench" ;;
    mbpp) DS_DIR="mbpp" ;;
esac

# Map iso_family to short folder name
case "$ISO_FAMILY" in
    affine_int) FAMILY_SHORT="affine" ;;
    base_conv|basek_int) FAMILY_SHORT="base_conv" ;;
    cubic_int) FAMILY_SHORT="cubic" ;;
    *) FAMILY_SHORT="$ISO_FAMILY" ;;
esac

# Determine dataset subdirectory (for results output)
if [[ -n "$ORIGINAL_FLAG" ]]; then
    ISO_DIR="${DATASET}"
elif [[ -n "$ISO_DEC_ONLY_FLAG" ]]; then
    ISO_DIR="${DATASET}_iso_${FAMILY_SHORT}_oracle"
elif [[ -n "$ISO_ENC_ONLY_FLAG" ]]; then
    ISO_DIR="${DATASET}_iso_${FAMILY_SHORT}_encode_only"
elif [[ -n "$FUZZ_TAG" ]]; then
    ISO_DIR="${DATASET}_${FUZZ_TAG}"
else
    ISO_DIR="${DATASET}_iso_${FAMILY_SHORT}"
fi

echo "=========================================="
echo "Part 2: Generation + Unittests Pipeline"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
if [[ -n "$ORIGINAL_FLAG" ]]; then
    MODE_STR="ORIGINAL"
elif [[ -n "$ISO_DEC_ONLY_FLAG" ]]; then
    MODE_STR="ISO (dec only)"
elif [[ -n "$ISO_ENC_ONLY_FLAG" ]]; then
    MODE_STR="ISO (enc only)"
elif [[ -n "$FUZZ_TAG" ]]; then
    MODE_STR="FUZZ: $FUZZ_TAG"
else
    MODE_STR="ISO"
fi
echo "Mode: $MODE_STR"
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
