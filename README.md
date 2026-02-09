# Invertible I/O-Isomorphism Pipeline

This directory contains scripts and utilities for applying **semantics-preserving, invertible I/O transformations** (isomorphisms) to code benchmark datasets.

## Purpose

Modern code models may exhibit **representation sensitivity**—producing correct solutions that behave differently (execution traces, resource usage) when given semantically equivalent but representationally different inputs. This pipeline:

1. **Transforms test I/O** using invertible bijections (e.g., affine integer encoding)
2. **Appends encoding contracts** to prompts describing the transformation
3. **Verifies invertibility** via roundtrip checks on every row
4. **Preserves semantics** so correctness is unchanged under the transformation

## Key Insight

> Accuracy can hide brittleness: same correctness, different execution strategy and resource envelope.

By measuring execution phenotype drift (DCTD/DMPD) across isomorphic representations, we can detect hidden representation sensitivity that standard pass@k metrics miss.

---

## Directory Structure

```
.
├── change_prompts/               # Part 1: I/O isomorphism transforms
│   ├── adapters/                 # Dataset-specific adapters
│   ├── apply_isomorphisms.py     # Main driver script
│   ├── contracts.py              # Contract generation utilities
│   └── iso_transforms.py         # Library of invertible transforms
├── datasets/                     # Input/output datasets
│   ├── BigOBench/                # Original BigOBench
│   ├── Effibench/                # Original EffiBench
│   ├── mbpp/                     # Original MBPP
│   ├── bigobench_iso/            # ISO-transformed BigOBench
│   ├── bigobench_iso_simple/     # ISO-simple (oracle-help) BigOBench
│   ├── effibench_iso/            # ISO-transformed EffiBench
│   ├── effibench_iso_simple/     # ISO-simple (oracle-help) EffiBench
│   ├── mbpp_iso/                 # ISO-transformed MBPP
│   └── mbpp_iso_simple/          # ISO-simple (oracle-help) MBPP
├── generation_and_testing/       # Part 2: Generation + Unittests
│   ├── gen_models.py             # Multi-backend generation script
│   ├── run_unittests.py          # Dataset-specific unittests
│   ├── compute_pass_at_k.py      # Pass@k computation
│   └── pipeline_generate_and_test.sh  # Generation pipeline
├── dynamic_stability/            # Part 3: Stability metrics
│   ├── calculate_sctd.py         # Static Code Trace Divergence
│   ├── calculate_dctd.py         # Dynamic Code Trace Divergence
│   ├── merge_shift.py            # Energy shift computation
│   ├── metric_ready.py           # Prepare metric-ready data
│   └── run_metrics.sh            # Metrics pipeline script
├── metrics/                      # Metrics output directory
│   ├── ready/                    # Metric-ready JSONL files
│   ├── sctd/                     # SCTD results
│   ├── dctd/                     # DCTD results
│   └── plots/                    # Visualization plots
├── results/                      # Generation/unittest results
│   └── <model>/<dataset>/        # Model-specific outputs
├── unittests/                    # Dataset-specific unittest harnesses
└── config.yaml                   # API keys configuration
```

---

## Supported Models

### API-Based Models (Gemini)
| Model | Backend | Default Workers | Notes |
|-------|---------|-----------------|-------|
| `gemini-2.0-flash` | Gemini API | 50 | Fast, high-throughput |
| `gemini-2.5-pro` | Gemini API | 8 | Lower rate limits |

### Local GPU Models (HuggingFace)
| Model | Backend | HF Model ID | Batch Size | Notes |
|-------|---------|-------------|------------|-------|
| `starcoder2-15b` | HuggingFace | `bigcode/starcoder2-15b` | 4 | Completion model (no chat template) |
| `codestral-22b` | HuggingFace | `mistralai/Codestral-22B-v0.1` | 2 | Chat model with template |

---

## Quick Start: Full Pipeline

### Prerequisites

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually:
# Core: pip install pyyaml tqdm
# Gemini: pip install google-generativeai
# HuggingFace (local GPU): pip install transformers torch accelerate
```

### Configuration

Create `config.yaml` in the project root:

```yaml
# Gemini API key (required for Gemini models)
api_key: "your-gemini-api-key"

# HuggingFace token (optional, for gated models like Codestral)
hf_token: "your-hf-token"
```

---

## Running the Pipeline

### Part 1: Apply I/O Isomorphisms (Already Done)

Isomorphic datasets are pre-generated in `./datasets/*_iso/`.
Oracle-help ablations are pre-generated in `./datasets/*_iso_simple/`.

```bash
# To regenerate (optional):
python change_prompts/apply_isomorphisms.py \
  --dataset bigobench \
  --datasets_root ./datasets \
  --out_root ./datasets/bigobench_iso \
  --iso_family affine_int \
  --seed 42

# ISO-simple (oracle-help inputs)
python change_prompts/apply_isomorphisms.py \
  --dataset bigobench \
  --datasets_root ./datasets \
  --out_root ./datasets/bigobench_iso_simple \
  --iso_family affine_int \
  --seed 42 \
  --oracle_help_inputs
```

### Part 2: Generation + Unittests

#### Single Dataset, Single Model

```bash
# BigOBench with Gemini 2.0 Flash (ISO)
./generation_and_testing/pipeline_generate_and_test.sh \
  --dataset bigobench \
  --model gemini-2.0-flash \
  --n_generations 5 \
  --resume

# BigOBench with Gemini 2.0 Flash (ISO-simple / oracle-help)
./generation_and_testing/pipeline_generate_and_test.sh \
  --dataset bigobench \
  --model gemini-2.0-flash \
  --iso_simple \
  --n_generations 5 \
  --resume

# BigOBench with Gemini 2.0 Flash (Original)
./generation_and_testing/pipeline_generate_and_test.sh \
  --dataset bigobench \
  --model gemini-2.0-flash \
  --original \
  --n_generations 5 \
  --resume
```

#### HuggingFace Models (Local GPU)

```bash
# StarCoder2 15B on MBPP ISO
./generation_and_testing/pipeline_generate_and_test.sh \
  --dataset mbpp \
  --model starcoder2-15b \
  --batch_size 4 \
  --n_generations 5 \
  --resume

# Codestral 22B on EffiBench Original
./generation_and_testing/pipeline_generate_and_test.sh \
  --dataset effibench \
  --model codestral-22b \
  --original \
  --batch_size 2 \
  --n_generations 5 \
  --resume
```

#### Run All Three Datasets (Original + ISO)

```bash
# === Gemini 2.0 Flash ===
for DATASET in bigobench effibench mbpp; do
  # ISO
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model gemini-2.0-flash \
    --n_generations 5 \
    --resume
  
  # Original
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model gemini-2.0-flash \
    --original \
    --n_generations 5 \
    --resume
done

# === StarCoder2 15B ===
for DATASET in bigobench effibench mbpp; do
  # ISO
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model starcoder2-15b \
    --batch_size 4 \
    --n_generations 5 \
    --resume
  
  # Original
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model starcoder2-15b \
    --original \
    --batch_size 4 \
    --n_generations 5 \
    --resume
done

# === Codestral 22B ===
for DATASET in bigobench effibench mbpp; do
  # ISO
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model codestral-22b \
    --batch_size 2 \
    --n_generations 5 \
    --resume
  
  # Original
  ./generation_and_testing/pipeline_generate_and_test.sh \
    --dataset $DATASET \
    --model codestral-22b \
    --original \
    --batch_size 2 \
    --n_generations 5 \
    --resume
done
```

### Part 3: Compute Stability Metrics

#### Single Model, All Datasets

```bash
# Gemini 2.0 Flash
./dynamic_stability/run_metrics.sh \
  --model gemini-2.0-flash \
  --datasets bigobench,effibench,mbpp

# StarCoder2 15B
./dynamic_stability/run_metrics.sh \
  --model starcoder2-15b \
  --datasets bigobench,effibench,mbpp

# Codestral 22B
./dynamic_stability/run_metrics.sh \
  --model codestral-22b \
  --datasets bigobench,effibench,mbpp
```

#### Options

```bash
# Only SCTD (faster, no execution)
./dynamic_stability/run_metrics.sh \
  --model starcoder2-15b \
  --sctd_only

# Only DCTD (requires execution)
./dynamic_stability/run_metrics.sh \
  --model codestral-22b \
  --dctd_only \
  --workers 8 \
  --timeout 30

# Skip plot generation
./dynamic_stability/run_metrics.sh \
  --model gemini-2.0-flash \
  --skip_plots
```

---

## Direct Script Usage

### Generation (gen_models.py)

```bash
# Gemini generation
python3 ./generation_and_testing/gen_models.py \
  --dataset bigobench \
  --model gemini-2.0-flash \
  --n_generations 5 \
  --config config.yaml \
  --resume

# HuggingFace generation with batching
python3 ./generation_and_testing/gen_models.py \
  --dataset mbpp \
  --model starcoder2-15b \
  --n_generations 5 \
  --batch_size 4 \
  --config config.yaml \
  --resume

# ISO-simple (oracle-help) dataset
python3 ./generation_and_testing/gen_models.py \
  --dataset effibench \
  --model gemini-2.0-flash \
  --iso_simple \
  --n_generations 5 \
  --config config.yaml \
  --resume

# Original (non-ISO) dataset
python3 ./generation_and_testing/gen_models.py \
  --dataset effibench \
  --model codestral-22b \
  --original \
  --n_generations 5 \
  --batch_size 2 \
  --config config.yaml
```

### Unittests

```bash
python3 ./generation_and_testing/run_unittests.py \
  --dataset bigobench \
  --generations_jsonl ./results/starcoder2-15b/bigobench_iso/generations/<FILE>.jsonl \
  --out_jsonl ./results/starcoder2-15b/bigobench_iso/unittests/<FILE>__unittest.jsonl \
  --workers 16 \
  --timeout 10 \
  --resume
```

### Pass@k Computation

```bash
python3 ./generation_and_testing/compute_pass_at_k.py \
  ./results/<model>/<dataset>/unittests/<file>.jsonl
```

### SCTD/DCTD Individual Steps

```bash
# 1. Prepare metric-ready data
python3 dynamic_stability/metric_ready.py \
  --results_root ./results \
  --out_root ./metrics/ready \
  --datasets bigobench,effibench,mbpp \
  --models starcoder2-15b \
  --rewrite_marker

# 2. Compute SCTD
python3 dynamic_stability/calculate_sctd.py \
  --ready_jsonl ./metrics/ready/starcoder2-15b/bigobench/original.ready.jsonl \
  --out_jsonl ./metrics/sctd/starcoder2-15b/bigobench/sctd_original.jsonl \
  --store_centroid

# 3. Compute DCTD
python3 dynamic_stability/calculate_dctd.py \
  --dataset bigobench \
  --ready_jsonl ./metrics/ready/starcoder2-15b/bigobench/original.ready.jsonl \
  --out_jsonl ./metrics/dctd/starcoder2-15b/bigobench/dctd_original.jsonl \
  --workers 16 \
  --timeout 15

# 4. Merge and compute energy shift
python3 dynamic_stability/merge_shift.py \
  --metric_prefix sctd \
  --orig_jsonl ./metrics/sctd/starcoder2-15b/bigobench/sctd_original.jsonl \
  --iso_jsonl ./metrics/sctd/starcoder2-15b/bigobench/sctd_iso.jsonl \
  --out_jsonl ./metrics/sctd/starcoder2-15b/bigobench/sctd_shift_original_vs_iso.jsonl

# 5. Plot
python3 metrics/plot_metric_shift.py \
  --in_jsonl ./metrics/sctd/starcoder2-15b/bigobench/sctd_shift_original_vs_iso.jsonl \
  --out_png ./metrics/plots/starcoder2-15b_bigobench_sctd_shift.png \
  --y_key sctd_jsd
```

---

## Output Schema

### Generation JSONL

Each line contains:
- `dataset`, `model`, `temperature`, `n_generations`
- `problem_id`, `source_file`, `row_index`, `completion_index`
- `prompt_used`, `prompt_hash`
- `generated_solution`, `raw_llm_output`
- `finish_reason`, `error_message`, `latency_s`
- `is_original`, `iso_applied`, `iso_family`, `iso_seed`, `iso_params`
- `row_meta`, `generation_metadata`

### Unittest JSONL

Each line contains:
- `dataset`, `model`, `temperature`
- `problem_id`, `completion_index`, `source_file`, `row_index`
- `status` (`success`, `test_failure`, `execution_error`, `timeout`, `error`)
- `pass` (boolean)
- `error`, `time_s`, `timeout`
- `iso_family`, `iso_seed`, `iso_params`

### Join Key

Generation and unittest JSONLs are joinable by:
```
(dataset, model, temperature, problem_id, completion_index)
```

---

## Experimental Results

### Gemini 2.0 Flash (temp=0.0, n=5)

| Dataset | Condition | Problems | Pass@1 | Pass@5 |
|---------|-----------|----------|--------|--------|
| **BigOBench** | Original | 311 | **66.1%** | **68.7%** |
| **BigOBench** | ISO | 311 | 50.3% | 53.4% |
| **EffiBench** | Original | 338 | **27.3%** | **56.4%** |
| **EffiBench** | ISO | 338 | 29.3% | 32.5% |
| **MBPP** | Original | 257 | **80.8%** | **81.7%** |
| **MBPP** | ISO | 257 | 58.1% | 61.1% |

### Energy Distance Results (Gemini 2.0 Flash)

Quantifies the distribution shift magnitude between Original and ISO representations. Higher values indicate larger phenotype drift.

| Dataset | Metric | Energy Distance (ED) |
|---------|--------|----------------------|
| **BigOBench** | SCTD | 0.0060 |
| **BigOBench** | DCTD | 0.0144 |
| **EffiBench** | SCTD | N/A (check plot) |
| **EffiBench** | DCTD | 0.0000 |
| **MBPP** | SCTD | 0.0105 |
| **MBPP** | DCTD | 0.0026 |

---

## Transform Families

| Family | Description | Use Case |
|--------|-------------|----------|
| `affine_int` | `x' = a*x + b` for integers | Universal, low overhead |
| `label_permutation` | Bijective label remapping | Discrete tokens (A→C, B→A, etc.) |
| `basek_int` | Integer → base-16/36 string | When string encoding is acceptable |
| `delimiter_rewrite` | Change delimiters (space→comma) | Structured output formatting |
| `identity` | No transformation (control) | Baseline comparison |

---

## Requirements

- Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

Or individually:
- `pyyaml`, `tqdm` (core dependencies)
- `google-generativeai` (for Gemini models)
- `transformers>=4.36.0`, `torch>=2.0.0`, `accelerate` (for HuggingFace models)
- `scipy`, `numpy` (for metrics)
- `matplotlib` (for plotting)

---

## NeurIPS Framing

This work contributes:

1. **A principled robustness probe** beyond paraphrases: invertible I/O isomorphisms + identity control
2. **A measurable "invariance gap"**: between-condition drift vs within-condition spread
3. **Practical mitigations**: canonical I/O contracts, schema normalization, augmentation strategies

The target is **representation-invariant execution**—stable algorithmic behavior under equivalent representations, up to predictable encode/decode overhead.
