#!/usr/bin/env python3
"""
Run CoDeC with flexible model and dataset selection.

Usage:
  python run_codec_scale_confirmation.py --models all --seen-datasets "" --unseen-datasets livecodebench_v5
  python run_codec_scale_confirmation.py --models pythia-410m,davinci-002 --seen-datasets hackernews,wikipedia --unseen-datasets gpqa_diamond,livecodebench_v5
  python run_codec_scale_confirmation.py --list-options   # Show available models and datasets

Models: pythia-410m, pythia-1.4b, pythia-12b, davinci-002, babbage-002, llama-3.1-70b, llama-3.1-405b, nemotron-4-340b (or "all")
Seen (Pile): hackernews, wikipedia, github, arxiv, dm_mathematics, pile_cc, pubmed_central
Unseen: gsm8k, gpqa_diamond, ifeval, humaneval, frames, aime_2024, aime_2025, livecodebench_v1, livecodebench_v5, math_500, ...
"""
import argparse
import os
import sys
from datetime import datetime

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from experiment import (
    run_experiment_five,
    _seen_source_for_experiment0_model,
)

MODEL_CONFIGS = [
    ("pythia-410m", "EleutherAI/pythia-410m", "Pythia 410M", True),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", "Pythia 1.4B", True),
    ("pythia-12b", "EleutherAI/pythia-12b", "Pythia 12B", True),
    ("davinci-002", "davinci-002", "OpenAI Davinci 002", False),
    ("babbage-002", "babbage-002", "OpenAI Babbage 002", False),
    ("llama-3.1-70b", "openrouter:meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B (OpenRouter)", False),
    ("llama-3.1-405b", "openrouter:meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B (OpenRouter)", False),
    ("nemotron-4-340b", "openrouter:nvidia/nemotron-4-340b-instruct", "Nemotron-4 340B (OpenRouter)", False),
]
MODEL_KEYS = [c[0] for c in MODEL_CONFIGS]
SEEN_KEYS = ["hackernews", "wikipedia", "github", "arxiv", "dm_mathematics", "pile_cc", "pubmed_central"]
SEEN_KEYS_BY_SOURCE = {
    "pile": ["hackernews", "wikipedia", "github", "arxiv", "dm_mathematics", "pile_cc", "pubmed_central"],
    "dolma": ["c4", "wikipedia", "pes2o_v2", "reddit_v5", "commoncrawl_head", "commoncrawl_middle", "commoncrawl_tail"],
    "nemotron_cc": ["wikipedia", "arxiv", "cc_2024_high", "cc_2020_high", "cc_2020_low", "cc_2019_medium", "cc_2016_medium", "cc_2014_low", "cc_2013_high"],
}
UNSEEN_KEYS = ["gsm8k", "gpqa_diamond", "ifeval", "humaneval", "frames", "aime_2024", "aime_2025",
               "livecodebench_v1", "livecodebench_v5", "bfcl_v3", "bbq", "rewardbench_v1", "rewardbench_v2",
               "math_500", "gutenberg_colonial", "gutenberg_jibby", "gutenberg_corbin", "global_news", "amazon_reviews_2023"]
UNSEEN_KEYS_BY_SOURCE = {
    "pile": ["gsm8k", "gpqa_diamond", "ifeval", "humaneval", "frames", "aime_2024", "aime_2025",
             "livecodebench_v1", "livecodebench_v5", "bfcl_v3", "bbq", "rewardbench_v1", "rewardbench_v2",
             "math_500", "gutenberg_colonial", "gutenberg_jibby", "gutenberg_corbin", "global_news", "amazon_reviews_2023"],
    "dolma": ["frames", "aime_2024", "aime_2025", "livecodebench_v5", "bfcl_v3", "rewardbench_v1", "rewardbench_v2",
              "gutenberg_colonial", "gutenberg_jibby", "gutenberg_corbin", "global_news", "amazon_reviews_2023"],
    "nemotron_cc": ["aime_2025", "gpqa_diamond", "livecodebench_v5"],
}


def _cutoff_for_model(model_key):
    # OpenAI davinci-002, babbage-002: released Oct 2023 — use release date as cutoff
    if model_key in ("davinci-002", "babbage-002"):
        return datetime(2023, 10, 1)
    # Llama 3.1 family data freshness cutoff: Dec 2023
    if model_key in ("llama-3.1-70b", "llama-3.1-405b"):
        return datetime(2023, 12, 1)
    # Nemotron-4 340B pretraining data freshness cutoff: June 2023
    if model_key == "nemotron-4-340b":
        return datetime(2023, 6, 1)
    return datetime(2022, 1, 1)  # Pythia (Pile)


def main():
    parser = argparse.ArgumentParser(description="CoDeC: flexible model and dataset selection")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model keys or 'all' (default: all)")
    parser.add_argument("--seen-datasets", type=str, default="",
                        help="Comma-separated seen keys (Pile). Empty = skip seen. e.g. hackernews,wikipedia")
    parser.add_argument("--unseen-datasets", type=str, default="livecodebench_v5",
                        help="Comma-separated unseen keys. e.g. livecodebench_v5,gpqa_diamond")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per dataset (100=fast, 1000=paper)")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel workers for vLLM scoring")
    parser.add_argument("--use-vllm-for-pythia", action="store_true",
                        help="Use vLLM endpoint for Pythia models instead of local HF")
    parser.add_argument("--use-vllm-for-llama", action="store_true",
                        help="Use vLLM endpoint for llama-3.1-70b / llama-3.1-405b instead of OpenRouter")
    parser.add_argument("--llama70b-vllm-model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="HF model id served by vLLM for llama-3.1-70b")
    parser.add_argument("--llama-vllm-model", type=str, default="nvidia/Llama-3.1-405B-Instruct-NVFP4",
                        help="HF model id served by vLLM for llama-3.1-405b")
    parser.add_argument("--use-vllm-for-nemotron", action="store_true",
                        help="Use vLLM endpoint for nemotron-4-340b instead of OpenRouter")
    parser.add_argument("--nemotron-vllm-model", type=str, default="mgoin/Nemotron-4-340B-Instruct-hf-FP8",
                        help="HF model id served by vLLM for nemotron-4-340b")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint (default: http://localhost:8000/v1)")
    parser.add_argument("--save-results", action="store_true", default=True,
                        help="Save per-dataset results (default: True)")
    parser.add_argument("--no-save-results", action="store_false", dest="save_results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for saved results")
    parser.add_argument("--list-options", action="store_true", help="Print available models and datasets, then exit")
    args = parser.parse_args()

    if args.list_options:
        print("Models:", ", ".join(MODEL_KEYS))
        print("Seen (pile):", ", ".join(SEEN_KEYS_BY_SOURCE["pile"]))
        print("Seen (dolma):", ", ".join(SEEN_KEYS_BY_SOURCE["dolma"]))
        print("Seen (nemotron_cc):", ", ".join(SEEN_KEYS_BY_SOURCE["nemotron_cc"]))
        print("Unseen (pile):", ", ".join(UNSEEN_KEYS_BY_SOURCE["pile"]))
        print("Unseen (dolma):", ", ".join(UNSEEN_KEYS_BY_SOURCE["dolma"]))
        print("Unseen (nemotron_cc):", ", ".join(UNSEEN_KEYS_BY_SOURCE["nemotron_cc"]))
        return

    # Parse models
    model_req = [k.strip() for k in args.models.split(",") if k.strip()]
    if not model_req or "all" in model_req:
        configs = list(MODEL_CONFIGS)
    else:
        key_set = set(model_req)
        configs = [c for c in MODEL_CONFIGS if c[0] in key_set]
        unknown = key_set - {c[0] for c in MODEL_CONFIGS}
        if unknown:
            print(f"Unknown models: {unknown}. Use --list-options for valid keys.")
            return

    # Parse seen/unseen datasets
    seen_keys = [k.strip() for k in args.seen_datasets.split(",") if k.strip()] if args.seen_datasets else []
    unseen_keys = [k.strip() for k in args.unseen_datasets.split(",") if k.strip()] if args.unseen_datasets else []
    if not seen_keys and not unseen_keys:
        print("At least one dataset side required: set --seen-datasets and/or --unseen-datasets.")
        return

    seen_only = len(seen_keys) > 0 and len(unseen_keys) == 0
    unseen_only = len(seen_keys) == 0
    mode_label = "seen-only" if seen_only else ("unseen-only" if unseen_only else "seen+unseen")
    print(f"CoDeC ({mode_label}): models={[c[0] for c in configs]}, seen={seen_keys or 'none'}, unseen={unseen_keys or 'none'}, n_samples={args.n_samples}")
    if any(c[0] in ("davinci-002", "babbage-002") for c in configs) and not utils.OPENAI_API_KEY:
        print("WARNING: Set OPENAI_API_KEY for OpenAI models: export OPENAI_API_KEY=sk-...")
    need_openrouter = any(
        c[1].startswith("openrouter:") and not (
            (c[0] == "llama-3.1-405b" and args.use_vllm_for_llama) or
            (c[0] == "llama-3.1-70b" and args.use_vllm_for_llama) or
            (c[0] == "nemotron-4-340b" and args.use_vllm_for_nemotron)
        )
        for c in configs
    )
    if need_openrouter and not utils.OPENROUTER_API_KEY:
        print("WARNING: Set OPENROUTER_API_KEY for OpenRouter models: export OPENROUTER_API_KEY=sk-or-...")

    all_results = []

    for model_key, actual_model_name, display_name, use_no_vllm in configs:
        cutoff_date = _cutoff_for_model(model_key)
        if model_key in ("davinci-002", "babbage-002") and not utils.OPENAI_API_KEY:
            print(f"Skipping {display_name} (no OPENAI_API_KEY)")
            continue
        use_vllm_override = (
            (args.use_vllm_for_llama and model_key == "llama-3.1-405b")
            or (args.use_vllm_for_llama and model_key == "llama-3.1-70b")
            or (args.use_vllm_for_nemotron and model_key == "nemotron-4-340b")
        )
        if actual_model_name.startswith("openrouter:") and not utils.OPENROUTER_API_KEY and not use_vllm_override:
            print(f"Skipping {display_name} (no OPENROUTER_API_KEY)")
            continue

        print("\n" + "=" * 70)
        print(f"Running CoDeC for: {display_name}")
        print("=" * 70)

        try:
            seen_source = _seen_source_for_experiment0_model(model_key)
            if seen_keys:
                allowed_seen_keys = SEEN_KEYS_BY_SOURCE.get(seen_source, SEEN_KEYS_BY_SOURCE["pile"])
                invalid_seen = [k for k in seen_keys if k not in allowed_seen_keys]
                valid_seen = [k for k in seen_keys if k in allowed_seen_keys]
                if invalid_seen:
                    print(
                        f"WARNING [{model_key}]: ignoring invalid seen dataset keys for source '{seen_source}': {invalid_seen}. "
                        f"Allowed: {allowed_seen_keys}"
                    )
                if not valid_seen:
                    print(f"WARNING [{model_key}]: no valid seen dataset keys remain -> running unseen only.")
            else:
                valid_seen = []

            if seen_only:
                valid_unseen = []
            else:
                allowed_unseen_keys = UNSEEN_KEYS_BY_SOURCE.get(seen_source, UNSEEN_KEYS_BY_SOURCE["pile"])
                invalid_unseen = [k for k in unseen_keys if k not in allowed_unseen_keys]
                valid_unseen = [k for k in unseen_keys if k in allowed_unseen_keys]
                if invalid_unseen:
                    print(
                        f"WARNING [{model_key}]: ignoring invalid unseen dataset keys for source '{seen_source}': {invalid_unseen}. "
                        f"Allowed: {allowed_unseen_keys}"
                    )
                if not valid_unseen:
                    print(f"WARNING [{model_key}]: no valid unseen dataset keys remain -> skipping model.")
                    continue

            use_vllm = (args.use_vllm_for_pythia and model_key.startswith("pythia-")) or (
                args.use_vllm_for_llama and model_key == "llama-3.1-405b"
            ) or (
                args.use_vllm_for_llama and model_key == "llama-3.1-70b"
            ) or (
                args.use_vllm_for_nemotron and model_key == "nemotron-4-340b"
            )
            if use_vllm:
                if model_key == "llama-3.1-405b":
                    vllm_target = args.llama_vllm_model
                elif model_key == "llama-3.1-70b":
                    vllm_target = args.llama70b_vllm_model
                elif model_key == "nemotron-4-340b":
                    vllm_target = args.nemotron_vllm_model
                else:
                    vllm_target = actual_model_name
                model = utils.setup_model(f"vllm:{vllm_target}", vllm_base_url=args.vllm_url)
                pw = args.parallel
                run_model_name = f"vllm:{vllm_target}"
            else:
                model = utils.setup_model(actual_model_name)
                pw = 0
                run_model_name = actual_model_name

            auc, results_score_data = run_experiment_five(
                model,
                run_model_name,
                cutoff_date,
                n_samples=args.n_samples,
                skip_token_figures=True,
                seen_source=seen_source,
                parallel_workers=pw,
                save_results=args.save_results,
                results_dir=args.results_dir,
                seen_dataset_keys=valid_seen if seen_keys else None,
                unseen_dataset_keys=valid_unseen if not seen_only else [],
                unseen_only=(unseen_only or (seen_keys and not valid_seen)),
            )

            seen_scores = [r["Score"] * 100 for r in results_score_data if "Seen" in r.get("Type", "")]
            unseen_scores = [r["Score"] * 100 for r in results_score_data if r.get("Type") == "Unseen"]
            seen_min = min(seen_scores) if seen_scores else None
            unseen_max = max(unseen_scores) if unseen_scores else None
            gap = (seen_min - unseen_max) if (seen_min is not None and unseen_max is not None) else None

            all_results.append({
                "model": display_name,
                "model_key": model_key,
                "auc": auc * 100 if auc is not None else None,
                "seen_mean": sum(seen_scores) / len(seen_scores) if seen_scores else None,
                "seen_min": seen_min,
                "unseen_mean": sum(unseen_scores) / len(unseen_scores) if unseen_scores else None,
                "unseen_max": unseen_max,
                "separation_gap": gap,
                "results": results_score_data,
            })
            if auc is not None:
                print(f"\n>>> {display_name}: AUC = {auc*100:.1f}%")
            else:
                unseen_s = [r for r in results_score_data if r.get("Type") == "Unseen"]
                parts = [f"{r['Dataset']}={r['Score']*100:.1f}%" for r in unseen_s]
                print(f"\n>>> {display_name}: " + (", ".join(parts) if parts else "N/A"))

        except Exception as e:
            print(f"ERROR running {display_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model": display_name,
                "model_key": model_key,
                "auc": None,
                "seen_mean": None,
                "seen_min": None,
                "unseen_mean": None,
                "unseen_max": None,
                "separation_gap": None,
                "results": [],
                "error": str(e),
            })

    # Write CSV
    csv_path = "codec_scale_confirmation.csv"
    with open(csv_path, "w") as f:
        f.write("Model,AUC (%),Seen mean (%),Seen min (%),Unseen mean (%),Unseen max (%),Separation gap (%)\n")
        for r in all_results:
            auc_s = f"{r['auc']:.1f}" if r['auc'] is not None else ""
            seen_m = f"{r['seen_mean']:.1f}" if r.get('seen_mean') is not None else ""
            seen_min = f"{r['seen_min']:.1f}" if r.get('seen_min') is not None else ""
            unseen_m = f"{r['unseen_mean']:.1f}" if r.get('unseen_mean') is not None else ""
            unseen_max = f"{r['unseen_max']:.1f}" if r.get('unseen_max') is not None else ""
            gap_s = f"{r['separation_gap']:.1f}" if r.get('separation_gap') is not None else ""
            f.write(f"{r['model']},{auc_s},{seen_m},{seen_min},{unseen_m},{unseen_max},{gap_s}\n")

    # Plot AUC vs scale
    try:
        import matplotlib.pyplot as plt
        models = [r["model"] for r in all_results if r["auc"] is not None]
        aucs = [r["auc"] for r in all_results if r["auc"] is not None]
        if models and aucs:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(models))
            bars = ax.bar(x, aucs, color=["#2ecc71"] * min(3, len(models)) + ["#e74c3c"] * max(0, len(models) - 3))
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylabel("AUC (seen vs unseen) %")
            ax.set_title("CoDeC discriminative power vs model scale (Appendix A.3)")
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random")
            ax.set_ylim(0, 105)
            for i, v in enumerate(aucs):
                ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig("codec_scale_confirmation.png", dpi=150)
            plt.close()
            print(f"  - codec_scale_confirmation.png")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    # Write Markdown summary
    md_path = "codec_scale_confirmation.md"
    with open(md_path, "w") as f:
        f.write("# CoDeC Scale Confirmation: Small vs Large Models\n\n")
        f.write(f"Confirms CoDeC paper Appendix A.3 & C.7: larger models show lower contamination scores.\n")
        f.write(f"(n_samples={args.n_samples} per dataset)\n\n")
        f.write("## Results\n\n")
        f.write("| Model | AUC (%) | Seen mean | Seen min | Unseen mean | Unseen max | Sep. gap |\n")
        f.write("|-------|---------|-----------|----------|-------------|------------|----------|\n")
        for r in all_results:
            auc_s = f"{r['auc']:.1f}" if r['auc'] is not None else "—"
            seen_m = f"{r['seen_mean']:.1f}" if r.get('seen_mean') is not None else "—"
            seen_min = f"{r['seen_min']:.1f}" if r.get('seen_min') is not None else "—"
            unseen_m = f"{r['unseen_mean']:.1f}" if r.get('unseen_mean') is not None else "—"
            unseen_max = f"{r['unseen_max']:.1f}" if r.get('unseen_max') is not None else "—"
            gap_s = f"{r['separation_gap']:.1f}" if r.get('separation_gap') is not None else "—"
            f.write(f"| {r['model']} | {auc_s} | {seen_m} | {seen_min} | {unseen_m} | {unseen_max} | {gap_s} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("- **Scale axis**: Pythia 410M → 1.4B → 12B → OpenAI davinci-002, babbage-002\n")
        f.write("- **Expected**: AUC decreases with scale (Appendix A.3); separation gap (Seen min - Unseen max) narrows.\n")

    print("\n" + "=" * 70)
    print("DONE. Outputs:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    if any(r.get("auc") for r in all_results):
        print(f"  - codec_scale_confirmation.png")
    if args.save_results:
        print(f"  - {args.results_dir}/<model>/ (saved for comparison)")
        print(f"\n  To compare all saved results:")
        print(f"    python compare_codec_results.py --results-dir {args.results_dir} --require-n-samples {args.n_samples}")
    print("=" * 70)


if __name__ == "__main__":
    main()
