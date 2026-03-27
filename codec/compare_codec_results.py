#!/usr/bin/env python3
"""
Compare CoDeC results from saved runs (--save-results).

Usage:
  python compare_codec_results.py [--results-dir results]
  python compare_codec_results.py --results-dir results --output comparison.csv

Reads results/<model>/summary.json (or *.json per dataset) and produces:
  - comparison.csv : AUC, per-dataset scores, separation gap per model
  - comparison.md  : markdown table
  - comparison.png : bar plot of AUC by model (optional)
"""
import argparse
import json
import os
from pathlib import Path


def _display_model_name(model_name: str) -> str:
    """Normalize model names for readable comparison outputs."""
    m = str(model_name or "").strip()
    if m.startswith("vllm:"):
        m = m[len("vllm:"):]
    m = m.replace("meta-llama/Meta-Llama-3.1-405B-Instruct", "Llama-3.1-405b-instruct")
    m = m.replace("nvidia/Llama-3.1-405B-Instruct-NVFP4", "Llama-3.1-405b-instruct-NVFP4")
    return m


def _load_from_dataset_files(model_dir: Path):
    """Build a model summary from per-dataset JSON files in results/<model>/."""
    dataset_files = [p for p in model_dir.glob("*.json") if p.name != "summary.json"]
    if not dataset_files:
        return None

    scores = []
    n_samples_values = set()
    model_name = None

    for p in sorted(dataset_files):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load {p}: {e}")
            continue

        dataset = d.get("dataset", p.stem)
        d_type = d.get("type", "Unseen")
        score = d.get("score")
        if score is None:
            continue

        model_name = model_name or d.get("model")
        if d.get("n_samples") is not None:
            n_samples_values.add(d.get("n_samples"))

        scores.append({
            "Dataset": dataset,
            "Type": d_type,
            "Score": float(score),
        })

    if not scores:
        return None

    n_samples = list(n_samples_values)[0] if len(n_samples_values) == 1 else None
    return {
        "model": _display_model_name(model_name or model_dir.name),
        "auc": None,  # kept as None when not provided by summary
        "scores": scores,
        "n_samples": n_samples,
        "_model_key": model_dir.name,
    }


def load_model_summaries(results_dir: str):
    """Load model results from dataset JSON files (fallback: summary.json)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    summaries = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        data = _load_from_dataset_files(model_dir)
        if data is not None:
            summaries.append(data)
            continue

        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
            data["_model_key"] = model_dir.name
            summaries.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {summary_path}: {e}")
    return summaries


def compute_separation_gap(scores: list):
    """Seen min - Unseen max. Positive = good separation."""
    seen = [r["Score"] * 100 for r in scores if "Seen" in r.get("Type", "")]
    unseen = [r["Score"] * 100 for r in scores if r.get("Type") == "Unseen"]
    if not seen or not unseen:
        return None
    return min(seen) - max(unseen)


def compute_auc_from_scores(scores: list):
    """Compute dataset-level AUC from seen/unseen scores without sklearn.

    AUC = P(score_seen > score_unseen) + 0.5 * P(tie)
    """
    seen = [r["Score"] for r in scores if "Seen" in r.get("Type", "")]
    unseen = [r["Score"] for r in scores if r.get("Type") == "Unseen"]
    if not seen or not unseen:
        return None

    wins = 0.0
    total = len(seen) * len(unseen)
    for s in seen:
        for u in unseen:
            if s > u:
                wins += 1.0
            elif s == u:
                wins += 0.5
    return wins / total if total > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Compare saved CoDeC results")
    parser.add_argument("--results-dir", default="results", help="Directory with results/<model>/")
    parser.add_argument("--output", default=None, help="Output CSV path (default: codec_comparison.csv)")
    parser.add_argument("--plot", action="store_true", help="Generate codec_comparison.png bar plot")
    parser.add_argument("--require-n-samples", type=int, default=100,
                        help="Only include models that used this many samples (default: 100). Use 0 to include all.")
    args = parser.parse_args()

    summaries = load_model_summaries(args.results_dir)
    if not summaries:
        print(f"No summaries found in {args.results_dir}/")
        print("Run experiments with --save-results first:")
        print("  python run_codec_scale_confirmation.py --save-results --skip-gemini-2.5")
        return

    # Filter by n_samples if required (ensures same conditions for all models)
    if args.require_n_samples > 0:
        before = len(summaries)
        summaries = [s for s in summaries if s.get("n_samples") == args.require_n_samples]
        if len(summaries) < before:
            print(f"  Excluded {before - len(summaries)} model(s) with n_samples != {args.require_n_samples}")
    if not summaries:
        print(f"No models with n_samples={args.require_n_samples}. Run with 1000 samples:")
        print("  python run_codec_scale_confirmation.py --save-results --skip-gemini-2.5")
        return
    n_samples_set = {s.get("n_samples") for s in summaries}
    if len(n_samples_set) > 1:
        print(f"  Warning: Models have different n_samples: {n_samples_set}")

    rows = []
    for s in summaries:
        model = _display_model_name(s.get("model", s.get("_model_key", "?")))
        auc = s.get("auc")
        scores = s.get("scores", [])
        if auc is None:
            auc = compute_auc_from_scores(scores)
        seen_scores = [r["Score"] * 100 for r in scores if "Seen" in r.get("Type", "")]
        unseen_scores = [r["Score"] * 100 for r in scores if r.get("Type") == "Unseen"]
        gap = compute_separation_gap(scores)

        rows.append({
            "model": model,
            "n_samples": s.get("n_samples"),
            "auc": auc * 100 if auc is not None else None,
            "seen_mean": sum(seen_scores) / len(seen_scores) if seen_scores else None,
            "seen_min": min(seen_scores) if seen_scores else None,
            "unseen_mean": sum(unseen_scores) / len(unseen_scores) if unseen_scores else None,
            "unseen_max": max(unseen_scores) if unseen_scores else None,
            "separation_gap": gap,
        })

    out_csv = args.output or "comparison.csv"
    with open(out_csv, "w") as f:
        f.write("Model,n_samples,AUC (%),Seen mean (%),Seen min (%),Unseen mean (%),Unseen max (%),Separation gap (%)\n")
        for r in rows:
            n_s = str(r.get("n_samples", ""))
            auc_s = f"{r['auc']:.1f}" if r["auc"] is not None else ""
            seen_m = f"{r['seen_mean']:.1f}" if r.get("seen_mean") is not None else ""
            seen_min = f"{r['seen_min']:.1f}" if r.get("seen_min") is not None else ""
            unseen_m = f"{r['unseen_mean']:.1f}" if r.get("unseen_mean") is not None else ""
            unseen_max = f"{r['unseen_max']:.1f}" if r.get("unseen_max") is not None else ""
            gap_s = f"{r['separation_gap']:.1f}" if r.get("separation_gap") is not None else ""
            f.write(f"{r['model']},{n_s},{auc_s},{seen_m},{seen_min},{unseen_m},{unseen_max},{gap_s}\n")
    print(f"Wrote {out_csv}")

    out_md = out_csv.replace(".csv", ".md")
    with open(out_md, "w") as f:
        f.write("# CoDeC Results Comparison\n\n")
        f.write("| Model | n_samples | AUC (%) | Seen mean | Seen min | Unseen mean | Unseen max | Sep. gap |\n")
        f.write("|-------|-----------|---------|-----------|----------|-------------|------------|----------|\n")
        for r in rows:
            n_s = str(r.get("n_samples", ""))
            auc_s = f"{r['auc']:.1f}" if r["auc"] is not None else "—"
            seen_m = f"{r['seen_mean']:.1f}" if r.get("seen_mean") is not None else "—"
            seen_min = f"{r['seen_min']:.1f}" if r.get("seen_min") is not None else "—"
            unseen_m = f"{r['unseen_mean']:.1f}" if r.get("unseen_mean") is not None else "—"
            unseen_max = f"{r['unseen_max']:.1f}" if r.get("unseen_max") is not None else "—"
            gap_s = f"{r['separation_gap']:.1f}" if r.get("separation_gap") is not None else "—"
            f.write(f"| {r['model']} | {n_s} | {auc_s} | {seen_m} | {seen_min} | {unseen_m} | {unseen_max} | {gap_s} |\n")
    print(f"Wrote {out_md}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            models = [r["model"].split("/")[-1] if "/" in r["model"] else r["model"] for r in rows]
            aucs = [r["auc"] or 0 for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(models))
            ax.bar(x, aucs, color="#2ecc71", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylabel("AUC (seen vs unseen) %")
            ax.set_title("CoDeC Results Comparison")
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim(0, 105)
            for i, v in enumerate(aucs):
                if v > 0:
                    ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig("codec_comparison.png", dpi=200)
            fig.savefig("codec_comparison.pdf", dpi=200)
            plt.close()
            print("Wrote codec_comparison.png")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
