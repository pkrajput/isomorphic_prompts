#!/usr/bin/env python3
"""
apply_isomorphisms.py - Main driver to produce transformed JSONLs with invertibility confirmation.

Usage:
    python apply_isomorphisms.py --dataset bigobench --iso_family affine_int --seed 1
    python apply_isomorphisms.py --dataset effibench --iso_family affine_int --seed 1 --identity_control
    python apply_isomorphisms.py --dataset mbpp --iso_family affine_int --seed 1 --oracle_help_inputs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from adapters import get_adapter
from contracts import append_contract_to_prompt, generate_identity_contract
from iso_transforms import get_transform, list_transforms


def load_schema_report(dataset: str, reports_dir: Path) -> Optional[Dict]:
    """Load schema report for a dataset."""
    report_file = reports_dir / f"{dataset}.schema.json"
    if not report_file.exists():
        return None
    
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_dataset_files(dataset: str, datasets_root: Path, adapter) -> List[Path]:
    """Find original JSONL files for a dataset.

    Supports both the reorganised layout (datasets/<DS>/original/*.jsonl)
    and the legacy flat layout (datasets/<DS>/*.jsonl).
    """
    expected_dir_name = adapter.get_dataset_dir_name()

    # Try case-insensitive matching
    matched_dir = None
    for child in datasets_root.iterdir():
        if child.is_dir() and child.name.lower() == expected_dir_name.lower():
            matched_dir = child
            break
    if matched_dir is None:
        matched_dir = datasets_root / expected_dir_name

    if not matched_dir.exists():
        return []

    # Prefer <DS>/original/ subfolder (new layout)
    orig_sub = matched_dir / "original"
    if orig_sub.is_dir():
        files = list(orig_sub.glob("*.jsonl"))
        if files:
            return files

    # Fallback: JSONL files directly under <DS>/
    return list(matched_dir.glob("*.jsonl"))


def process_row(
    row: Dict[str, Any],
    adapter,
    transform,
    params: Dict[str, Any],
    schema_report: Optional[Dict],
    iso_family: str,
    seed: int,
    identity_control: bool,
    oracle_help_inputs: bool,
    encode_only: bool = False,
) -> Dict[str, Any]:
    """
    Process a single row: extract, transform, verify, rebuild.
    """
    try:
        # Extract prompt and tests
        prompt = adapter.extract_prompt(row, schema_report)
        tests = adapter.extract_tests(row, schema_report)

        if encode_only:
            variant = "iso_enc_only"
        elif oracle_help_inputs:
            variant = "iso_dec_only"
        else:
            variant = "iso"
        
        # Check if we have valid data
        if not prompt or not tests:
            return adapter.rebuild_row(
                row,
                prompt_with_contract=prompt,
                tests_transformed=tests,
                meta={
                    "iso_applied": False,
                    "iso_family": iso_family,
                    "iso_seed": seed,
                    "iso_params": params,
                    "iso_variant": variant,
                    "iso_proof": {
                        "roundtrip_ok": False,
                        "n_checked": 0,
                        "examples": [],
                        "error": "No prompt or tests found"
                    }
                }
            )
        
        # Apply transform to tests (unless identity control)
        if identity_control:
            tests_transformed = tests  # Keep original
            samples = []
            contract = generate_identity_contract()
        elif encode_only:
            tests_transformed, samples = adapter.apply_transform_outputs_only(tests, transform, params)
            a, b = params.get("a", "?"), params.get("b", "?")
            sign_b = "+" if b >= 0 else "-"
            abs_b = abs(b)
            contract = f"""### Encoding Contract (must follow exactly)
Your inputs are in their original (unencoded) form --- read them as-is.
Your output integers must be encoded using the formula: x' = {a}*x {sign_b} {abs_b}

Examples:
- If your computed answer is 10, you must output {a*10+b}
- If your computed answer is 0, you must output {b}
"""
        else:
            tests_transformed, samples = adapter.apply_transform_to_tests(tests, transform, params)
            contract = transform.contract(params)
        
        # Append contract to prompt
        prompt_with_contract = append_contract_to_prompt(prompt, contract)
        
        # Oracle-help ablation: show both encoded and decoded inputs
        if oracle_help_inputs and hasattr(adapter, "format_oracle_help_inputs"):
            oracle_block = adapter.format_oracle_help_inputs(tests, tests_transformed)
            if oracle_block:
                prompt_with_contract = append_contract_to_prompt(prompt_with_contract, oracle_block)
        
        # Verify roundtrip (unless identity)
        if identity_control or not samples:
            proof = {
                "roundtrip_ok": True,
                "n_checked": 0,
                "examples": [],
                "note": "identity transform" if identity_control else "no samples"
            }
        else:
            roundtrip_ok, proof = transform.validate_roundtrip(samples, params)
        
        return adapter.rebuild_row(
            row,
            prompt_with_contract=prompt_with_contract,
            tests_transformed=tests_transformed,
            meta={
                "iso_applied": not identity_control,
                "iso_family": iso_family if not identity_control else "identity",
                "iso_seed": seed,
                "iso_params": params,
                "iso_proof": proof,
                "iso_variant": variant,
            }
        )
    
    except Exception as e:
        # Never crash on a single row
        return adapter.rebuild_row(
            row,
            prompt_with_contract=adapter.extract_prompt(row, schema_report),
            tests_transformed=adapter.extract_tests(row, schema_report),
            meta={
                "iso_applied": False,
                "iso_family": iso_family,
                "iso_seed": seed,
                "iso_params": params,
                "iso_variant": "iso_dec_only" if oracle_help_inputs else "iso",
                "iso_proof": {
                    "roundtrip_ok": False,
                    "n_checked": 0,
                    "examples": [],
                    "error": str(e)
                }
            }
        )


def process_file(
    input_file: Path,
    output_file: Path,
    adapter,
    transform,
    params: Dict[str, Any],
    schema_report: Optional[Dict],
    iso_family: str,
    seed: int,
    identity_control: bool,
    oracle_help_inputs: bool,
    max_rows: Optional[int] = None,
    dry_run: bool = False,
    max_test_chars: Optional[int] = None,
    encode_only: bool = False,
) -> Dict[str, int]:
    """
    Process a single JSONL file.
    
    Returns stats dict with counts.
    """
    stats = {
        "total": 0,
        "processed": 0,
        "iso_applied": 0,
        "roundtrip_ok": 0,
        "errors": 0,
        "skipped_large": 0,
    }
    
    if dry_run:
        print(f"[DRY RUN] Would process: {input_file} -> {output_file}")
        return stats
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line_num, line in enumerate(fin):
                if max_rows and line_num >= max_rows:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                stats["total"] += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    stats["errors"] += 1
                    continue

                # Skip rows whose raw test payload exceeds the cap
                if max_test_chars:
                    raw_tests = row.get("generated_tests", row.get("tests", ""))
                    test_len = len(raw_tests) if isinstance(raw_tests, str) else len(json.dumps(raw_tests))
                    if test_len > max_test_chars:
                        stats["skipped_large"] += 1
                        # Still write the row through untransformed so
                        # indices stay aligned with the original file
                        fout.write(json.dumps(row) + "\n")
                        continue

                # Process the row
                new_row = process_row(
                    row, adapter, transform, params,
                    schema_report, iso_family, seed, identity_control, oracle_help_inputs,
                    encode_only=encode_only,
                )
                
                stats["processed"] += 1
                
                if new_row.get("iso_applied", False):
                    stats["iso_applied"] += 1
                
                if new_row.get("iso_proof", {}).get("roundtrip_ok", False):
                    stats["roundtrip_ok"] += 1
                
                if new_row.get("iso_proof", {}).get("error"):
                    stats["errors"] += 1
                
                # Write output
                fout.write(json.dumps(new_row) + "\n")
                
                # Progress
                if stats["processed"] % 100 == 0:
                    print(f"  Processed {stats['processed']} rows...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply invertible I/O isomorphisms to dataset JSONLs"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["bigobench", "effibench", "mbpp"],
                        help="Dataset to process")
    parser.add_argument("--datasets_root", type=str, default="./datasets",
                        help="Root directory containing dataset folders")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Output directory for transformed JSONLs (default: ./datasets/<dataset>_iso)")
    parser.add_argument("--reports_dir", type=str, default="./change_prompts/reports",
                        help="Directory containing schema reports")
    parser.add_argument("--iso_family", type=str, required=True,
                        choices=list_transforms(),
                        help="Transform family: affine_int, base_conv, digit_permutation, identity, ...")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for transform parameters")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum rows to process per file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without processing")
    parser.add_argument("--identity_control", action="store_true",
                        help="Apply identity transform (control experiment)")
    parser.add_argument("--oracle_help_inputs", action="store_true",
                        help="Oracle-help ablation: show decoded inputs alongside encoded inputs")
    parser.add_argument("--encode_only", action="store_true",
                        help="Encode-only ablation: keep original inputs, only encode outputs")
    parser.add_argument("--max_test_chars", type=int, default=None,
                        help="Skip transform for rows whose raw test payload exceeds this many chars (write row untransformed)")

    args = parser.parse_args()
    
    # Paths
    datasets_root = Path(args.datasets_root)

    # Resolve the dataset's top-level directory name (preserving case)
    dataset_dir_map = {"bigobench": "BigOBench", "effibench": "Effibench", "mbpp": "mbpp"}
    ds_dir_name = dataset_dir_map.get(args.dataset, args.dataset)

    # Build a human-readable variant sub-folder name
    family_alias = {"affine_int": "affine", "basek_int": "base_conv",
                     "base_conv": "base_conv", "cubic_int": "cubic",
                     "identity": "identity"}
    family_short = family_alias.get(args.iso_family, args.iso_family)

    if args.out_root is None:
        if args.encode_only:
            out_root = datasets_root / ds_dir_name / f"iso_{family_short}_encode_only"
        elif args.oracle_help_inputs:
            out_root = datasets_root / ds_dir_name / f"iso_{family_short}_oracle"
        else:
            out_root = datasets_root / ds_dir_name / f"iso_{family_short}"
    else:
        out_root = Path(args.out_root)
    reports_dir = Path(args.reports_dir)
    
    # Load adapter
    adapter = get_adapter(args.dataset)
    
    # Load schema report
    schema_report = load_schema_report(args.dataset, reports_dir)
    if schema_report is None:
        print(f"Warning: No schema report found at {reports_dir / f'{args.dataset}.schema.json'}")
        print("Run inspect_schema.py first, or continuing with default field detection.")
    
    # Load transform and sample params
    transform = get_transform(args.iso_family)
    params = transform.sample_params(args.seed)
    
    print(f"Dataset: {args.dataset}")
    print(f"Transform: {args.iso_family}")
    print(f"Params: {params}")
    print(f"Seed: {args.seed}")
    print(f"Identity control: {args.identity_control}")
    print(f"Oracle-help inputs: {args.oracle_help_inputs}")
    print(f"Encode-only: {args.encode_only}")
    print()
    
    # Find input files
    input_files = find_dataset_files(args.dataset, datasets_root, adapter)
    if not input_files:
        print(f"Error: No JSONL files found for dataset {args.dataset} in {datasets_root}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # Process each file
    total_stats = {
        "total": 0,
        "processed": 0,
        "iso_applied": 0,
        "roundtrip_ok": 0,
        "errors": 0,
        "skipped_large": 0,
    }
    
    for input_file in input_files:
        # Output directly to out_root (e.g., datasets/bigobench_iso/)
        output_file = out_root / (input_file.stem + ".iso.jsonl")
        print(f"\nProcessing: {input_file.name}")
        
        stats = process_file(
            input_file, output_file, adapter, transform, params,
            schema_report, args.iso_family, args.seed,
            args.identity_control, args.oracle_help_inputs,
            args.max_rows, args.dry_run,
            max_test_chars=args.max_test_chars,
            encode_only=args.encode_only,
        )
        
        for k, v in stats.items():
            total_stats[k] += v
        
        print(f"  -> {output_file}")
        print(f"  Stats: {stats}")
    
    print("\n" + "="*60)
    print("TOTAL STATS:")
    print(f"  Rows processed: {total_stats['processed']}/{total_stats['total']}")
    print(f"  Transforms applied: {total_stats['iso_applied']}")
    print(f"  Roundtrip verified: {total_stats['roundtrip_ok']}")
    print(f"  Errors: {total_stats['errors']}")
    
    if total_stats['processed'] > 0:
        success_rate = total_stats['roundtrip_ok'] / total_stats['processed'] * 100
        print(f"  Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
