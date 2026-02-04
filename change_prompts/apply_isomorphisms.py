#!/usr/bin/env python3
"""
apply_isomorphisms.py - Main driver to produce transformed JSONLs with invertibility confirmation.

Usage:
    python apply_isomorphisms.py --dataset bigobench --iso_family affine_int --seed 1
    python apply_isomorphisms.py --dataset effibench --iso_family affine_int --seed 1 --identity_control
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
    """Find all JSONL files for a dataset."""
    # Handle case-insensitive matching
    expected_dir_name = adapter.get_dataset_dir_name()
    
    for child in datasets_root.iterdir():
        if child.is_dir() and child.name.lower() == expected_dir_name.lower():
            return list(child.glob("*.jsonl"))
    
    # Fallback: try exact name
    dataset_dir = datasets_root / expected_dir_name
    if dataset_dir.exists():
        return list(dataset_dir.glob("*.jsonl"))
    
    return []


def process_row(
    row: Dict[str, Any],
    adapter,
    transform,
    params: Dict[str, Any],
    schema_report: Optional[Dict],
    iso_family: str,
    seed: int,
    identity_control: bool
) -> Dict[str, Any]:
    """
    Process a single row: extract, transform, verify, rebuild.
    """
    try:
        # Extract prompt and tests
        prompt = adapter.extract_prompt(row, schema_report)
        tests = adapter.extract_tests(row, schema_report)
        
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
        else:
            tests_transformed, samples = adapter.apply_transform_to_tests(tests, transform, params)
            contract = transform.contract(params)
        
        # Append contract to prompt
        prompt_with_contract = append_contract_to_prompt(prompt, contract)
        
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
                "iso_proof": proof
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
    max_rows: Optional[int] = None,
    dry_run: bool = False
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
        "errors": 0
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
                
                # Process the row
                new_row = process_row(
                    row, adapter, transform, params,
                    schema_report, iso_family, seed, identity_control
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
                        help="Transform family to apply")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for transform parameters")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Maximum rows to process per file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without processing")
    parser.add_argument("--identity_control", action="store_true",
                        help="Apply identity transform (control experiment)")
    
    args = parser.parse_args()
    
    # Paths
    datasets_root = Path(args.datasets_root)
    # Default output to datasets/<dataset>_iso
    if args.out_root is None:
        out_root = datasets_root / f"{args.dataset}_iso"
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
        "errors": 0
    }
    
    for input_file in input_files:
        # Output directly to out_root (e.g., datasets/bigobench_iso/)
        output_file = out_root / (input_file.stem + ".iso.jsonl")
        print(f"\nProcessing: {input_file.name}")
        
        stats = process_file(
            input_file, output_file, adapter, transform, params,
            schema_report, args.iso_family, args.seed,
            args.identity_control, args.max_rows, args.dry_run
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
