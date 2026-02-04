#!/usr/bin/env python3
"""
inspect_schema.py - Infer schema from JSONL files for BigOBench and EffiBench datasets.

Analyzes JSONL rows to identify:
- Prompt fields: long strings with natural language
- Test fields: structures containing input/output patterns
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def is_natural_language(text: str, min_length: int = 100) -> bool:
    """Check if text appears to be natural language (prompt-like)."""
    if len(text) < min_length:
        return False
    # Check for sentence-like patterns
    sentence_markers = ['. ', '? ', '! ', '\n\n']
    has_sentences = any(marker in text for marker in sentence_markers)
    # Check for common prompt words
    prompt_words = ['input', 'output', 'return', 'given', 'find', 'write', 'function', 'array', 'string']
    text_lower = text.lower()
    has_prompt_words = sum(1 for w in prompt_words if w in text_lower) >= 2
    return has_sentences or has_prompt_words


def looks_like_test_field(key: str, value: Any) -> Tuple[bool, str]:
    """Check if a field looks like it contains test data."""
    reasons = []
    
    # Check key name
    test_key_patterns = ['test', 'input', 'output', 'case', 'example', 'sample']
    if any(p in key.lower() for p in test_key_patterns):
        reasons.append(f"key contains test-related word")
    
    if isinstance(value, str):
        # Check for assert statements
        if 'assert' in value:
            reasons.append("contains 'assert'")
        # Check for input/output patterns in string
        if re.search(r'"input":\s*"', value) or re.search(r'"output":\s*"', value):
            reasons.append("contains input/output JSON pattern")
        # Check if it's a JSON string
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    if 'input' in parsed[0] or 'output' in parsed[0]:
                        reasons.append("JSON list with input/output dicts")
        except:
            pass
    
    elif isinstance(value, dict):
        # Check for test-related keys
        if 'public_tests' in value or 'private_tests' in value:
            reasons.append("contains public_tests/private_tests")
        if 'input' in value or 'output' in value:
            reasons.append("contains input/output keys")
    
    elif isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict):
            if 'input' in value[0] or 'output' in value[0]:
                reasons.append("list of dicts with input/output")
    
    return len(reasons) > 0, "; ".join(reasons)


def analyze_value_type(value: Any) -> Dict[str, Any]:
    """Analyze the type and structure of a value."""
    result = {"type": type(value).__name__}
    
    if isinstance(value, str):
        result["length"] = len(value)
        result["is_natural_language"] = is_natural_language(value)
        # Check if it's JSON
        try:
            parsed = json.loads(value)
            result["is_json"] = True
            result["json_type"] = type(parsed).__name__
            if isinstance(parsed, list):
                result["json_list_length"] = len(parsed)
                if len(parsed) > 0:
                    result["json_item_type"] = type(parsed[0]).__name__
                    if isinstance(parsed[0], dict):
                        result["json_item_keys"] = list(parsed[0].keys())[:10]
        except:
            result["is_json"] = False
        result["sample"] = value[:200] + "..." if len(value) > 200 else value
    
    elif isinstance(value, dict):
        result["keys"] = list(value.keys())
        result["n_keys"] = len(value)
        # Sample nested structure
        if len(value) > 0:
            first_key = list(value.keys())[0]
            first_val = value[first_key]
            if isinstance(first_val, list) and len(first_val) > 0:
                result["first_value_type"] = "list"
                result["first_value_length"] = len(first_val)
                if isinstance(first_val[0], dict):
                    result["first_value_item_keys"] = list(first_val[0].keys())
    
    elif isinstance(value, list):
        result["length"] = len(value)
        if len(value) > 0:
            result["item_type"] = type(value[0]).__name__
            if isinstance(value[0], dict):
                result["item_keys"] = list(value[0].keys())
    
    elif isinstance(value, (int, float)):
        result["value_sample"] = value
    
    elif value is None:
        result["type"] = "null"
    
    return result


def inspect_jsonl_file(filepath: Path, max_rows: int = 200) -> Dict[str, Any]:
    """Inspect a single JSONL file and return schema analysis."""
    key_stats = defaultdict(lambda: {"count": 0, "types": defaultdict(int), "samples": []})
    prompt_candidates = []
    test_candidates = []
    
    row_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if row_count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                continue
            
            row_count += 1
            
            for key, value in row.items():
                stats = key_stats[key]
                stats["count"] += 1
                
                type_info = analyze_value_type(value)
                type_key = type_info["type"]
                stats["types"][type_key] += 1
                
                # Keep sample if we don't have enough
                if len(stats["samples"]) < 3:
                    stats["samples"].append(type_info)
                
                # Check for prompt candidate (only on first few rows)
                if row_count <= 5 and isinstance(value, str):
                    if type_info.get("is_natural_language", False):
                        if key not in [c["field"] for c in prompt_candidates]:
                            prompt_candidates.append({
                                "field": key,
                                "avg_length": type_info["length"],
                                "reason": "natural language text"
                            })
                
                # Check for test candidate
                if row_count <= 5:
                    is_test, reason = looks_like_test_field(key, value)
                    if is_test:
                        if key not in [c["field"] for c in test_candidates]:
                            test_candidates.append({
                                "field": key,
                                "type": type_info["type"],
                                "reason": reason,
                                "structure": type_info
                            })
    
    # Finalize key stats
    final_stats = {}
    for key, stats in key_stats.items():
        final_stats[key] = {
            "count": stats["count"],
            "types": dict(stats["types"]),
            "samples": stats["samples"]
        }
    
    return {
        "file": str(filepath.name),
        "rows_analyzed": row_count,
        "keys": final_stats,
        "prompt_candidates": prompt_candidates,
        "test_candidates": test_candidates
    }


def inspect_dataset(dataset: str, datasets_root: str, max_rows: int = 200) -> Dict[str, Any]:
    """Inspect all JSONL files in a dataset directory."""
    # Handle case-insensitive folder matching
    root = Path(datasets_root)
    dataset_dir = None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == dataset.lower():
            dataset_dir = child
            break
    
    if dataset_dir is None:
        raise FileNotFoundError(f"Dataset directory not found: {dataset} in {datasets_root}")
    
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {dataset_dir}")
    
    file_reports = []
    all_prompt_candidates = {}
    all_test_candidates = {}
    
    for filepath in jsonl_files:
        print(f"Analyzing {filepath.name}...")
        report = inspect_jsonl_file(filepath, max_rows)
        file_reports.append(report)
        
        # Aggregate candidates
        for pc in report["prompt_candidates"]:
            field = pc["field"]
            if field not in all_prompt_candidates:
                all_prompt_candidates[field] = {"files": [], "reasons": set()}
            all_prompt_candidates[field]["files"].append(filepath.name)
            all_prompt_candidates[field]["reasons"].add(pc["reason"])
        
        for tc in report["test_candidates"]:
            field = tc["field"]
            if field not in all_test_candidates:
                all_test_candidates[field] = {"files": [], "reasons": set(), "structure": tc["structure"]}
            all_test_candidates[field]["files"].append(filepath.name)
            all_test_candidates[field]["reasons"].add(tc["reason"])
    
    # Convert sets to lists for JSON serialization
    for k, v in all_prompt_candidates.items():
        v["reasons"] = list(v["reasons"])
    for k, v in all_test_candidates.items():
        v["reasons"] = list(v["reasons"])
    
    # Determine best candidates
    best_prompt_field = None
    best_test_field = None
    
    if all_prompt_candidates:
        # Prefer 'description' or 'prompt' if available
        preferred = ['description', 'prompt', 'statement', 'problem']
        for pref in preferred:
            if pref in all_prompt_candidates:
                best_prompt_field = pref
                break
        if not best_prompt_field:
            best_prompt_field = list(all_prompt_candidates.keys())[0]
    
    if all_test_candidates:
        # Prefer 'tests' or 'generated_tests' if available
        preferred = ['tests', 'generated_tests', 'test_cases', 'public_tests']
        for pref in preferred:
            if pref in all_test_candidates:
                best_test_field = pref
                break
        if not best_test_field:
            best_test_field = list(all_test_candidates.keys())[0]
    
    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "files_analyzed": len(file_reports),
        "file_reports": file_reports,
        "prompt_candidates": all_prompt_candidates,
        "test_candidates": all_test_candidates,
        "recommended": {
            "prompt_field": best_prompt_field,
            "test_field": best_test_field
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect JSONL schema for BigOBench/EffiBench")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["bigobench", "effibench"],
                        help="Dataset to inspect")
    parser.add_argument("--datasets_root", type=str, default="./datasets",
                        help="Root directory containing dataset folders")
    parser.add_argument("--max_rows", type=int, default=200,
                        help="Maximum rows to analyze per file")
    parser.add_argument("--output_dir", type=str, default="./change_prompts/reports",
                        help="Directory to write schema report")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inspect dataset
    print(f"Inspecting {args.dataset} dataset...")
    report = inspect_dataset(args.dataset, args.datasets_root, args.max_rows)
    
    # Write report
    output_file = output_dir / f"{args.dataset}.schema.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSchema report written to: {output_file}")
    print(f"\nRecommended fields:")
    print(f"  Prompt field: {report['recommended']['prompt_field']}")
    print(f"  Test field: {report['recommended']['test_field']}")
    
    # Print candidates summary
    print(f"\nPrompt candidates: {list(report['prompt_candidates'].keys())}")
    print(f"Test candidates: {list(report['test_candidates'].keys())}")


if __name__ == "__main__":
    main()
