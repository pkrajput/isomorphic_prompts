#!/usr/bin/env python3
"""
adapters/effibench.py - EffiBench dataset adapter.

EffiBench schema:
- Prompt field: 'prompt'
- Test structure: 'generated_tests' (JSON string containing list of test cases)
  Each test is {'input': str, 'output': str}
- Input/output format: more varied (arrays, grids, structured data)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iso_transforms import Transform


class EffiBenchAdapter:
    """Adapter for EffiBench dataset."""
    
    def __init__(self):
        self.prompt_field = "prompt"
        self.test_field = "generated_tests"
    
    def extract_prompt(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> str:
        """Extract the problem prompt from a row."""
        if schema_report and "recommended" in schema_report:
            field = schema_report["recommended"].get("prompt_field", self.prompt_field)
        else:
            field = self.prompt_field
        
        return row.get(field, "")
    
    def extract_tests(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Extract test cases from a row.
        
        Returns list of {'input': str, 'output': str} dicts.
        EffiBench stores tests as a JSON string that needs parsing.
        """
        if schema_report and "recommended" in schema_report:
            field = schema_report["recommended"].get("test_field", self.test_field)
        else:
            field = self.test_field
        
        tests_raw = row.get(field, "[]")
        
        if isinstance(tests_raw, str):
            try:
                tests = json.loads(tests_raw)
            except json.JSONDecodeError:
                return []
        elif isinstance(tests_raw, list):
            tests = tests_raw
        else:
            return []
        
        return tests
    
    def _transform_io_value(self, value: str, transform: Transform, params: Dict[str, Any]) -> str:
        """Apply transform to a single I/O value."""
        return transform.forward(value, params)
    
    def _collect_io_samples(self, tests: List[Dict[str, str]]) -> List[str]:
        """Collect all I/O strings for roundtrip validation."""
        samples = []
        for test in tests:
            if "input" in test:
                samples.append(test["input"])
            if "output" in test:
                samples.append(test["output"])
        return samples
    
    def apply_transform_to_tests(
        self,
        tests: List[Dict[str, str]],
        transform: Transform,
        params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Apply transform to all test inputs and outputs.
        
        Returns:
            - Transformed tests (same structure - list of dicts)
            - List of original samples for roundtrip verification
        """
        samples = self._collect_io_samples(tests)
        transformed = []
        
        for test in tests:
            new_test = {}
            if "input" in test:
                new_test["input"] = self._transform_io_value(test["input"], transform, params)
            if "output" in test:
                new_test["output"] = self._transform_io_value(test["output"], transform, params)
            # Preserve any other keys
            for k, v in test.items():
                if k not in new_test:
                    new_test[k] = v
            transformed.append(new_test)
        
        return transformed, samples

    def apply_transform_outputs_only(
        self,
        tests: List[Dict[str, str]],
        transform: Transform,
        params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Encode-only: keep original inputs, transform only outputs."""
        samples = self._collect_io_samples(tests)
        transformed = []
        for test in tests:
            new_test = dict(test)
            if "output" in test:
                new_test["output"] = self._transform_io_value(test["output"], transform, params)
            transformed.append(new_test)
        return transformed, samples

    def format_oracle_help_inputs(
        self,
        tests: List[Dict[str, str]],
        tests_transformed: List[Dict[str, str]]
    ) -> str:
        """
        Compact oracle block: show at most 2 worked examples so the model
        sees what decoded inputs look like without bloating the prompt.
        """
        if not tests or not tests_transformed:
            return ""

        examples = []
        for orig, enc in zip(tests, tests_transformed):
            orig_in = orig.get("input", "")
            enc_in = enc.get("input", "")
            if orig_in and enc_in and orig_in != enc_in:
                examples.append((enc_in, orig_in))
            if len(examples) >= 2:
                break

        if not examples:
            return ""

        lines = [
            "### Decoded-Input Examples",
            "Below are example decoded inputs so you can verify your decoding.",
            "Apply the same decoding to all test inputs; return ONLY encoded outputs.",
            "",
        ]
        for i, (enc_in, orig_in) in enumerate(examples, 1):
            enc_short = enc_in if len(enc_in) <= 120 else enc_in[:120] + " ..."
            orig_short = orig_in if len(orig_in) <= 120 else orig_in[:120] + " ..."
            lines.append(f"Example {i}:")
            lines.append(f"  Encoded:  {enc_short}")
            lines.append(f"  Decoded:  {orig_short}")

        return "\n".join(lines)
    
    def rebuild_row(
        self,
        row: Dict[str, Any],
        prompt_with_contract: str,
        tests_transformed: List[Dict[str, str]],
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rebuild a row with transformed data and metadata.
        
        Preserves all original keys and adds new isomorphism keys.
        Note: tests_transformed is stored as a JSON string to match original format.
        """
        # Copy all original keys
        new_row = dict(row)
        
        # Add isomorphism fields
        new_row["prompt_with_contract"] = prompt_with_contract
        # Store as JSON string to match original format
        new_row["tests_transformed"] = json.dumps(tests_transformed)
        new_row["iso_applied"] = meta.get("iso_applied", False)
        new_row["iso_family"] = meta.get("iso_family", "")
        new_row["iso_seed"] = meta.get("iso_seed", 0)
        new_row["iso_params"] = meta.get("iso_params", {})
        new_row["iso_variant"] = meta.get("iso_variant", "")
        new_row["iso_proof"] = meta.get("iso_proof", {})
        
        return new_row
    
    def get_dataset_dir_name(self) -> str:
        """Get the expected directory name for this dataset."""
        return "Effibench"
