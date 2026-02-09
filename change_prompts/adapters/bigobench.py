#!/usr/bin/env python3
"""
adapters/bigobench.py - BigOBench dataset adapter.

BigOBench schema:
- Prompt field: 'description'
- Test structure: 'tests' dict with 'public_tests' and 'private_tests'
  Each is a list of {'input': str, 'output': str}
- Input/output format: plaintext with newlines and spaces (competitive programming style)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iso_transforms import Transform


class BigOBenchAdapter:
    """Adapter for BigOBench dataset."""
    
    def __init__(self):
        self.prompt_field = "description"
        self.test_field = "tests"
    
    def extract_prompt(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> str:
        """Extract the problem prompt/description from a row."""
        # Use schema report recommendation if available
        if schema_report and "recommended" in schema_report:
            field = schema_report["recommended"].get("prompt_field", self.prompt_field)
        else:
            field = self.prompt_field
        
        return row.get(field, "")
    
    def extract_tests(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract test cases from a row.
        
        Returns dict with 'public_tests' and 'private_tests' keys,
        each containing a list of {'input': str, 'output': str} dicts.
        """
        if schema_report and "recommended" in schema_report:
            field = schema_report["recommended"].get("test_field", self.test_field)
        else:
            field = self.test_field
        
        tests = row.get(field, {})
        
        if isinstance(tests, str):
            # Try to parse as JSON
            try:
                tests = json.loads(tests)
            except json.JSONDecodeError:
                return {"public_tests": [], "private_tests": []}
        
        return tests
    
    def _transform_io_value(self, value: str, transform: Transform, params: Dict[str, Any]) -> str:
        """Apply transform to a single I/O value."""
        return transform.forward(value, params)
    
    def _collect_io_samples(self, tests: Dict[str, List[Dict[str, str]]]) -> List[str]:
        """Collect all I/O strings for roundtrip validation."""
        samples = []
        for test_type in ["public_tests", "private_tests"]:
            for test in tests.get(test_type, []):
                if "input" in test:
                    samples.append(test["input"])
                if "output" in test:
                    samples.append(test["output"])
        return samples
    
    def apply_transform_to_tests(
        self,
        tests: Dict[str, List[Dict[str, str]]],
        transform: Transform,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, List[Dict[str, str]]], List[str]]:
        """
        Apply transform to all test inputs and outputs.
        
        Returns:
            - Transformed tests (same structure)
            - List of original samples for roundtrip verification
        """
        samples = self._collect_io_samples(tests)
        transformed = {}
        
        for test_type in ["public_tests", "private_tests"]:
            transformed[test_type] = []
            for test in tests.get(test_type, []):
                new_test = {}
                if "input" in test:
                    new_test["input"] = self._transform_io_value(test["input"], transform, params)
                if "output" in test:
                    new_test["output"] = self._transform_io_value(test["output"], transform, params)
                # Preserve any other keys
                for k, v in test.items():
                    if k not in new_test:
                        new_test[k] = v
                transformed[test_type].append(new_test)
        
        return transformed, samples

    def format_oracle_help_inputs(
        self,
        tests: Dict[str, List[Dict[str, str]]],
        tests_transformed: Dict[str, List[Dict[str, str]]]
    ) -> str:
        """
        Format oracle-help input pairs (encoded vs decoded) for prompts.
        Only inputs are shown; outputs remain encoded per contract.
        """
        if not tests or not tests_transformed:
            return ""
        
        pairs = []
        for test_type in ["public_tests", "private_tests"]:
            orig_list = tests.get(test_type, []) or []
            enc_list = tests_transformed.get(test_type, []) or []
            
            for idx, (orig, enc) in enumerate(zip(orig_list, enc_list)):
                orig_in = orig.get("input", "")
                enc_in = enc.get("input", "")
                if orig_in or enc_in:
                    label = f"{test_type} #{idx + 1}"
                    pairs.append((label, enc_in, orig_in))
        
        if not pairs:
            return ""
        
        lines = [
            "### Oracle-Provided Decoded Inputs",
            "The inputs below are provided in both encoded and decoded form.",
            "Use the decoded inputs to solve the task, but return ONLY encoded outputs.",
            "",
        ]
        
        for label, enc_in, orig_in in pairs:
            lines.append(f"Test case: {label}")
            lines.append("Encoded input:")
            lines.append(enc_in)
            lines.append("Decoded input (oracle-provided):")
            lines.append(orig_in)
            lines.append("---")
        
        return "\n".join(lines).strip()
    
    def rebuild_row(
        self,
        row: Dict[str, Any],
        prompt_with_contract: str,
        tests_transformed: Dict[str, List[Dict[str, str]]],
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rebuild a row with transformed data and metadata.
        
        Preserves all original keys and adds new isomorphism keys.
        """
        # Copy all original keys
        new_row = dict(row)
        
        # Add isomorphism fields
        new_row["prompt_with_contract"] = prompt_with_contract
        new_row["tests_transformed"] = tests_transformed
        new_row["iso_applied"] = meta.get("iso_applied", False)
        new_row["iso_family"] = meta.get("iso_family", "")
        new_row["iso_seed"] = meta.get("iso_seed", 0)
        new_row["iso_params"] = meta.get("iso_params", {})
        new_row["iso_variant"] = meta.get("iso_variant", "")
        new_row["iso_proof"] = meta.get("iso_proof", {})
        
        return new_row
    
    def get_dataset_dir_name(self) -> str:
        """Get the expected directory name for this dataset."""
        return "BigOBench"
