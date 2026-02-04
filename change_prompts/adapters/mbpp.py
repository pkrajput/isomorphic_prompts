#!/usr/bin/env python3
"""
adapters/mbpp.py - MBPP dataset adapter.

MBPP schema:
- Prompt field: 'prompt'
- Test structure: 'test_list' - list of assert statement strings
- Problem ID: 'task_id'
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iso_transforms import Transform


class MBPPAdapter:
    """Adapter for MBPP dataset."""
    
    def __init__(self):
        self.prompt_field = "prompt"
        self.test_field = "test_list"
    
    def extract_prompt(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> str:
        """Extract the problem prompt/description from a row."""
        return row.get(self.prompt_field, "")
    
    def extract_tests(self, row: Dict[str, Any], schema_report: Optional[Dict] = None) -> List[str]:
        """
        Extract test cases from a row.
        
        Returns list of assert statement strings.
        """
        tests = row.get(self.test_field, [])
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except json.JSONDecodeError:
                return []
        return tests if isinstance(tests, list) else []
    
    def _extract_values_from_assert(self, assert_str: str) -> Tuple[str, str, str]:
        """
        Extract function call and expected value from assert statement.
        
        Returns (func_call, expected_value, original_assert)
        """
        # Match: assert func(...) == value
        match = re.match(r'assert\s+(.+?)\s*==\s*(.+)$', assert_str.strip())
        if match:
            return match.group(1).strip(), match.group(2).strip(), assert_str
        return "", "", assert_str
    
    def _transform_io_value(self, value: str, transform: Transform, params: Dict[str, Any]) -> str:
        """Apply transform to a single I/O value."""
        return transform.forward(value, params)
    
    def _collect_io_samples(self, tests: List[str]) -> List[str]:
        """Collect all I/O strings from assert statements for roundtrip validation."""
        samples = []
        for test in tests:
            func_call, expected, _ = self._extract_values_from_assert(test)
            if expected:
                samples.append(expected)
            # Also try to extract input values from function call
            # Match: func(arg1, arg2, ...) - extract the args
            match = re.match(r'\w+\s*\((.*)\)$', func_call)
            if match:
                samples.append(match.group(1))
        return samples
    
    def apply_transform_to_tests(
        self,
        tests: List[str],
        transform: Transform,
        params: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Apply transform to all test expected values.
        
        For MBPP, we transform the expected output values in assert statements.
        Input arguments are embedded in the function call.
        
        Returns:
            - Transformed tests (list of assert strings)
            - List of original samples for roundtrip verification
        """
        samples = self._collect_io_samples(tests)
        transformed = []
        
        for test in tests:
            func_call, expected, original = self._extract_values_from_assert(test)
            if func_call and expected:
                # Transform expected value
                transformed_expected = self._transform_io_value(expected, transform, params)
                # Also need to transform input args in function call
                # Extract function name and args
                match = re.match(r'(\w+)\s*\((.*)\)$', func_call)
                if match:
                    func_name = match.group(1)
                    args = match.group(2)
                    transformed_args = self._transform_io_value(args, transform, params)
                    new_test = f"assert {func_name}({transformed_args}) == {transformed_expected}"
                else:
                    new_test = f"assert {func_call} == {transformed_expected}"
                transformed.append(new_test)
            else:
                # Keep original if we can't parse
                transformed.append(original)
        
        return transformed, samples
    
    def rebuild_row(
        self,
        row: Dict[str, Any],
        prompt_with_contract: str,
        tests_transformed: List[str],
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
        new_row["iso_proof"] = meta.get("iso_proof", {})
        
        return new_row
    
    def get_dataset_dir_name(self) -> str:
        """Get the expected directory name for this dataset."""
        return "mbpp"
