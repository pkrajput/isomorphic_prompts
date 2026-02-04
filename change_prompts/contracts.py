#!/usr/bin/env python3
"""
contracts.py - Generate encoding contract appendices for prompts.

Provides utility functions to generate and append encoding contracts
that describe the I/O transformation applied to test data.
"""

from typing import Dict, Any, Optional


def generate_contract(transform_name: str, params: Dict[str, Any]) -> str:
    """
    Generate an encoding contract for the given transform and params.
    
    This is a convenience wrapper that imports the transform and calls its contract method.
    """
    from iso_transforms import get_transform
    transform = get_transform(transform_name)
    return transform.contract(params)


def generate_identity_contract() -> str:
    """Generate the identity contract (no transformation)."""
    return """### Encoding Contract (must follow exactly)
Input and output use standard formatting (no encoding applied).
Process values as-is.
"""


def append_contract_to_prompt(prompt: str, contract: str) -> str:
    """
    Append an encoding contract to a prompt.
    
    Adds appropriate spacing and formatting.
    """
    # Ensure prompt ends with newlines
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    
    # Add spacing before contract
    return prompt + "\n" + contract


def format_contract_with_examples(
    transform_name: str,
    params: Dict[str, Any],
    sample_inputs: Optional[list] = None,
    sample_outputs: Optional[list] = None
) -> str:
    """
    Generate a contract with additional concrete examples from actual test data.
    
    This provides more context to the model by showing real transformed values.
    """
    from iso_transforms import get_transform
    transform = get_transform(transform_name)
    
    # Start with the base contract
    contract = transform.contract(params)
    
    # Add concrete examples if provided
    if sample_inputs or sample_outputs:
        contract += "\nConcrete examples from this problem:\n"
        
        if sample_inputs:
            for i, inp in enumerate(sample_inputs[:2]):
                # Show truncated version
                truncated = inp[:80] + "..." if len(inp) > 80 else inp
                contract += f"- Sample input {i+1}: {repr(truncated)}\n"
        
        if sample_outputs:
            for i, out in enumerate(sample_outputs[:2]):
                truncated = out[:80] + "..." if len(out) > 80 else out
                contract += f"- Sample output {i+1}: {repr(truncated)}\n"
    
    return contract
