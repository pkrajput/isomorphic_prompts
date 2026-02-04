#!/usr/bin/env python3
"""
adapters/__init__.py - Adapter registry for dataset-specific handling.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bigobench import BigOBenchAdapter
    from .effibench import EffiBenchAdapter
    from .mbpp import MBPPAdapter


def get_adapter(dataset: str):
    """Get the appropriate adapter for a dataset."""
    dataset_lower = dataset.lower()
    
    if dataset_lower == "bigobench":
        from .bigobench import BigOBenchAdapter
        return BigOBenchAdapter()
    elif dataset_lower == "effibench":
        from .effibench import EffiBenchAdapter
        return EffiBenchAdapter()
    elif dataset_lower == "mbpp":
        from .mbpp import MBPPAdapter
        return MBPPAdapter()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: bigobench, effibench, mbpp")
