#!/usr/bin/env python3
"""
gen_gemini.py - Backward-compatible wrapper for gen_models.py

This script is kept for backward compatibility. For new usage, prefer gen_models.py
which supports both Gemini (API) and HuggingFace (local GPU) models.

Supported models in gen_models.py:
- gemini-2.0-flash, gemini-2.5-pro (API-based)
- starcoder-15b, codestral-22b (HuggingFace, local GPU)
"""

import sys
import os

# Import and run main from gen_models
if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add to path if needed
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    from gen_models import main
    main()
