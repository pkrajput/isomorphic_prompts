#!/usr/bin/env python3
"""
iso_transforms.py - Library of seeded, invertible I/O transforms.

Each transform family provides:
- name: str
- sample_params(seed) -> dict
- forward(value, params) -> transformed
- inverse(transformed, params) -> original
- contract(params) -> str (encoding contract text)
- validate_roundtrip(samples, params) -> (ok, proof)
"""

import random
import re
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class Transform(ABC):
    """Base class for all transforms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Transform family name."""
        pass
    
    @abstractmethod
    def sample_params(self, seed: int) -> Dict[str, Any]:
        """Generate random parameters for this transform."""
        pass
    
    @abstractmethod
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        """Apply forward transform to a string value."""
        pass
    
    @abstractmethod
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        """Apply inverse transform to recover original."""
        pass
    
    @abstractmethod
    def contract(self, params: Dict[str, Any]) -> str:
        """Generate encoding contract text."""
        pass
    
    def validate_roundtrip(self, samples: List[str], params: Dict[str, Any]) -> Tuple[bool, Dict]:
        """Validate that inverse(forward(x)) == x for all samples."""
        examples = []
        errors = []
        
        for i, orig in enumerate(samples[:10]):  # Check up to 10 samples
            try:
                fwd = self.forward(orig, params)
                inv = self.inverse(fwd, params)
                ok = (inv == orig)
                
                example = {
                    "orig": orig[:100] + "..." if len(orig) > 100 else orig,
                    "fwd": fwd[:100] + "..." if len(fwd) > 100 else fwd,
                    "inv": inv[:100] + "..." if len(inv) > 100 else inv,
                    "match": ok
                }
                examples.append(example)
                
                if not ok:
                    errors.append(f"Sample {i}: roundtrip mismatch")
            except Exception as e:
                errors.append(f"Sample {i}: {str(e)}")
                examples.append({
                    "orig": orig[:100] + "..." if len(orig) > 100 else orig,
                    "error": str(e)
                })
        
        all_ok = len(errors) == 0 and all(ex.get("match", False) for ex in examples if "match" in ex)
        
        return all_ok, {
            "roundtrip_ok": all_ok,
            "n_checked": len(examples),
            "examples": examples[:5],  # Truncate for storage
            "errors": errors if errors else None
        }


class AffineIntTransform(Transform):
    """
    Affine integer transform: x' = a*x + b
    
    Applies to all integers found in the input/output text.
    Only uses multipliers that preserve integrality (a divides result).
    """
    
    @property
    def name(self) -> str:
        return "affine_int"
    
    # Safe multipliers that are commonly used
    MULTIPLIERS = [2, 3, 5, 7, 11]
    
    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        a = rng.choice(self.MULTIPLIERS)
        b = rng.randint(-100, 100)
        return {"a": a, "b": b}
    
    def _find_integers(self, text: str) -> List[Tuple[int, int, str, int]]:
        """Find all integers in text with their positions.
        
        Returns list of (start, end, matched_str, int_value).
        """
        # Match integers (with optional sign, but not part of a float)
        pattern = r'(?<![.\d])(-?\d+)(?![.\d])'
        matches = []
        for m in re.finditer(pattern, text):
            try:
                val = int(m.group(1))
                matches.append((m.start(), m.end(), m.group(1), val))
            except ValueError:
                continue
        return matches
    
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        """Transform all integers: x -> a*x + b"""
        a, b = params["a"], params["b"]
        matches = self._find_integers(value)
        
        # Replace from end to preserve positions
        result = value
        for start, end, orig_str, val in reversed(matches):
            new_val = a * val + b
            result = result[:start] + str(new_val) + result[end:]
        
        return result
    
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        """Recover original integers: x' -> (x' - b) / a"""
        a, b = params["a"], params["b"]
        matches = self._find_integers(value)
        
        # Replace from end to preserve positions
        result = value
        for start, end, orig_str, val in reversed(matches):
            # Check if this would produce an integer
            if (val - b) % a != 0:
                # Skip non-integer results (might be a value that wasn't transformed)
                continue
            orig_val = (val - b) // a
            result = result[:start] + str(orig_val) + result[end:]
        
        return result
    
    def contract(self, params: Dict[str, Any]) -> str:
        a, b = params["a"], params["b"]
        
        # Generate example
        example_orig = 10
        example_enc = a * example_orig + b
        
        sign_b = "+" if b >= 0 else "-"
        abs_b = abs(b)
        
        return f"""### Encoding Contract (must follow exactly)
All integers in the input have been encoded using the formula: x' = {a}*x {sign_b} {abs_b}
To decode an input integer: x = (x' {'-' if b >= 0 else '+'} {abs_b}) / {a}
Your output integers must also be encoded the same way.

Examples:
- An input value that appears as {example_enc} represents the original value {example_orig}
- If your computed answer is {example_orig}, you must output {example_enc}
- If your computed answer is 0, you must output {b}
"""


class LabelPermutationTransform(Transform):
    """
    Bijective token/label remapping.
    
    Discovers discrete labels in the text and creates a permutation map.
    """
    
    @property
    def name(self) -> str:
        return "label_permutation"
    
    # Default label alphabet
    DEFAULT_LABELS = list(string.ascii_uppercase[:10])  # A-J
    
    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        labels = self.DEFAULT_LABELS.copy()
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        
        # Create forward and inverse maps
        forward_map = dict(zip(labels, shuffled))
        inverse_map = dict(zip(shuffled, labels))
        
        return {
            "forward_map": forward_map,
            "inverse_map": inverse_map,
            "labels": labels
        }
    
    def _apply_map(self, text: str, mapping: Dict[str, str]) -> str:
        """Apply character-level mapping."""
        result = []
        for char in text:
            result.append(mapping.get(char, char))
        return "".join(result)
    
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        return self._apply_map(value, params["forward_map"])
    
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        return self._apply_map(value, params["inverse_map"])
    
    def contract(self, params: Dict[str, Any]) -> str:
        fmap = params["forward_map"]
        # Show a few mappings
        examples = list(fmap.items())[:5]
        mapping_str = ", ".join(f"{k}→{v}" for k, v in examples)
        
        return f"""### Encoding Contract (must follow exactly)
Labels in the input have been permuted using a fixed bijection.
Mapping (showing first 5): {mapping_str}
Apply the same mapping to your output labels.

Examples:
- If you see '{examples[0][1]}' in input, it represents original label '{examples[0][0]}'
- If your answer should be '{examples[0][0]}', output '{examples[0][1]}' instead
"""


class BaseKIntTransform(Transform):
    """
    Encode integers as base-k strings (e.g., base-16 or base-36).
    
    Integers are encoded with a prefix (e.g., 'x') for unambiguous parsing.
    """
    
    @property
    def name(self) -> str:
        return "basek_int"
    
    BASES = [16, 36]
    DIGITS = string.digits + string.ascii_lowercase
    
    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        base = rng.choice(self.BASES)
        prefix = rng.choice(["#", "@", "x"])
        return {"base": base, "prefix": prefix}
    
    def _int_to_basek(self, n: int, base: int) -> str:
        """Convert integer to base-k string."""
        if n == 0:
            return "0"
        
        sign = ""
        if n < 0:
            sign = "-"
            n = -n
        
        digits = []
        while n:
            digits.append(self.DIGITS[n % base])
            n //= base
        
        return sign + "".join(reversed(digits))
    
    def _basek_to_int(self, s: str, base: int) -> int:
        """Convert base-k string to integer."""
        sign = 1
        if s.startswith("-"):
            sign = -1
            s = s[1:]
        
        result = 0
        for char in s:
            result = result * base + self.DIGITS.index(char.lower())
        
        return sign * result
    
    def _find_integers(self, text: str) -> List[Tuple[int, int, str, int]]:
        """Find all integers in text."""
        pattern = r'(?<![.\d])(-?\d+)(?![.\d])'
        matches = []
        for m in re.finditer(pattern, text):
            try:
                val = int(m.group(1))
                matches.append((m.start(), m.end(), m.group(1), val))
            except ValueError:
                continue
        return matches
    
    def _find_encoded(self, text: str, prefix: str, base: int) -> List[Tuple[int, int, str, int]]:
        """Find all encoded integers in text."""
        # Match prefix followed by base-k digits
        if base <= 10:
            digit_pattern = r'\d'
        else:
            digit_pattern = r'[0-9a-zA-Z]'
        
        pattern = re.escape(prefix) + r'(-?' + digit_pattern + r'+)'
        matches = []
        for m in re.finditer(pattern, text):
            try:
                encoded = m.group(1)
                val = self._basek_to_int(encoded, base)
                matches.append((m.start(), m.end(), m.group(0), val))
            except (ValueError, IndexError):
                continue
        return matches
    
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        base = params["base"]
        prefix = params["prefix"]
        matches = self._find_integers(value)
        
        result = value
        for start, end, orig_str, val in reversed(matches):
            encoded = prefix + self._int_to_basek(val, base)
            result = result[:start] + encoded + result[end:]
        
        return result
    
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        base = params["base"]
        prefix = params["prefix"]
        matches = self._find_encoded(value, prefix, base)
        
        result = value
        for start, end, orig_str, val in reversed(matches):
            result = result[:start] + str(val) + result[end:]
        
        return result
    
    def contract(self, params: Dict[str, Any]) -> str:
        base = params["base"]
        prefix = params["prefix"]
        
        example_orig = 42
        example_enc = prefix + self._int_to_basek(example_orig, base)
        
        return f"""### Encoding Contract (must follow exactly)
All integers are encoded in base-{base} with prefix '{prefix}'.
To decode: remove prefix, parse as base-{base} integer.
Your output integers must also be encoded the same way.

Examples:
- Input '{example_enc}' represents the integer {example_orig}
- If your answer is {example_orig}, output '{example_enc}'
- If your answer is 0, output '{prefix}0'
"""


class DelimiterRewriteTransform(Transform):
    """
    Rewrite delimiters in structured output.
    
    E.g., space-separated integers -> comma-separated in brackets.
    """
    
    @property
    def name(self) -> str:
        return "delimiter_rewrite"
    
    STYLES = [
        {"sep_in": " ", "sep_out": ",", "wrap": ("", "")},      # space -> comma
        {"sep_in": " ", "sep_out": ", ", "wrap": ("[", "]")},   # space -> [,]
        {"sep_in": "\n", "sep_out": ";", "wrap": ("", "")},     # newline -> semicolon
    ]
    
    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        style = rng.choice(self.STYLES)
        return style
    
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        sep_in = params["sep_in"]
        sep_out = params["sep_out"]
        wrap_l, wrap_r = params["wrap"]
        
        # Process each line separately for newline delimiter
        if sep_in == "\n":
            parts = value.split(sep_in)
            result = sep_out.join(parts)
        else:
            # Process line by line
            lines = value.split("\n")
            new_lines = []
            for line in lines:
                if sep_in in line:
                    parts = line.split(sep_in)
                    new_line = wrap_l + sep_out.join(parts) + wrap_r
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            result = "\n".join(new_lines)
        
        return result
    
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        sep_in = params["sep_in"]
        sep_out = params["sep_out"]
        wrap_l, wrap_r = params["wrap"]
        
        if sep_in == "\n":
            parts = value.split(sep_out)
            result = sep_in.join(parts)
        else:
            lines = value.split("\n")
            new_lines = []
            for line in lines:
                # Remove wrapping if present
                if wrap_l and wrap_r and line.startswith(wrap_l) and line.endswith(wrap_r):
                    line = line[len(wrap_l):-len(wrap_r) if wrap_r else len(line)]
                
                if sep_out in line:
                    parts = line.split(sep_out)
                    new_line = sep_in.join(parts)
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            result = "\n".join(new_lines)
        
        return result
    
    def contract(self, params: Dict[str, Any]) -> str:
        sep_in = params["sep_in"]
        sep_out = params["sep_out"]
        wrap_l, wrap_r = params["wrap"]
        
        sep_in_name = {"": "empty", " ": "space", "\n": "newline", ",": "comma", ";": "semicolon"}.get(sep_in, repr(sep_in))
        sep_out_name = {"": "empty", " ": "space", "\n": "newline", ",": "comma", ", ": "comma-space", ";": "semicolon"}.get(sep_out, repr(sep_out))
        
        example_orig = "1 2 3" if sep_in == " " else "1\n2\n3"
        example_enc = self.forward(example_orig, params)
        example_orig_display = example_orig.replace("\n", "\\n")
        
        wrap_note = f"Values are wrapped in {repr(wrap_l)} and {repr(wrap_r)}." if wrap_l else ""
        
        return f"""### Encoding Contract (must follow exactly)
Values are delimited by {sep_out_name} instead of {sep_in_name}.
{wrap_note}
Your output must use the same delimiter format.

Examples:
- Input "{example_enc}" represents "{example_orig_display}"
- Format your output the same way
"""


class IdentityTransform(Transform):
    """
    Identity transform (no change) for control experiments.
    """
    
    @property
    def name(self) -> str:
        return "identity"
    
    def sample_params(self, seed: int) -> Dict[str, Any]:
        return {"identity": True}
    
    def forward(self, value: str, params: Dict[str, Any]) -> str:
        return value
    
    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        return value
    
    def contract(self, params: Dict[str, Any]) -> str:
        return """### Encoding Contract (must follow exactly)
Input and output use standard formatting (no encoding applied).
Process values as-is.
"""


# Registry of all transforms
TRANSFORMS: Dict[str, Transform] = {
    "affine_int": AffineIntTransform(),
    "label_permutation": LabelPermutationTransform(),
    "basek_int": BaseKIntTransform(),
    "delimiter_rewrite": DelimiterRewriteTransform(),
    "identity": IdentityTransform(),
}


def get_transform(name: str) -> Transform:
    """Get a transform by name."""
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]


def list_transforms() -> List[str]:
    """List available transform names."""
    return list(TRANSFORMS.keys())
