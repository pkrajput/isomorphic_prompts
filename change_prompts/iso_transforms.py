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


class CubicIntTransform(Transform):
    """
    Cubic integer transform: x' = x³ + c

    Bijective over the integers (cube is strictly monotone and the cube
    root of an integer is unique).  Targets the *nonlinear arithmetic*
    axis: the model must compute a cube root to decode, which is
    qualitatively harder than the linear inverse of affine.
    """

    @property
    def name(self) -> str:
        return "cubic_int"

    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        c = rng.randint(-50, 50)
        return {"c": c}

    def _find_integers(self, text: str) -> List[Tuple[int, int, str, int]]:
        pattern = r'(?<![.\d])(-?\d+)(?![.\d])'
        matches = []
        for m in re.finditer(pattern, text):
            try:
                val = int(m.group(1))
                matches.append((m.start(), m.end(), m.group(1), val))
            except ValueError:
                continue
        return matches

    @staticmethod
    def _icbrt(n: int) -> int:
        """Integer cube root (exact). Raises ValueError if not a perfect cube."""
        if n == 0:
            return 0
        sign = 1 if n > 0 else -1
        a = abs(n)
        # Newton's method for integer cube root
        x = int(round(a ** (1 / 3)))
        # Check x-1, x, x+1 to handle floating-point imprecision
        for candidate in (x - 1, x, x + 1):
            if candidate >= 0 and candidate ** 3 == a:
                return sign * candidate
        raise ValueError(f"{n} is not a perfect cube")

    def forward(self, value: str, params: Dict[str, Any]) -> str:
        c = params["c"]
        matches = self._find_integers(value)
        result = value
        for start, end, orig_str, val in reversed(matches):
            new_val = val ** 3 + c
            result = result[:start] + str(new_val) + result[end:]
        return result

    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        c = params["c"]
        matches = self._find_integers(value)
        result = value
        for start, end, orig_str, val in reversed(matches):
            try:
                orig_val = self._icbrt(val - c)
                result = result[:start] + str(orig_val) + result[end:]
            except ValueError:
                continue
        return result

    def contract(self, params: Dict[str, Any]) -> str:
        c = params["c"]
        sign_c = "+" if c >= 0 else "-"
        abs_c = abs(c)

        example_orig = 5
        example_enc = example_orig ** 3 + c

        return f"""### Encoding Contract (must follow exactly)
All integers in the input have been encoded using the formula: x_enc = x_orig**3 {sign_c} {abs_c}
To decode an encoded input integer `x_enc`, use:
    val = x_enc {'-' if c >= 0 else '+'} {abs_c}
    x_orig = int(round(abs(val) ** (1/3))) * (1 if val >= 0 else -1)
To encode an original integer `x_orig`, use:
    x_enc = x_orig**3 {sign_c} {abs_c}
Your output integers must also be encoded the same way.

Examples:
- An input value that appears as {example_enc} represents the original value {example_orig}
- If your computed answer is {example_orig}, you must output {example_enc}
- If your computed answer is 0, you must output {c}
"""


class BaseKIntTransform(Transform):
    """
    Base-conversion codec: reinterpret each integer's decimal digit string
    as a base-b numeral and emit its decimal value.

    Example (base 16):
      42  →  digits "42"  →  interpret as hex 0x42  →  66 decimal
      66  →  digits "66"  →  interpret as hex 0x66  →  102 decimal  (NOT the inverse)

    Inverse: convert the decimal value TO base-b, read resulting digit
    string as a plain decimal number.
      66  →  to hex "42"  →  read as decimal  →  42  ✓

    The output is always a plain integer, so stdin/stdout parsing and
    Python assert syntax are never broken.
    """

    @property
    def name(self) -> str:
        return "basek_int"

    BASES = [16]
    DIGITS = string.digits + string.ascii_lowercase

    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        base = rng.choice(self.BASES)
        return {"base": base}

    def _decimal_digits_as_base(self, n: int, base: int) -> Optional[int]:
        """Reinterpret the decimal digit-string of *n* as a base-*base*
        numeral and return the resulting integer, or None if any digit
        is >= base."""
        sign = 1
        if n < 0:
            sign = -1
            n = -n
        s = str(n)
        for ch in s:
            if int(ch) >= base:
                return None
        return sign * int(s, base)

    def _int_to_base_digits(self, n: int, base: int) -> str:
        """Convert *n* to its base-*base* representation (digit chars only)."""
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

    def _find_integers(self, text: str) -> List[Tuple[int, int, str, int]]:
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
        base = params["base"]
        matches = self._find_integers(value)
        result = value
        for start, end, orig_str, val in reversed(matches):
            enc = self._decimal_digits_as_base(val, base)
            if enc is None:
                continue
            result = result[:start] + str(enc) + result[end:]
        return result

    def inverse(self, value: str, params: Dict[str, Any]) -> str:
        base = params["base"]
        matches = self._find_integers(value)
        result = value
        for start, end, orig_str, val in reversed(matches):
            dec_str = self._int_to_base_digits(val, base)
            if not all(c.isdigit() for c in dec_str.lstrip("-")):
                continue
            result = result[:start] + dec_str + result[end:]
        return result

    def contract(self, params: Dict[str, Any]) -> str:
        base = params["base"]
        example_orig = 42
        enc = self._decimal_digits_as_base(example_orig, base)
        zero_enc = self._decimal_digits_as_base(0, base)

        fmt_char = "x" if base == 16 else ("o" if base == 8 else "d")
        
        return f"""### Encoding Contract (must follow exactly)
All integers have been re-encoded: each digit string is reinterpreted as a base-{base} numeral.
To decode an encoded input integer `x_enc`, use:
    sign = "-" if x_enc < 0 else ""
    x_orig = int(f"{{sign}}{{abs(x_enc):{fmt_char}}}")
To encode an original integer `x_orig`, use:
    x_enc = int(str(x_orig), {base})
Your output integers must also be encoded the same way.

Examples:
- An input value {enc} represents the original value {example_orig} (because "{example_orig}" in base {base} = {enc} in decimal)
- If your computed answer is {example_orig}, you must output {enc}
- If your computed answer is 0, you must output {zero_enc}
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
    "cubic_int": CubicIntTransform(),
    "basek_int": BaseKIntTransform(),
    "base_conv": BaseKIntTransform(),
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
