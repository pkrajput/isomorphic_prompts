#!/usr/bin/env python3
"""
gen_models.py - Generate N completions per problem using Gemini or HuggingFace models.

Part 2 of the I/O Isomorphism Pipeline.
Consumes iso-transformed datasets from ./datasets/*_iso/ and produces
streaming JSONL with generated solutions.

Supported Models:
- Gemini: gemini-2.0-flash, gemini-2.5-pro (via Google API)
- HuggingFace: starcoder-15b, codestral-22b (local GPU inference)

Features:
- Retry with exponential backoff for transient failures (Gemini)
- Batched inference for HuggingFace models (single GPU)
- Model-aware default workers and batch sizes
- Detailed error classification for debugging
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import yaml
from tqdm import tqdm

# === CONSTANTS ===

# Prompt wrapper for iso-transformed datasets (includes encoding contract reference)
PROMPT_WRAPPER_ISO = """Return ONLY valid Python code.
No Markdown. No explanations. No triple backticks.
Follow the Encoding Contract exactly.
Implement an efficient solution.

"""

# Prompt wrapper for original datasets (no encoding contract)
PROMPT_WRAPPER_ORIGINAL = """Return ONLY valid Python code.
No Markdown. No explanations. No triple backticks.
Implement an efficient solution.

"""

# Config for ISO-transformed datasets
DATASETS_CONFIG_ISO = {
    "bigobench": {
        "data_dir": "bigobench_iso",
        "prompt_field": "prompt_with_contract",
        "fallback_prompt_fields": ["description", "prompt"],
        "problem_id_field": "problem_id",
        "starter_code_field": None,
    },
    "effibench": {
        "data_dir": "effibench_iso",
        "prompt_field": "prompt_with_contract",
        "fallback_prompt_fields": ["prompt"],
        "problem_id_field": "id",
        "starter_code_field": "starter_code_python3",  # Must complete class method
    },
    "mbpp": {
        "data_dir": "mbpp_iso",
        "prompt_field": "prompt_with_contract",
        "fallback_prompt_fields": ["prompt"],
        "problem_id_field": "task_id",
        "starter_code_field": None,
    },
}

# Config for ORIGINAL (non-perturbed) datasets
DATASETS_CONFIG_ORIGINAL = {
    "bigobench": {
        "data_dir": "BigOBench",
        "prompt_field": "description",
        "fallback_prompt_fields": ["prompt"],
        "problem_id_field": "problem_id",
        "starter_code_field": None,
    },
    "effibench": {
        "data_dir": "Effibench",
        "prompt_field": "prompt",
        "fallback_prompt_fields": [],
        "problem_id_field": "id",
        "starter_code_field": "starter_code_python3",  # Must complete class method
    },
    "mbpp": {
        "data_dir": "mbpp",
        "prompt_field": "prompt",
        "fallback_prompt_fields": [],
        "problem_id_field": "task_id",
        "starter_code_field": None,
    },
}

# Model configurations
MODEL_CONFIGS = {
    # Gemini models (API-based)
    "gemini-2.0-flash": {
        "backend": "gemini",
        "model_id": "gemini-2.0-flash",
        "workers": 50,
        "timeout": 120,
        "batch_size": 1,  # API-based, no batching
    },
    "gemini-2.5-pro": {
        "backend": "gemini",
        "model_id": "gemini-2.5-pro",
        "workers": 8,   # Much lower for Pro to avoid rate limits
        "timeout": 300, # Longer timeout for Pro
        "batch_size": 1,
    },
    # HuggingFace models (local GPU inference)
    "starcoder-15b": {
        "backend": "huggingface",
        "model_id": "bigcode/starcoder",
        "workers": 1,       # Sequential for GPU
        "timeout": 300,
        "batch_size": 4,    # Batch for efficiency
        "max_new_tokens": 2048,
        "use_chat_template": False,  # Completion model, no chat template
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
    },
    "codestral-22b": {
        "backend": "huggingface",
        "model_id": "mistralai/Codestral-22B-v0.1",
        "workers": 1,
        "timeout": 300,
        "batch_size": 2,    # Smaller batch due to model size
        "max_new_tokens": 2048,
        "use_chat_template": True,  # Chat model with proper template
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "device_map": "auto",
    },
}

# Retry configuration (for Gemini)
RETRY_CONFIG = {
    "max_retries": 6,
    "base_delay_s": 1.0,
    "max_delay_s": 60.0,
    "jitter_factor": 0.3,
}

# Error patterns that should trigger a retry
RETRYABLE_ERRORS = [
    "No content in response",
    "No candidates in response",
    "429",
    "quota",
    "rate limit",
    "Resource exhausted",
    "503",
    "500",
    "internal",
    "Deadline Exceeded",
    "504",
    "timeout",
    "UNAVAILABLE",
    "Connection",
    "reset by peer",
]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sha256_hex(text: str) -> str:
    """Compute SHA256 hex digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_python_code(raw_text: str) -> str:
    """
    Extract Python code from LLM output.
    Handles markdown code fences and returns cleaned code.
    """
    if not raw_text:
        return ""
    
    # Try to extract from code fences
    # Pattern: ```python ... ``` or ``` ... ```
    fence_patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    
    for pattern in fence_patterns:
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the largest match (most likely the main code)
            return max(matches, key=len).strip()
    
    # No fences found, return as-is
    return raw_text.strip()


def validate_python_ast(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python code by parsing AST.
    Returns (is_valid, error_message).
    """
    if not code:
        return False, "Empty code"
    
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"AST_PARSE_FAIL: {e}"


def get_prompt_from_row(row: dict, config: dict) -> tuple[str, dict]:
    """
    Extract prompt from row, following field priority.
    Returns (prompt, warnings_dict).
    """
    warnings = {}
    
    # Primary field
    prompt = row.get(config["prompt_field"], "")
    if prompt and isinstance(prompt, str) and prompt.strip():
        return prompt.strip(), warnings
    
    # Fallback fields
    warnings["prompt_source"] = "fallback"
    for field in config.get("fallback_prompt_fields", []):
        prompt = row.get(field, "")
        if prompt and isinstance(prompt, str) and prompt.strip():
            warnings["prompt_field_used"] = field
            return prompt.strip(), warnings
    
    warnings["prompt_missing"] = True
    return "", warnings


def get_problem_id(row: dict, source_file: str, row_index: int, config: dict) -> str:
    """
    Get stable problem_id from row.
    Falls back to hash of source_file:row_index if not present.
    """
    pid = row.get(config["problem_id_field"], "")
    if pid and isinstance(pid, str):
        return pid
    
    # Fallback: hash-based ID
    return sha256_hex(f"{source_file}:{row_index}")[:16]


def is_retryable_error(error_message: str) -> bool:
    """Check if the error should trigger a retry."""
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(pattern.lower() in error_lower for pattern in RETRYABLE_ERRORS)


def compute_backoff_delay(attempt: int) -> float:
    """
    Compute delay with exponential backoff and jitter.
    attempt is 0-indexed.
    """
    base = RETRY_CONFIG["base_delay_s"]
    max_delay = RETRY_CONFIG["max_delay_s"]
    jitter = RETRY_CONFIG["jitter_factor"]
    
    # Exponential backoff: base * 2^attempt
    delay = base * (2 ** attempt)
    delay = min(delay, max_delay)
    
    # Add jitter
    jitter_amount = delay * jitter * random.uniform(-1, 1)
    delay = max(0.1, delay + jitter_amount)
    
    return delay


# =============================================================================
# GEMINI BACKEND
# =============================================================================

def init_gemini(api_key: str, model_id: str):
    """Initialize Gemini client and model."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_id)


def generate_gemini_single(
    model,
    prompt: str,
    temperature: float,
    timeout: int,
) -> tuple[str, Optional[str], Optional[str], float, dict]:
    """
    Generate a single completion from Gemini (no retry).
    Returns (raw_output, finish_reason, error_message, latency_s, metadata).
    """
    import google.generativeai as genai
    
    start_time = time.time()
    metadata = {}
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096,
            ),
            request_options={"timeout": timeout},
        )
        
        latency = time.time() - start_time
        
        # Capture metadata for debugging
        if hasattr(response, 'prompt_feedback'):
            metadata["prompt_feedback"] = str(response.prompt_feedback)
        
        # Extract text
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            finish_reason = str(candidate.finish_reason.name) if candidate.finish_reason else None
            metadata["finish_reason"] = finish_reason
            
            # Check for safety blocks
            if hasattr(candidate, 'safety_ratings'):
                metadata["safety_ratings"] = str(candidate.safety_ratings)
            
            if candidate.content and candidate.content.parts:
                text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
                if text.strip():
                    return text, finish_reason, None, latency, metadata
                else:
                    return "", finish_reason, "Empty text in response", latency, metadata
            else:
                return "", finish_reason, "No content in response", latency, metadata
        else:
            metadata["candidate_count"] = 0
            return "", None, "No candidates in response", latency, metadata
            
    except Exception as e:
        latency = time.time() - start_time
        error_str = str(e)[:500]
        metadata["exception_type"] = type(e).__name__
        return "", None, f"API_ERROR: {error_str}", latency, metadata


def generate_gemini_with_retry(
    model,
    prompt: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> tuple[str, Optional[str], Optional[str], float, dict]:
    """
    Generate completion with retry and exponential backoff.
    Returns (raw_output, finish_reason, error_message, total_latency_s, metadata).
    """
    total_latency = 0.0
    last_error = None
    last_metadata = {}
    
    for attempt in range(max_retries + 1):
        raw_output, finish_reason, error_message, latency, metadata = generate_gemini_single(
            model, prompt, temperature, timeout
        )
        total_latency += latency
        last_metadata = metadata
        last_metadata["attempt"] = attempt + 1
        
        # Success: got non-empty text
        if raw_output and raw_output.strip():
            last_metadata["total_attempts"] = attempt + 1
            return raw_output, finish_reason, None, total_latency, last_metadata
        
        # Check if this error is retryable
        if error_message and is_retryable_error(error_message):
            last_error = error_message
            
            if attempt < max_retries:
                delay = compute_backoff_delay(attempt)
                time.sleep(delay)
                continue
        else:
            # Non-retryable error, return immediately
            last_metadata["total_attempts"] = attempt + 1
            return raw_output, finish_reason, error_message, total_latency, last_metadata
    
    # Exhausted retries
    last_metadata["total_attempts"] = max_retries + 1
    last_metadata["exhausted_retries"] = True
    final_error = f"RETRY_EXHAUSTED({max_retries + 1}): {last_error}"
    return "", None, final_error, total_latency, last_metadata


# =============================================================================
# HUGGINGFACE BACKEND
# =============================================================================

class HuggingFaceGenerator:
    """
    HuggingFace model wrapper for batched GPU inference.
    Supports both completion models (StarCoder) and chat models (Codestral).
    """
    
    def __init__(self, model_config: dict, hf_token: Optional[str] = None):
        self.model_config = model_config
        self.model_id = model_config["model_id"]
        self.use_chat_template = model_config.get("use_chat_template", False)
        self.max_new_tokens = model_config.get("max_new_tokens", 2048)
        self.trust_remote_code = model_config.get("trust_remote_code", True)
        
        # Lazy import transformers
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires: pip install transformers torch accelerate"
            )
        
        # Determine torch dtype
        torch_dtype_str = model_config.get("torch_dtype", "float16")
        if torch_dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        print(f"Loading model: {self.model_id}")
        print(f"  - torch_dtype: {torch_dtype_str}")
        print(f"  - use_chat_template: {self.use_chat_template}")
        print(f"  - max_new_tokens: {self.max_new_tokens}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            token=hf_token,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=model_config.get("device_map", "auto"),
            trust_remote_code=self.trust_remote_code,
            token=hf_token,
        )
        self.model.eval()
        
        print(f"Model loaded on device(s): {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'auto'}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt according to model's expected format."""
        if self.use_chat_template:
            # Use chat template for instruction-tuned models
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception:
                # Fallback: simple instruction format
                return f"[INST] {prompt} [/INST]"
        else:
            # Completion models: just use the prompt directly
            # For StarCoder, we can add a comment prefix for context
            return f"# Python code\n{prompt}\n\n"
    
    def generate_batch(
        self,
        prompts: list[str],
        temperature: float,
    ) -> list[tuple[str, Optional[str], Optional[str], float, dict]]:
        """
        Generate completions for a batch of prompts.
        Returns list of (raw_output, finish_reason, error_message, latency_s, metadata).
        """
        import torch
        
        start_time = time.time()
        results = []
        
        try:
            # Format prompts
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            
            # Tokenize with padding
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            # Track input lengths for extracting only new tokens
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            
            # Generate
            with torch.no_grad():
                # Set up generation config
                gen_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.95
                
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            
            latency = time.time() - start_time
            latency_per_sample = latency / len(prompts)
            
            # Decode outputs
            for i, (output_ids, input_len) in enumerate(zip(outputs, input_lengths)):
                # Extract only newly generated tokens
                new_tokens = output_ids[input_len:]
                raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Determine finish reason
                finish_reason = "STOP"
                if len(new_tokens) >= self.max_new_tokens:
                    finish_reason = "MAX_TOKENS"
                
                metadata = {
                    "batch_index": i,
                    "input_length": input_len,
                    "output_length": len(new_tokens),
                    "total_length": len(output_ids),
                }
                
                results.append((raw_output, finish_reason, None, latency_per_sample, metadata))
            
        except torch.cuda.OutOfMemoryError as e:
            latency = time.time() - start_time
            error_msg = f"CUDA_OOM: {str(e)[:200]}"
            for _ in prompts:
                results.append(("", None, error_msg, latency / len(prompts), {"exception_type": "OutOfMemoryError"}))
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"HF_ERROR: {str(e)[:500]}"
            for _ in prompts:
                results.append(("", None, error_msg, latency / len(prompts), {"exception_type": type(e).__name__}))
        
        return results


# =============================================================================
# TASK PROCESSING
# =============================================================================

def process_gemini_task(
    model,
    task: dict,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict:
    """Process a single Gemini generation task with retry support."""
    prompt_used = task["prompt_used"]
    
    # Generate with retry
    raw_output, finish_reason, error_message, latency, metadata = generate_gemini_with_retry(
        model, prompt_used, temperature, timeout, max_retries
    )
    
    # Extract and validate code
    generated_solution = ""
    if not error_message and raw_output:
        generated_solution = extract_python_code(raw_output)
        is_valid, ast_error = validate_python_ast(generated_solution)
        if not is_valid:
            # Keep the code but mark the error
            if error_message:
                error_message = f"{error_message}; {ast_error}"
            else:
                error_message = ast_error
    
    # Build output record
    return build_record(task, temperature, prompt_used, raw_output, generated_solution,
                        finish_reason, error_message, latency, metadata)


def process_huggingface_batch(
    generator: HuggingFaceGenerator,
    tasks: list[dict],
    temperature: float,
) -> list[dict]:
    """Process a batch of HuggingFace generation tasks."""
    prompts = [task["prompt_used"] for task in tasks]
    
    # Generate batch
    results = generator.generate_batch(prompts, temperature)
    
    records = []
    for task, (raw_output, finish_reason, error_message, latency, metadata) in zip(tasks, results):
        prompt_used = task["prompt_used"]
        
        # Extract and validate code
        generated_solution = ""
        if not error_message and raw_output:
            generated_solution = extract_python_code(raw_output)
            is_valid, ast_error = validate_python_ast(generated_solution)
            if not is_valid:
                if error_message:
                    error_message = f"{error_message}; {ast_error}"
                else:
                    error_message = ast_error
        
        record = build_record(task, temperature, prompt_used, raw_output, generated_solution,
                              finish_reason, error_message, latency, metadata)
        records.append(record)
    
    return records


def build_record(
    task: dict,
    temperature: float,
    prompt_used: str,
    raw_output: str,
    generated_solution: str,
    finish_reason: Optional[str],
    error_message: Optional[str],
    latency: float,
    metadata: dict,
) -> dict:
    """Build output record from generation results."""
    return {
        "dataset": task["dataset"],
        "model": task["model"],
        "temperature": temperature,
        "n_generations": task["n_generations"],
        "problem_id": task["problem_id"],
        "source_file": task["source_file"],
        "row_index": task["row_index"],
        "completion_index": task["completion_index"],
        "prompt_used": prompt_used,
        "prompt_hash": sha256_hex(prompt_used),
        "generated_solution": generated_solution,
        "raw_llm_output": raw_output,
        "finish_reason": finish_reason,
        "error_message": error_message,
        "latency_s": latency,
        "is_original": task.get("is_original", False),
        "iso_applied": task.get("iso_applied"),
        "iso_family": task.get("iso_family"),
        "iso_seed": task.get("iso_seed"),
        "iso_params": task.get("iso_params"),
        "row_meta": task.get("row_meta", {}),
        "generation_metadata": metadata,
    }


def load_existing_completions(output_path: str, skip_errors: bool = False) -> set:
    """
    Load existing (problem_id, completion_index) pairs from output JSONL.
    Used for resume support.
    """
    existing = set()
    if not os.path.exists(output_path):
        return existing
    
    try:
        with open(output_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    
                    # Optionally skip error records for retry
                    if skip_errors:
                        if rec.get("error_message") or not rec.get("generated_solution"):
                            continue
                    
                    key = (rec.get("problem_id", ""), rec.get("completion_index", -1))
                    existing.add(key)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return existing


def build_tasks(
    dataset: str,
    datasets_root: str,
    model_name: str,
    n_generations: int,
    max_problems: Optional[int],
    resume_set: set,
    only_iso_applied: bool,
    use_original: bool = False,
) -> list[dict]:
    """Build list of generation tasks from dataset files."""
    # Select config based on original vs iso
    if use_original:
        config = DATASETS_CONFIG_ORIGINAL[dataset]
        prompt_wrapper = PROMPT_WRAPPER_ORIGINAL
    else:
        config = DATASETS_CONFIG_ISO[dataset]
        prompt_wrapper = PROMPT_WRAPPER_ISO
    
    data_dir = Path(datasets_root) / config["data_dir"]
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    tasks = []
    problem_count = 0
    
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    
    for jsonl_path in jsonl_files:
        source_file = jsonl_path.name
        
        with open(jsonl_path, "r") as f:
            for row_index, line in enumerate(f):
                if max_problems and problem_count >= max_problems:
                    break
                
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Check iso_applied filter (only applies to iso datasets)
                iso_applied = row.get("iso_applied", None) if not use_original else None
                if not use_original and only_iso_applied and not iso_applied:
                    continue
                
                # Get prompt
                prompt, warnings = get_prompt_from_row(row, config)
                if not prompt:
                    warnings["skipped"] = "no_prompt"
                    continue
                
                # Get problem_id
                problem_id = get_problem_id(row, source_file, row_index, config)
                
                # Build full prompt with appropriate wrapper
                prompt_used = prompt_wrapper + prompt
                
                # Append starter code if available (for EffiBench-style datasets)
                starter_code_field = config.get("starter_code_field")
                if starter_code_field:
                    starter_code = row.get(starter_code_field, "")
                    if starter_code and starter_code.strip():
                        prompt_used += f"\n\nComplete the following code. Return the ENTIRE class/function with your implementation:\n\n```python\n{starter_code.strip()}\n    # Your implementation here\n```"
                
                # For MBPP: extract function name from test_list and add to prompt
                test_list = row.get("test_list", [])
                if test_list and dataset == "mbpp":
                    # Extract function name from first test: "assert func_name(...) == ..."
                    first_test = test_list[0] if isinstance(test_list, list) else str(test_list).split(",")[0]
                    match = re.search(r"assert\s+(\w+)\s*\(", first_test)
                    if match:
                        func_name = match.group(1)
                        prompt_used += f"\n\nYour function must be named `{func_name}`.\n\nExample test:\n```python\n{first_test}\n```"
                
                # Create tasks for each completion
                for comp_idx in range(n_generations):
                    key = (problem_id, comp_idx)
                    if key in resume_set:
                        continue  # Skip already completed
                    
                    task = {
                        "dataset": dataset,
                        "model": model_name,
                        "n_generations": n_generations,
                        "problem_id": problem_id,
                        "source_file": source_file,
                        "row_index": row_index,
                        "completion_index": comp_idx,
                        "prompt_used": prompt_used,
                        "is_original": use_original,
                        "iso_applied": iso_applied,
                        "iso_family": row.get("iso_family") if not use_original else None,
                        "iso_seed": row.get("iso_seed") if not use_original else None,
                        "iso_params": row.get("iso_params") if not use_original else None,
                        "row_meta": warnings,
                    }
                    tasks.append(task)
                
                problem_count += 1
        
        if max_problems and problem_count >= max_problems:
            break
    
    return tasks


def get_output_path(
    out_dir: str,
    model: str,
    dataset: str,
    source_file: str,
    temperature: float,
    n_generations: int,
    use_original: bool = False,
) -> str:
    """Build output path following the required naming convention."""
    if use_original:
        dataset_subdir = DATASETS_CONFIG_ORIGINAL[dataset]["data_dir"].lower()
    else:
        dataset_subdir = DATASETS_CONFIG_ISO[dataset]["data_dir"]
    
    basename = source_file.replace(".jsonl", "").replace(".iso", "")
    filename = f"{basename}__temp{temperature}__n{n_generations}.jsonl"
    
    path = Path(out_dir) / model / dataset_subdir / "generations" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_gemini_pipeline(
    args,
    config: dict,
    model_config: dict,
    tasks: list[dict],
    output_path: str,
):
    """Run generation pipeline with Gemini backend."""
    api_key = config.get("api_key")
    if not api_key:
        print("ERROR: api_key not found in config", file=sys.stderr)
        sys.exit(1)
    
    workers = args.workers if args.workers is not None else model_config.get("workers", 32)
    timeout = args.timeout if args.timeout is not None else model_config.get("timeout", 180)
    
    model = init_gemini(api_key, model_config["model_id"])
    
    completed = 0
    errors = 0
    retried = 0
    
    with open(output_path, "a") as out_file:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_gemini_task, model, task, args.temperature, timeout, args.max_retries
                ): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Generating") as pbar:
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
                        
                        if record.get("error_message"):
                            errors += 1
                        
                        # Track retried requests
                        gen_meta = record.get("generation_metadata", {})
                        if gen_meta.get("total_attempts", 1) > 1:
                            retried += 1
                        
                        completed += 1
                        
                    except Exception as e:
                        task = futures[future]
                        error_record = build_record(
                            task, args.temperature, task["prompt_used"],
                            "", "", None, f"WORKER_ERROR: {str(e)[:500]}", 0.0,
                            {"exception_type": type(e).__name__}
                        )
                        out_file.write(json.dumps(error_record) + "\n")
                        out_file.flush()
                        errors += 1
                    
                    pbar.update(1)
    
    print(f"\nCompleted: {completed}")
    print(f"Errors: {errors} ({100*errors/max(1,completed):.1f}%)")
    print(f"Retried: {retried} ({100*retried/max(1,completed):.1f}%)")


def run_huggingface_pipeline(
    args,
    config: dict,
    model_config: dict,
    tasks: list[dict],
    output_path: str,
):
    """Run generation pipeline with HuggingFace backend (batched)."""
    hf_token = config.get("hf_token")  # Optional, for gated models
    
    batch_size = args.batch_size if args.batch_size is not None else model_config.get("batch_size", 4)
    
    print(f"Initializing HuggingFace model: {model_config['model_id']}")
    generator = HuggingFaceGenerator(model_config, hf_token)
    
    completed = 0
    errors = 0
    
    # Process in batches
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    
    with open(output_path, "a") as out_file:
        with tqdm(total=len(tasks), desc="Generating") as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                try:
                    records = process_huggingface_batch(generator, batch_tasks, args.temperature)
                    
                    for record in records:
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
                        
                        if record.get("error_message"):
                            errors += 1
                        
                        completed += 1
                        pbar.update(1)
                        
                except Exception as e:
                    for task in batch_tasks:
                        error_record = build_record(
                            task, args.temperature, task["prompt_used"],
                            "", "", None, f"BATCH_ERROR: {str(e)[:500]}", 0.0,
                            {"exception_type": type(e).__name__}
                        )
                        out_file.write(json.dumps(error_record) + "\n")
                        out_file.flush()
                        errors += 1
                        completed += 1
                        pbar.update(1)
    
    print(f"\nCompleted: {completed}")
    print(f"Errors: {errors} ({100*errors/max(1,completed):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate N completions per problem using Gemini or HuggingFace models."
    )
    parser.add_argument("--dataset", required=True, choices=["bigobench", "effibench", "mbpp"])
    parser.add_argument("--datasets_root", default="./datasets")
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--model", required=True, 
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use for generation")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_generations", type=int, default=5)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for Gemini (default: model-specific)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for HuggingFace (default: model-specific)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="API timeout in seconds for Gemini (default: model-specific)")
    parser.add_argument("--max_retries", type=int, default=6,
                        help="Max retries for transient failures (Gemini only, default: 6)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output, skip all completed entries")
    parser.add_argument("--retry_errors", action="store_true",
                        help="With --resume, also retry entries that had errors")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only_iso_applied", action="store_true")
    parser.add_argument("--original", action="store_true",
                        help="Use original (non-perturbed) datasets instead of ISO-transformed")
    
    args = parser.parse_args()
    
    # Get model config
    model_config = MODEL_CONFIGS[args.model]
    backend = model_config["backend"]
    
    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Select dataset config based on --original flag
    use_original = args.original
    if use_original:
        dataset_config = DATASETS_CONFIG_ORIGINAL[args.dataset]
    else:
        dataset_config = DATASETS_CONFIG_ISO[args.dataset]
    
    # Determine output path (using first JSONL file for naming)
    data_dir = Path(args.datasets_root) / dataset_config["data_dir"]
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No JSONL files in {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    source_file = jsonl_files[0].name
    output_path = get_output_path(
        args.out_dir, args.model, args.dataset,
        source_file, args.temperature, args.n_generations,
        use_original=use_original
    )
    
    mode_str = "ORIGINAL" if use_original else "ISO"
    print(f"Dataset: {args.dataset} ({mode_str})")
    print(f"Model: {args.model} (backend: {backend})")
    print(f"Temperature: {args.temperature}")
    print(f"N generations: {args.n_generations}")
    print(f"Output: {output_path}")
    
    # Load resume set
    resume_set = set()
    if args.resume:
        skip_errors = args.retry_errors
        resume_set = load_existing_completions(output_path, skip_errors=skip_errors)
        if args.retry_errors:
            print(f"Resume: {len(resume_set)} successful completions (will retry errors)")
        else:
            print(f"Resume: {len(resume_set)} completions already done")
    
    # Build tasks
    tasks = build_tasks(
        args.dataset, args.datasets_root, args.model,
        args.n_generations, args.max_problems,
        resume_set, args.only_iso_applied,
        use_original=use_original
    )
    
    print(f"Tasks to process: {len(tasks)}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for task in tasks[:3]:
            print(f"\n--- Problem: {task['problem_id']}, Completion: {task['completion_index']} ---")
            print(f"Prompt (first 500 chars):\n{task['prompt_used'][:500]}...")
        print(f"\n... and {len(tasks) - 3} more tasks")
        return
    
    if not tasks:
        print("No tasks to process. Exiting.")
        return
    
    # Run appropriate pipeline
    if backend == "gemini":
        run_gemini_pipeline(args, config, model_config, tasks, output_path)
    else:
        run_huggingface_pipeline(args, config, model_config, tasks, output_path)
    
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
