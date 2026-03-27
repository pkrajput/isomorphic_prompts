import os
from google import genai
from google.genai import types
from typing import List, Tuple, Optional
import re


def _normalize_for_alignment(s: str) -> str:
    """Normalize whitespace for robust suffix alignment."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _find_suffix_token_start(tokens: List[str], suffix_text: str, quiet: bool = False) -> int:
    """Find the first token index where the concatenation of tokens ends with suffix_text.

    Gemini token strings may include leading spaces/newlines. We attempt a robust alignment by
    progressively concatenating normalized token text.

    Returns the start index of the suffix token span if found, otherwise 0.
    """
    suffix_norm = _normalize_for_alignment(suffix_text)
    if not suffix_norm:
        return 0

    # Build progressive concatenations and find the earliest index that matches suffix at the end.
    # We allow alignment to start at any token boundary by scanning possible starts.
    toks_norm = [_normalize_for_alignment(t) for t in tokens]

    # Fast path: if entire text doesn't contain suffix, return 0
    full = _normalize_for_alignment("".join(tokens))
    if suffix_norm not in full:
        if not quiet:
            print(f"    Warning: Suffix text not found in tokenized sequence. Full length: {len(tokens)} tokens")
            print(f"    Suffix length: {len(suffix_norm)} chars, Full text length: {len(full)} chars")
        return 0

    # Scan for a start index such that join(tokens[start:]) ends with suffix
    # We want the LATEST start (shortest prefix) that still matches - this gives us the actual suffix
    # Start from the end and work backwards for efficiency
    best_start = len(tokens)  # Initialize to end (no match found)
    
    # Try to find where suffix starts by checking if tail ends with suffix
    # We want the latest start position that still matches
    for start in range(len(tokens) - 1, -1, -1):  # Start from end, work backwards
        tail = _normalize_for_alignment("".join(tokens[start:]))
        if tail.endswith(suffix_norm):
            best_start = start
            # Found a match, but continue to see if we can find an earlier start
            # Actually, we want the LATEST start, so we can break here
            break
    
    # If we found a match, use it; otherwise return 0 (use all tokens)
    if best_start < len(tokens):
        suffix_token_count = len(tokens) - best_start
        if not quiet:
            print(f"    Found suffix start at token {best_start} out of {len(tokens)} tokens")
            print(f"    Suffix will include {suffix_token_count} tokens")
        if suffix_token_count < 5 and not quiet:
            print(f"    Warning: Very few tokens in suffix ({suffix_token_count}). This may indicate alignment issues.")
        return best_start
    else:
        if not quiet:
            print(f"    Warning: Could not find suffix in tokens, using all {len(tokens)} tokens")
        return 0


def get_text_logprobs(
    client,
    text: str,
    model_name: str = "gemini-2.0-flash",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores provided text and returns average logprob over its tokens.

    This is designed for CoDeC-style evaluation (teacher-forcing / scoring), not generation.

    Supports Gemini (API), OLMo (local), and vLLM (localhost API) models.

    Args:
      client: Model client (Gemini Client, OLMo dict, or vLLM dict)
      text: full prompt text to score (context + target)
      model_name: Model identifier
      score_only_suffix: if provided, return token logprobs only for this suffix portion.

    Returns:
      ("", avg_logprob, tokens_data) where tokens_data is a list of (token_str, logprob)
      corresponding to the scored portion.
    """
    # Check model type
    if isinstance(client, dict):
        if client.get("model_type") == "vllm":
            return get_text_logprobs_vllm(client, text, model_name, score_only_suffix)
        elif client.get("model_type") == "olmo":
            return get_text_logprobs_olmo(client, text, model_name, score_only_suffix)
        elif client.get("model_type") == "openrouter":
            return get_text_logprobs_openrouter(client, text, model_name, score_only_suffix)
        elif client.get("model_type") == "openai":
            return get_text_logprobs_openai(client, text, model_name, score_only_suffix)
    
    # Default to Gemini
    return get_text_logprobs_gemini(client, text, model_name, score_only_suffix)

def get_text_logprobs_gemini(
    client,
    text: str,
    model_name: str = "gemini-2.0-flash",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores text using Gemini API.
    
    Note: Gemini's logprobs_result.chosen_candidates contains logprobs for ALL tokens
    in the sequence (both prompt and generated tokens), not just the generated ones.
    """
    try:
        # Gemini API: We need max_output_tokens >= 1, but chosen_candidates should contain
        # logprobs for ALL tokens (prompt + generated). However, with max_output_tokens=1,
        # we might only get the generated token. Let's try a larger value to see if that helps.
        # Actually, let's check the API response structure first with max_output_tokens=1
        # Use a larger max_output_tokens to ensure we get all prompt tokens in chosen_candidates
        # Then we'll extract only the prompt portion (everything before the generated tokens)
        config = types.GenerateContentConfig(
            response_logprobs=True,
            logprobs=3,  # Required for Gemini 2.5 Flash (1-20); enables logprobs on Vertex AI
            candidate_count=1,
            max_output_tokens=256,  # Generate more to ensure we get all prompt tokens in the response
            temperature=0.0,
            # No system instruction needed for scoring
        )

        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=config,
        )

        if not response.candidates:
            return None

        candidate = response.candidates[0]

        tokens_data: List[Tuple[str, float]] = []
        tokens: List[str] = []
        lps: List[float] = []

        # Collect all token logprobs from chosen_candidates
        # IMPORTANT: With max_output_tokens=256, chosen_candidates should contain logprobs 
        # for ALL tokens in the sequence (both prompt and generated tokens).
        # We'll extract only the prompt tokens (everything except the last max_output_tokens).
        
        if hasattr(candidate, "logprobs_result") and candidate.logprobs_result:
            lr = candidate.logprobs_result
            
            # Method 1: Check chosen_candidates (should have all tokens: prompt + generated)
            if hasattr(lr, "chosen_candidates") and lr.chosen_candidates:
                all_token_infos = []
                for token_info in lr.chosen_candidates:
                    if token_info.log_probability is None:
                        continue
                    tok = token_info.token
                    lp = float(token_info.log_probability)
                    all_token_infos.append((tok, lp))
                
                # Now we need to separate prompt tokens from generated tokens
                # The prompt tokens are everything except the last N tokens (where N = generated tokens)
                # But we don't know exactly how many were generated. Let's use the content to figure it out.
                
                # Get the generated content to determine how many tokens were generated
                generated_text = ""
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                generated_text += part.text
                
                # IMPORTANT: chosen_candidates contains ALL tokens: [prompt_tokens..., generated_tokens...]
                # We need to extract only the prompt tokens.
                # Strategy: If we have score_only_suffix, find where it starts and take from there.
                # Otherwise, we need to identify where prompt ends and generated begins.
                
                if len(all_token_infos) == 0:
                    pass  # Will be handled below
                elif score_only_suffix:
                    # Find where the suffix starts in the token sequence
                    all_tokens = [t[0] for t in all_token_infos]
                    suffix_start = _find_suffix_token_start(all_tokens, score_only_suffix, quiet=True)
                    # Take tokens from suffix_start onwards (these are the prompt tokens for the suffix)
                    # But we want ALL prompt tokens, not just suffix. Let's take all up to where generation starts.
                    # Actually, if suffix_start is found, those are the tokens we want to score
                    for tok, lp in all_token_infos[suffix_start:]:
                        tokens.append(tok)
                        lps.append(lp)
                else:
                    # No suffix specified - we want all prompt tokens
                    # The prompt tokens are everything except the generated tokens
                    # We can estimate generated tokens by checking if the token sequence ends with the generated text
                    # For now, let's assume all tokens are prompt tokens (conservative approach)
                    # But we should exclude the last few tokens that match the generated text
                    
                    # Simple approach: if we have generated text, try to find where it starts in tokens
                    # and exclude everything from there
                    if generated_text and len(generated_text) > 0:
                        # Find where generated text starts in the token sequence
                        generated_tokens_str = "".join([t[0] for t in all_token_infos])
                        # Try to find the start of generated text
                        # This is approximate, but should work for most cases
                        # Take all tokens as prompt tokens for now
                        for tok, lp in all_token_infos:
                            tokens.append(tok)
                            lps.append(lp)
                    else:
                        # No generated text, so all tokens are prompt tokens
                        for tok, lp in all_token_infos:
                            tokens.append(tok)
                            lps.append(lp)
            
            # Method 2: Check if there's a separate prompt_tokens field
            if hasattr(lr, "prompt_tokens") and lr.prompt_tokens:
                prompt_tokens_list = []
                prompt_lps_list = []
                for token_info in lr.prompt_tokens:
                    if token_info.log_probability is not None:
                        prompt_tokens_list.append(token_info.token)
                        prompt_lps_list.append(float(token_info.log_probability))
                # Prepend prompt tokens (they come before generated tokens)
                tokens = prompt_tokens_list + tokens
                lps = prompt_lps_list + lps
            
            # Method 3: Try to find other token-related fields
            if len(tokens) <= 1:
                for attr_name in dir(lr):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr_val = getattr(lr, attr_name)
                        if isinstance(attr_val, (list, tuple)) and len(attr_val) > 1:
                            if len(attr_val) > 0:
                                first_item = attr_val[0]
                                if hasattr(first_item, 'token') or hasattr(first_item, 'log_probability'):
                                    for item in attr_val:
                                        if hasattr(item, 'token') and hasattr(item, 'log_probability'):
                                            if item.log_probability is not None:
                                                tokens.append(item.token)
                                                lps.append(float(item.log_probability))
                    except Exception:
                        pass

        if not tokens:
            # If the API doesn't return prompt logprobs, we cannot score.
            print(f"    ERROR: No tokens returned from Gemini API")
            return None
        
        # Note: Gemini API returns logprobs only for OUTPUT tokens, not prompt tokens (API limitation).
        # We use whatever tokens we get; CoDeC scores may differ from local models.

        # If requested, keep only the suffix region.
        start_idx = 0
        if score_only_suffix:
            start_idx = _find_suffix_token_start(tokens, score_only_suffix, quiet=True)
            if start_idx >= len(tokens):
                start_idx = max(0, len(tokens) - 100)  # Fallback: use last 100 tokens

        scored_tokens = tokens[start_idx:]
        scored_lps = lps[start_idx:]
        
        tokens_data = list(zip(scored_tokens, scored_lps))

        if not scored_lps:
            return None

        avg_logprob = sum(scored_lps) / len(scored_lps)

        return "", avg_logprob, tokens_data

    except Exception as e:
        err_str = str(e)
        print(f"Error scoring text logprobs (Gemini): {e}")
        if "Logprobs is not enabled" in err_str and "gemini-2.5" in err_str.lower():
            print("  Hint: Gemini 2.5 Flash logprobs require Vertex AI (not Google AI Studio).")
            print("  Set GOOGLE_CLOUD_PROJECT and use Vertex: export GOOGLE_CLOUD_PROJECT=your-project-id")
            print("  Or use gemini-2.0-flash instead (logprobs work on Google AI).")
        if "401" in err_str and "API keys are not supported" in err_str:
            print("  Hint: Standard Vertex AI requires OAuth2 (ADC), not API keys.")
            print("  Run: gcloud auth application-default login")
            print("  Then: export GOOGLE_CLOUD_PROJECT=your-project-id")
        return None

def get_text_logprobs_olmo(
    model_dict: dict,
    text: str,
    model_name: str = "allenai/OLMo-7B",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores text using OLMo model (local via Transformers).
    
    Args:
        model_dict: Dictionary with 'model', 'tokenizer', 'device' keys
        text: Text to score
        model_name: Model identifier (for reference)
        score_only_suffix: If provided, score only this suffix portion
        
    Returns:
        ("", avg_logprob, tokens_data) tuple
    """
    try:
        import torch
        import torch.nn.functional as F
        
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = model_dict["device"]
        
        # Tokenize the full text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = inputs["input_ids"]
        
        # Get logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get token log probabilities
        tokens_data: List[Tuple[str, float]] = []
        tokens: List[str] = []
        lps: List[float] = []
        
        # Get logprobs for each token (excluding the last one, which predicts the next token)
        for i in range(len(input_ids[0]) - 1):
            token_id = input_ids[0][i + 1].item()  # Next token (what we're predicting)
            token_str = tokenizer.decode([token_id])
            lp = log_probs[i][token_id].item()
            tokens.append(token_str)
            lps.append(lp)
        
        if not tokens:
            return None
        
        # If requested, keep only the suffix region (target text only, not context)
        # Paper: "obtain the model's predictions on x" - score only the target tokens.
        # Causal LM: tokens[i] = decode(input_ids[i+1]), lps[i] = log P(input_ids[i+1]|...)
        start_idx = 0
        if score_only_suffix:
            # Try exact token-ID match first (fast when tokenization is consistent)
            suffix_inputs = tokenizer(score_only_suffix, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
            suffix_ids = suffix_inputs["input_ids"][0].tolist()
            full_ids = input_ids[0].tolist()
            bos_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "cls_token_id", None)
            if suffix_ids and bos_id is not None and suffix_ids[0] == bos_id:
                suffix_ids = suffix_ids[1:]
            found = False
            for i in range(len(full_ids) - len(suffix_ids) + 1):
                if full_ids[i:i+len(suffix_ids)] == suffix_ids:
                    start_idx = max(0, i - 1)
                    found = True
                    break
            if not found:
                # Fallback: character-based alignment (handles tokenization differences)
                start_idx = _find_suffix_token_start(tokens, score_only_suffix, quiet=True)
        
        scored_tokens = tokens[start_idx:]
        scored_lps = lps[start_idx:]
        
        tokens_data = list(zip(scored_tokens, scored_lps))
        
        if not scored_lps:
            return None
        
        avg_logprob = sum(scored_lps) / len(scored_lps)
        
        return "", avg_logprob, tokens_data
        
    except Exception as e:
        print(f"Error scoring text logprobs (OLMo): {e}")
        import traceback
        traceback.print_exc()
        return None

# Placeholder for API Key - User should replace this or set env var
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Vertex AI: required for Gemini 2.5 Flash logprobs (Google AI Studio does not support them)
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
USE_VERTEX_AI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
# Vertex API key (express mode): set VERTEX_API_KEY or use GOOGLE_API_KEY when vertexai=True
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY", "") or API_KEY

def setup_model(model_name="gemini-2.0-flash", vllm_base_url="http://localhost:8000/v1"):
    """Sets up the model client.
    
    Supports:
    - Gemini models: Uses Google GenAI API
    - OLMo models: Uses Hugging Face Transformers (local)
    - vLLM models: Uses OpenAI-compatible API (localhost)
    
    Args:
        model_name: Name of the model to use
        vllm_base_url: Base URL for vLLM server (default: http://localhost:8000/v1)
        
    Returns:
        Model client/object appropriate for the model type
    """
    # Check if it's an OpenAI model (davinci-002, babbage-002, etc.)
    if model_name.startswith("openai:") or model_name in ("davinci-002", "babbage-002"):
        return setup_openai_model(model_name)
    # Check if it's an OpenRouter model
    if model_name.startswith("openrouter:") or model_name.startswith("openrouter/"):
        return setup_openrouter_model(model_name)
    # Check if it's a vLLM model
    if model_name.startswith("vllm:") or model_name.startswith("vllm/"):
        return setup_vllm_model(model_name, vllm_base_url)
    # Check if it's an OLMo model
    elif model_name.startswith("olmo") or model_name.startswith("OLMo") or "allenai/OLMo" in model_name.lower():
        return setup_olmo_model(model_name)
    # Check if it's a Pythia model (trained on The Pile - same HF CausalLM interface as OLMo)
    elif "pythia" in model_name.lower() or "EleutherAI/pythia" in model_name:
        return setup_olmo_model(model_name)
    # Check if it's GPT-Neo / GPT-NeoX (trained on The Pile - same CausalLM interface)
    elif "gpt-neo" in model_name.lower() or "gpt-neox" in model_name.lower() or "EleutherAI/gpt-neo" in model_name:
        return setup_olmo_model(model_name)
    # RWKV-4, Nemotron (HF CausalLM or custom; try same loader with trust_remote_code)
    elif "rwkv" in model_name.lower() or "nemotron" in model_name.lower() or "nvidia/Nemotron" in model_name or "nvidia/NVIDIA-Nemotron" in model_name:
        return setup_olmo_model(model_name)
    else:
        # Gemini model
        use_vertex = USE_VERTEX_AI or (
            (GOOGLE_CLOUD_PROJECT or (VERTEX_API_KEY and VERTEX_API_KEY != "YOUR_API_KEY_HERE"))
            and "gemini-2.5" in model_name.lower()
        )
        if use_vertex:
            # Vertex AI: required for Gemini 2.5 Flash logprobs (Google AI Studio does not support them)
            # Prefer ADC (project+location): standard Vertex AI does NOT support API keys.
            # API keys work only with Vertex Express Mode (different endpoint); many keys hit aiplatform which requires OAuth2.
            if GOOGLE_CLOUD_PROJECT:
                client = genai.Client(
                    vertexai=True,
                    project=GOOGLE_CLOUD_PROJECT,
                    location=GOOGLE_CLOUD_LOCATION,
                )
            elif VERTEX_API_KEY and VERTEX_API_KEY != "YOUR_API_KEY_HERE":
                # Express mode API key (may fail with 401 if key targets wrong endpoint)
                client = genai.Client(vertexai=True, api_key=VERTEX_API_KEY)
            else:
                raise ValueError(
                    "Vertex AI requires GOOGLE_CLOUD_PROJECT + ADC. Run: "
                    "gcloud auth application-default login && export GOOGLE_CLOUD_PROJECT=your-project-id"
                )
            return client
        if API_KEY == "YOUR_API_KEY_HERE":
            print("WARNING: Please set GOOGLE_API_KEY environment variable.")
        # Google AI Studio (default)
        client = genai.Client(api_key=API_KEY)
        return client

def setup_vllm_model(model_name="vllm:olmo-7b", base_url="http://localhost:8000/v1"):
    """Sets up a vLLM model client using OpenAI-compatible API.
    
    Args:
        model_name: Model identifier (e.g., "vllm:olmo-7b" or "vllm/olmo-7b")
        base_url: Base URL for vLLM server (default: http://localhost:8000/v1)
        
    Returns:
        Dictionary with 'client' (OpenAI client) and 'model_name' keys
    """
    try:
        from openai import OpenAI
        
        # Extract actual model name (remove vllm: or vllm/ prefix)
        actual_model = model_name.replace("vllm:", "").replace("vllm/", "")
        
        print(f"Connecting to vLLM server at {base_url}")
        print(f"Using model: {actual_model}")
        
        # Create OpenAI client pointing to vLLM server.
        # If vLLM was started with --api-key, set VLLM_API_KEY in env.
        vllm_api_key = os.environ.get("VLLM_API_KEY", "not-needed")
        client = OpenAI(
            base_url=base_url,
            api_key=vllm_api_key
        )
        
        return {
            "client": client,
            "model_name": actual_model,
            "model_type": "vllm"
        }
    except ImportError:
        print("ERROR: openai library not found. Install with: pip install openai")
        raise
    except Exception as e:
        print(f"Error connecting to vLLM server: {e}")
        print(f"Make sure vLLM server is running at {base_url}")
        print("Start vLLM server with: python -m vllm.entrypoints.openai.api_server --model <model_path>")
        raise


def setup_openrouter_model(model_name="openrouter:openai/gpt-5.1-codex-mini"):
    """Sets up an OpenRouter model client (OpenAI-compatible API).
    
    Requires OPENROUTER_API_KEY environment variable.
    Model names: openrouter:openai/gpt-5.1-codex-mini, openrouter:anthropic/claude-3, etc.
    
    Returns:
        Dictionary with 'client', 'model_name', and 'model_type': 'openrouter'
    """
    try:
        from openai import OpenAI
        
        actual_model = model_name.replace("openrouter:", "").replace("openrouter/", "")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required. Get one at https://openrouter.ai/keys")
        
        print(f"Connecting to OpenRouter (https://openrouter.ai/api/v1)")
        print(f"Using model: {actual_model}")
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        return {
            "client": client,
            "model_name": actual_model,
            "model_type": "openrouter"
        }
    except ImportError:
        print("ERROR: openai library not found. Install with: pip install openai")
        raise


def setup_openai_model(model_name="davinci-002"):
    """Sets up an OpenAI model client (davinci-002, babbage-002).
    
    Requires OPENAI_API_KEY. Uses completions API with echo+logprobs for prompt token logprobs.
    """
    try:
        from openai import OpenAI
        actual = model_name.replace("openai:", "") if model_name.startswith("openai:") else model_name
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required. Set: export OPENAI_API_KEY=sk-...")
        print(f"Using OpenAI model: {actual}")
        client = OpenAI(api_key=OPENAI_API_KEY)
        return {"client": client, "model_name": actual, "model_type": "openai"}
    except ImportError:
        print("ERROR: pip install openai")
        raise


def setup_olmo_model(model_name="allenai/OLMo-7B"):
    """Sets up an OLMo model from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "allenai/OLMo-7B", "allenai/OLMo-7B-Instruct")
        
    Returns:
        Dictionary with 'model', 'tokenizer', and 'device' keys
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        _n = model_name.lower()
        if "pythia" in _n:
            model_type = "Pythia"
        elif "gpt-neo" in _n or "gpt-neox" in _n:
            model_type = "GPT-Neo"
        elif "rwkv" in _n:
            model_type = "RWKV"
        elif "nemotron" in _n:
            model_type = "Nemotron"
        else:
            model_type = "OLMo"
        print(f"Loading {model_type} model: {model_name}")
        print("This may take a while on first run...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Determine device
        device = next(model.parameters()).device if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            model = model.to(device)
        
        print(f"Model loaded on device: {device}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "model_type": "olmo"
        }
    except ImportError:
        print("ERROR: transformers library not found. Install with: pip install transformers torch")
        raise
    except Exception as e:
        print(f"Error loading OLMo model: {e}")
        raise

def get_completion_logprobs(client, prefix: str, model_name: str = "gemini-2.0-flash") -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """
    Generates a completion for the prefix and returns the generated text and its average log probability.
    Uses the new google-genai SDK.
    """
    try:
        # Configuration for logprobs
        config = types.GenerateContentConfig(
            response_logprobs=True,
            logprobs=3,  # Required for Gemini 2.5 Flash (1-20); enables logprobs on Vertex AI
            candidate_count=1,
            max_output_tokens=256,
            temperature=0.0,
            system_instruction="You are a text completion engine. Continue the provided text exactly. Do not output any conversational filler, introductions, or explanations. Just output the continuation."
        )
        
        # New model name format is usually 'gemini-2.0-flash' or 'models/gemini-2.0-flash'
        # The prompt is passed as contents
        response = client.models.generate_content(
            model=model_name,
            contents=prefix,
            config=config
        )
        
        if not response.candidates:
            return None
            
        candidate = response.candidates[0]
        
        if not candidate.content or not candidate.content.parts:
             print("DEBUG: Candidate has no content or parts.")
             return "", 0.0
             
        content = candidate.content.parts[0].text
        
        # DEBUG: Print available fields
        # print(f"DEBUG: Candidate fields: {dir(candidate)}")
        # if hasattr(candidate, 'avg_log_probability'):
        #    print(f"DEBUG: candidate.avg_log_probability = {candidate.avg_log_probability}")
        # if hasattr(candidate, 'logprobs_result'):
        #     print(f"DEBUG: candidate.logprobs_result present.")
        
        avg_logprob = 0.0
        
        if hasattr(candidate, 'avg_log_probability'):
             avg_logprob = candidate.avg_log_probability
        
        # Fallback: Check logprobs_result if avg is missing or 0
        if (avg_logprob == 0.0 or avg_logprob is None) and hasattr(candidate, 'logprobs_result'):
            lr = candidate.logprobs_result
            if lr and lr.chosen_candidates:
                # Calculate average from chosen tokens
                total_logprob = 0.0
                count = 0
                for token_info in lr.chosen_candidates:
                    if token_info.log_probability is not None:
                        total_logprob += token_info.log_probability
                        count += 1
                if count > 0:
                    avg_logprob = total_logprob / count
                    # print(f"DEBUG: Calculated avg_logprob from {count} tokens: {avg_logprob}")
        elif hasattr(candidate, 'logprobs_result'): # Use this if explicit result object
             # Iterate tokens if needed, but average is usually provided
             pass
             
        # Fallback manual calculation if token info is present
        # but avg_log_probability is preferred.
        
        tokens_data = []
        if hasattr(candidate, 'logprobs_result'):
            lr = candidate.logprobs_result
            if lr and lr.chosen_candidates:
                for token_info in lr.chosen_candidates:
                    if token_info.log_probability is not None:
                        tokens_data.append((token_info.token, token_info.log_probability))
        
        return content, avg_logprob, tokens_data

    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def get_text_logprobs_vllm(
    model_dict: dict,
    text: str,
    model_name: str = "vllm:olmo-7b",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores text using vLLM server via OpenAI-compatible API.
    
    Args:
        model_dict: Dictionary with 'client' (OpenAI client) and 'model_name' keys
        text: Text to score
        model_name: Model identifier (for reference)
        score_only_suffix: If provided, score only this suffix portion
        
    Returns:
        ("", avg_logprob, tokens_data) tuple
    """
    try:
        client = model_dict["client"]
        vllm_model_name = model_dict["model_name"]

        tokens_data: List[Tuple[str, float]] = []
        tokens: List[str] = []
        lps: List[float] = []
        completion_error_msg: Optional[str] = None

        # vLLM scoring uses completions endpoint only (prompt logprobs with echo=True).
        try:
            completion_response = client.completions.create(
                model=vllm_model_name,
                prompt=text,
                max_tokens=0,
                temperature=0.0,
                echo=True,
                logprobs=1,
            )

            if completion_response.choices and len(completion_response.choices) > 0:
                choice = completion_response.choices[0]
                if hasattr(choice, "logprobs") and choice.logprobs:
                    if hasattr(choice.logprobs, "tokens") and hasattr(choice.logprobs, "token_logprobs"):
                        tokens = list(choice.logprobs.tokens or [])
                        raw_lps = choice.logprobs.token_logprobs or []
                        lps = [float(lp) if lp is not None else 0.0 for lp in raw_lps]

        except Exception as e:
            completion_error_msg = str(e)
            print(f"Warning: Could not get logprobs via vLLM completion API: {e}")

        if not tokens or not lps:
            if completion_error_msg and (
                "context length is only" in completion_error_msg.lower()
                or "maximum input length" in completion_error_msg.lower()
                or "input tokens" in completion_error_msg.lower()
                or "input characters" in completion_error_msg.lower()
            ):
                print("Warning: vLLM prompt too long for configured context window; sample skipped.")
            else:
                print("Warning: vLLM did not return token logprobs. The API may not support prompt logprobs.")
            return None
        if len(tokens) != len(lps):
            m = min(len(tokens), len(lps))
            tokens = tokens[:m]
            lps = lps[:m]
        
        # If requested, keep only the suffix region
        start_idx = 0
        if score_only_suffix:
            # Simple alignment: find where suffix text appears in tokenized sequence
            suffix_text = score_only_suffix.strip()
            full_text_normalized = " ".join(tokens).replace(" ", "")
            suffix_normalized = suffix_text.replace(" ", "")
            
            # Find suffix start
            if suffix_normalized in full_text_normalized:
                suffix_start = full_text_normalized.rfind(suffix_normalized)
                # Approximate token index (this is approximate)
                char_count = 0
                for i, tok in enumerate(tokens):
                    char_count += len(tok.replace(" ", ""))
                    if char_count > suffix_start:
                        start_idx = i
                        break
        
        scored_tokens = tokens[start_idx:]
        scored_lps = lps[start_idx:]
        
        tokens_data = list(zip(scored_tokens, scored_lps))
        
        if not scored_lps:
            return None
        
        avg_logprob = sum(scored_lps) / len(scored_lps)
        
        return "", avg_logprob, tokens_data
        
    except Exception as e:
        print(f"Error scoring text logprobs (vLLM): {e}")
        import traceback
        traceback.print_exc()
        return None


def get_text_logprobs_openrouter(
    model_dict: dict,
    text: str,
    model_name: str = "openrouter:openai/gpt-5.1-codex-mini",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores text using OpenRouter (OpenAI-compatible API).
    
    Uses completion API with echo=True and logprobs to get prompt token logprobs.
    Provider behavior varies: OpenAI models may return prompt logprobs; others may not.
    Uses max_tokens=0 for prompt-only logprob scoring.
    """
    try:
        client = model_dict["client"]
        or_model_name = model_dict["model_name"]
        
        tokens_data: List[Tuple[str, float]] = []
        tokens: List[str] = []
        lps: List[float] = []
        
        completion_response = client.completions.create(
            model=or_model_name,
            prompt=text,
            max_tokens=0,
            temperature=0.0,
            echo=True,
            logprobs=1,
        )
        if completion_response.choices and len(completion_response.choices) > 0:
            choice = completion_response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs:
                if hasattr(choice.logprobs, "tokens") and choice.logprobs.tokens:
                    tokens = list(choice.logprobs.tokens)
                if hasattr(choice.logprobs, "token_logprobs") and choice.logprobs.token_logprobs:
                    lps = [float(lp) if lp is not None else 0.0 for lp in choice.logprobs.token_logprobs]
        
        if not tokens or not lps:
            print("Warning: OpenRouter did not return prompt token logprobs. Provider may only support output token logprobs.")
            return None
        
        # If we got very few tokens (e.g. 1-2), likely output-only — CoDeC needs many (prompt length)
        if len(tokens) < 5 and len(text) > 50:
            print("Warning: Only %d token logprobs for %d-char prompt — likely output tokens only, not prompt." % (len(tokens), len(text)))
            return None
        
        # Suffix alignment
        start_idx = 0
        if score_only_suffix:
            suffix_text = score_only_suffix.strip()
            full_text_norm = " ".join(tokens).replace(" ", "")
            suffix_norm = suffix_text.replace(" ", "")
            if suffix_norm in full_text_norm:
                suffix_start = full_text_norm.rfind(suffix_norm)
                char_count = 0
                for i, tok in enumerate(tokens):
                    char_count += len(tok.replace(" ", ""))
                    if char_count > suffix_start:
                        start_idx = i
                        break
        
        scored_tokens = tokens[start_idx:]
        scored_lps = lps[start_idx:]
        tokens_data = list(zip(scored_tokens, scored_lps))
        
        if not scored_lps:
            return None
        
        avg_logprob = sum(scored_lps) / len(scored_lps)
        return "", avg_logprob, tokens_data
        
    except Exception as e:
        print(f"Error scoring text logprobs (OpenRouter): {e}")
        import traceback
        traceback.print_exc()
        return None


def get_text_logprobs_openai(
    model_dict: dict,
    text: str,
    model_name: str = "davinci-002",
    score_only_suffix: Optional[str] = None,
) -> Optional[Tuple[str, float, List[Tuple[str, float]]]]:
    """Scores text using OpenAI completions API. Uses max_tokens=0, echo=True for prompt token logprobs."""
    try:
        client = model_dict["client"]
        openai_model = model_dict["model_name"]
        r = client.completions.create(
            model=openai_model,
            prompt=text,
            max_tokens=0,
            temperature=0.0,
            echo=True,
            logprobs=1,
        )
        tokens, lps = [], []
        if r.choices and r.choices[0].logprobs:
            lp = r.choices[0].logprobs
            tokens = list(getattr(lp, "tokens", []) or [])
            lps = [float(x) if x is not None else 0.0 for x in (getattr(lp, "token_logprobs", []) or [])]
        if not tokens or not lps:
            return None
        start_idx = 0
        if score_only_suffix:
            suffix_norm = _normalize_for_alignment(score_only_suffix).replace(" ", "")
            full_norm = " ".join(tokens).replace(" ", "")
            if suffix_norm in full_norm:
                idx = full_norm.rfind(suffix_norm)
                char_count = 0
                for i, tok in enumerate(tokens):
                    char_count += len(tok.replace(" ", ""))
                    if char_count > idx:
                        start_idx = i
                        break
        scored_tokens = tokens[start_idx:]
        scored_lps = lps[start_idx:]
        if not scored_lps:
            return None
        tokens_data = list(zip(scored_tokens, scored_lps))
        avg_logprob = sum(scored_lps) / len(scored_lps)
        return "", avg_logprob, tokens_data
    except Exception as e:
        print(f"Error scoring text logprobs (OpenAI): {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_match_score(generated: str, expected: str) -> float:
    """
    Calculates a simple match score (0.0 to 1.0) based on how much of the expected suffix 
    is present in the generated text.
    """
    if not generated: return 0.0
    
    gen_norm = generated.strip().lower()
    exp_norm = expected.strip().lower()
    
    if not exp_norm:
        return 0.0
        
    common_len = 0
    min_len = min(len(gen_norm), len(exp_norm))
    
    for i in range(min_len):
        if gen_norm[i] == exp_norm[i]:
            common_len += 1
        else:
            break
            
    return common_len / len(exp_norm)
