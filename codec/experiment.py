import hashlib
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from datetime import datetime

import utils
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset

DATA_DIR = "paper_reproduction/data"
# SEEN_FILE = os.path.join(DATA_DIR, "romeo_and_juliet.txt") # Replaced by HumanEval


def _load_hf_text_chunks(dataset_name, n_samples=20, config=None, split="test", text_keys=None):
    """Generic loader: load HF dataset and return list of text chunks (first available text column). No fallbacks."""
    def _extract(ds):
        chunks = []
        keys = text_keys or ["question", "problem", "prompt", "text", "content", "body", "article", "instruction", "input"]
        for item in ds:
            text = ""
            for k in keys:
                v = item.get(k)
                if v and isinstance(v, str) and len(v.strip()) > 50:
                    text = v.strip()
                    break
            if not text and isinstance(item, dict):
                for k, v in item.items():
                    if v and isinstance(v, str) and len(v.strip()) > 50:
                        text = v.strip()
                        break
            if text:
                chunks.append(text)
                if len(chunks) >= n_samples:
                    break
        return chunks
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        return _extract(ds)
    except Exception as e:
        print(f"  _load_hf_text_chunks({dataset_name}): {e}")
        return []


def load_livecodebench_chunks(cutoff_date, n_samples=20, version="v5"):
    """Loads LiveCodeBench dataset for unseen data.
    
    Args:
        cutoff_date: Date cutoff for filtering (only items after this date)
        n_samples: Number of samples to load
        version: Version to use - "v5" for latest, "lite" for code_generation_lite
    """
    try:
        from datetime import datetime
        print(f"Loading LiveCodeBench dataset (version: {version})...")
        
        # LiveCodeBench structure: the main dataset has a 'test' split
        # Version v5 refers to problems released up to Jan 2025
        # We'll load from the main dataset and filter by date
        
        dataset = load_dataset("livecodebench/code_generation", split="test")
        print("Using LiveCodeBench code_generation (test split)")
        
        chunks = []
        print(f"Filtering for problems released after {cutoff_date.strftime('%Y-%m-%d')}...")
        
        skipped_count = 0
        for item in dataset:
            # Check date (contest_date can be str or datetime from HF)
            date_val = item.get('contest_date')
            if date_val is None:
                skipped_count += 1
                continue
            try:
                if isinstance(date_val, datetime):
                    item_date = date_val.replace(tzinfo=None) if date_val.tzinfo else date_val
                elif isinstance(date_val, str):
                    s = date_val.split('T')[0] if 'T' in date_val else date_val
                    item_date = datetime.strptime(s, "%Y-%m-%d")
                else:
                    skipped_count += 1
                    continue
                if item_date <= cutoff_date:
                    skipped_count += 1
                    continue
            except (ValueError, TypeError):
                skipped_count += 1
                continue

            # Construct text: Question Content (Prompt)
            # Try different field names based on dataset structure
            prompt = item.get('question_content', '') or item.get('prompt', '') or item.get('problem', '')
            starter = item.get('starter_code', '') or item.get('starter', '')
            solution = item.get('canonical_solution', '') or item.get('solution', '')
            
            if prompt:
                text = prompt
                if starter:
                    text += "\n\n" + starter
                # Include solution if available (for v5, solutions might be available)
                if solution and version == "v5":
                    text += "\n\n" + solution
                
                chunks.append(text)
                
            if len(chunks) >= n_samples:
                break
        
        print(f"Skipped {skipped_count} items due to date filtering.")
        return chunks
    except Exception as e:
        print(f"Error loading LiveCodeBench: {e}")
        return []

def load_humaneval_chunks(n_samples=20):
    """Loads HumanEval dataset and formats as code chunks."""
    try:
        print("Loading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")
        chunks = []
        # Use prompt + canonical_solution as the text to memorize
        for item in dataset:
            text = item['prompt'] + item['canonical_solution']
            chunks.append(text)
            if len(chunks) >= n_samples:
                break
        return chunks
    except Exception as e:
        print(f"Error loading HumanEval: {e}")
        return []

def load_mbpp_chunks(n_samples=20):
    """Loads MBPP dataset (Mostly Basic Python Problems) for seen data."""
    try:
        print("Loading MBPP dataset...")
        dataset = load_dataset("mbpp", split="test")
        chunks = []
        for item in dataset:
            text = item['text'] + "\n" + item['code']
            chunks.append(text)
            if len(chunks) >= n_samples:
                break
        return chunks
    except Exception as e:
        print(f"Error loading MBPP: {e}")
        return []

def load_gsm8k_chunks(n_samples=20):
    """Loads GSM8K dataset (Grade School Math) for seen data."""
    try:
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        chunks = []
        for item in dataset:
            text = item['question'] + "\n" + item['answer']
            chunks.append(text)
            if len(chunks) >= n_samples:
                break
        return chunks
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []


def load_gpqa_diamond_chunks(n_samples=20):
    """Paper Appendix B.2: GPQA Diamond (graduate-level QA)."""
    return _load_hf_text_chunks("reasoningMIA/gpqa_diamond", n_samples=n_samples, split="train", text_keys=["question", "text", "prompt"])


def load_ifeval_chunks(n_samples=20):
    """Paper Appendix B.2: IFEval (instruction-following)."""
    return _load_hf_text_chunks("google/IFEval", n_samples=n_samples, split="train", text_keys=["instruction", "prompt", "input", "text"])


def load_frames_chunks(n_samples=20):
    """Paper Appendix B.2: FRAMES benchmark. No fallbacks."""
    return _load_hf_text_chunks("google/frames-benchmark", n_samples=n_samples, split="train")


def load_aime_2024_chunks(n_samples=20):
    """Paper Appendix B.2: AIME 2024."""
    return _load_hf_text_chunks("HuggingFaceH4/aime_2024", n_samples=n_samples, split="train", text_keys=["problem", "question", "text"])


def load_aime_2025_chunks(n_samples=20):
    """Paper Appendix B.2: AIME 2025 (Nemotron-H uses only this). No fallbacks."""
    return _load_hf_text_chunks("math-ai/aime25", n_samples=n_samples, split="test", text_keys=["problem", "question", "text"])


def load_bfcl_v3_chunks(n_samples=20):
    """Paper Appendix B.2: BFCL v3 (function calling)."""
    return _load_hf_text_chunks("gorilla-llm/Berkeley-Function-Calling-Leaderboard", n_samples=n_samples, split="train", text_keys=["question", "prompt", "input", "text"])


def load_bbq_chunks(n_samples=20):
    """Paper Appendix B.2: BBQ benchmark. No fallbacks."""
    return _load_hf_text_chunks("Elfsong/BBQ", n_samples=n_samples, split="test", text_keys=["context", "question", "text"])


def load_rewardbench_v1_chunks(n_samples=20):
    """Paper Appendix B.2: RewardBench v1."""
    return _load_hf_text_chunks("allenai/reward-bench", n_samples=n_samples, split="train", text_keys=["prompt", "chosen", "rejected", "text"])


def load_rewardbench_v2_chunks(n_samples=20):
    """Paper Appendix B.2: RewardBench v2 (same dataset, different split or subset)."""
    return _load_hf_text_chunks("allenai/reward-bench", n_samples=n_samples, split="test", text_keys=["prompt", "chosen", "rejected", "text"])


def load_math_500_chunks(n_samples=20):
    """Paper Appendix B.2: MATH 500."""
    return _load_hf_text_chunks("HuggingFaceH4/MATH-500", n_samples=min(n_samples, 500), split="test", text_keys=["problem", "question", "text"])


def load_gutenberg_book_chunks(book_key, n_samples=20):
    """Paper Appendix B.2: Project Gutenberg books. Paper: continuous stream split into 600-char chunks."""
    path_map = {
        "jibby_jones": os.path.join(DATA_DIR, "jibby_jones.txt"),
        "colonial_memories": os.path.join(DATA_DIR, "colonial_memories.txt"),
        "corbin_necklace": os.path.join(DATA_DIR, "corbin_necklace.txt"),
    }
    path = path_map.get(book_key)
    if path and os.path.isfile(path):
        return load_and_chunk(path, chunk_size_chars=PAPER_CHUNK_SIZE_CHARS)
    return []


def load_global_news_chunks(n_samples=20):
    """Paper Appendix B.2: NickyNicky/global-news-dataset."""
    return _load_hf_text_chunks("NickyNicky/global-news-dataset", n_samples=n_samples, split="train", text_keys=["title", "description", "content", "text"])


def load_amazon_reviews_2023_chunks(n_samples=20):
    """Paper Appendix B.2: McAuley-Lab/Amazon-Reviews-2023."""
    return _load_hf_text_chunks("McAuley-Lab/Amazon-Reviews-2023", n_samples=n_samples, split="train", text_keys=["review_text", "body", "text"])


def load_ukraine_article_chunks(n_samples=20):
    """Paper Appendix B.2: Ukraine conflict updates article. Paper: continuous stream split into 600-char chunks."""
    try:
        import urllib.request
        import re
        url = "https://www.understandingwar.org/backgrounder/ukraine-conflict-updates"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < PAPER_CHUNK_SIZE_CHARS:
            return []
        chunks = [text[i:i + PAPER_CHUNK_SIZE_CHARS].strip() for i in range(0, len(text), PAPER_CHUNK_SIZE_CHARS)]
        return [c for c in chunks if len(c) >= PAPER_CHUNK_SIZE_CHARS // 2]
    except Exception as e:
        print(f"  load_ukraine_article_chunks: {e}")
        return []


def load_bigobench_chunks(n_samples=20):
    """Loads BigOBench dataset (time complexity) as unseen data."""
    try:
        print("Loading BigOBench dataset...")
        # Note: The configuration is "time_complexity_test_set.jsonl", but Hugging Face
        # loads this single file into the 'train' split by default.
        ds_dict = load_dataset("facebook/BigOBench", "time_complexity_test_set.jsonl")
        
        # Access the split (it's mapped to 'train')
        dataset = ds_dict['train']
        
        chunks = []
        for item in dataset:
            # Using solution_code as the content
            text = item.get('solution_code', '')
            if text:
                 chunks.append(text)
            if len(chunks) >= n_samples:
                break
        return chunks
    except Exception as e:
        print(f"Error loading BigOBench: {e}")
        return []

def load_wikipedia_wikibooks_chunks(n_samples=20):
    """Loads Wikipedia directly from Hugging Face (paper lists Wikipedia only, not Wikibooks)."""
    try:
        print("Loading Wikipedia...")
        # Use wikimedia/wikipedia: the legacy 'wikipedia' dataset uses deprecated loading scripts
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_dataset = wiki_dataset.shuffle(seed=42, buffer_size=10000)
        chunks = []
        for item in wiki_dataset:
            text = item.get('text', '')
            if text and text.strip() and len(text.strip()) > 100:
                chunks.append(text.strip())
                if len(chunks) >= n_samples:
                    break
        print(f"Loaded {len(chunks)} samples from Wikipedia")
        return chunks
    except Exception as e:
        print(f"Error loading Wikipedia: {e}")
        return []


def load_c4_chunks(n_samples=20, split="train"):
    """Loads C4 (Colossal Clean Crawled Corpus) chunks. Used in Dolma and Nemotron-CC."""
    try:
        print("Loading C4 (streaming)...")
        # Use allenai/c4: the legacy 'c4' dataset uses deprecated loading scripts
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
        chunks = []
        for item in dataset:
            text = item.get("text", "") or ""
            if text and text.strip() and len(text.strip()) > 100:
                chunks.append(text.strip())
                if len(chunks) >= n_samples:
                    break
        print(f"Loaded {len(chunks)} samples from C4")
        return chunks
    except Exception as e:
        print(f"Error loading C4: {e}")
        return []


def load_stackexchange_chunks(n_samples=20, split="train"):
    """Loads Stack Exchange–style text from codeparrot/stack-exchange-as-a-dataset. No fallbacks."""
    try:
        print("Loading Stack Exchange data...")
        ds = load_dataset("codeparrot/stack-exchange-as-a-dataset", split=split)
        chunks = []
        for item in ds:
            text = (item.get("body") or item.get("content") or item.get("text") or item.get("Body") or "").strip()
            if isinstance(text, str) and len(text) > 100:
                chunks.append(text)
                if len(chunks) >= n_samples:
                    break
        print(f"Loaded {len(chunks)} samples from Stack Exchange")
        return chunks
    except Exception as e:
        print(f"Error loading Stack Exchange: {e}")
        return []


def load_pes2o_chunks(n_samples=20):
    """Loads peS2o (STEM papers) - Dolma component. Uses allenai/peS2o (no loading script)."""
    try:
        print("Loading peS2o (STEM papers)...")
        ds = load_dataset("allenai/peS2o", "v2", split="train")
        chunks = []
        for item in ds:
            text = item.get("text", "") or ""
            if text and text.strip() and len(text.strip()) > 100:
                chunks.append(text.strip())
                if len(chunks) >= n_samples:
                    break
        print(f"Loaded {len(chunks)} samples from peS2o")
        return chunks
    except Exception as e:
        print(f"Error loading peS2o: {e}")
        return []


def load_reddit_chunks(n_samples=20):
    """Loads Reddit-style text. Dolma component. Uses tiiuae/falcon-refinedweb (web content, no deprecated scripts)."""
    try:
        print("Loading Reddit-style content (Falcon RefinedWeb)...")
        # RefinedWeb: content field; use streaming for large dataset
        ds = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=10000)
        chunks = []
        for item in ds:
            text = item.get("content", "") or item.get("text", "") or ""
            if text and text.strip() and len(text.strip()) > 100:
                chunks.append(text.strip())
                if len(chunks) >= n_samples:
                    break
        print(f"Loaded {len(chunks)} samples from RefinedWeb (Reddit proxy)")
        return chunks
    except Exception as e:
        print(f"Error loading Reddit proxy: {e}")
        return []


def load_dolma_chunks(n_samples=20, source=None, split="train", streaming=True):
    """Loads Dolma-like chunks. allenai/dolma uses deprecated loading scripts; we use component datasets.
    
    Args:
        n_samples: Number of samples to load
        source: 'pes2o' | 'reddit' | 'commoncrawl' | 'wikipedia' | 'c4' | None (C4 as default)
        split, streaming: Ignored for component loaders.
    
    Returns:
        List of text chunks.
    """
    source = (source or "").lower()
    if source in ("pes2o", "arxiv", "scientific", "papers"):
        return load_pes2o_chunks(n_samples=n_samples)
    if source in ("reddit",):
        return load_reddit_chunks(n_samples=n_samples)
    if source in ("commoncrawl", "common crawl", "cc", "c4") or not source:
        return load_c4_chunks(n_samples=n_samples)
    if source in ("wikipedia", "wiki"):
        return load_wikipedia_wikibooks_chunks(n_samples=n_samples)
    # Default: C4
    return load_c4_chunks(n_samples=n_samples)


# ---------------------------------------------------------------------------
# Paper Appendix B: Training (seen) and unseen datasets
# ---------------------------------------------------------------------------
# The Pile (paper: HackerNews, Wikipedia, GitHub, ArXiv, DM Mathematics, Pile CommonCrawl,
#   PubMed Central, Full Pile; + low-diversity: Wikipedia Music, GitHub Licenses).
#   Pile not available for direct download; we use ArmelR/the-pile-splitted (or iamgroot42/mimir samples).
# Dolma (paper): C4, Wikipedia, Pes2o v2, Reddit v5, CommonCrawl head/middle/tail.
# Nemotron-CC (paper): Wikipedia, ArXiv, CommonCrawl (multiple quality buckets).
# Unseen (paper): Pile-trained → gsm8k, GPQA Diamond, IFEval, HumanEval, FRAMES, AIME 2024/2025,
#   LiveCodeBench v1/v5, BFCL v3, BBQ, RewardBench v1/v2, MATH 500; OLMo → subset of above;
#   Nemotron-H → AIME 2025; + Project Gutenberg (3 books), HF (global-news, Amazon-Reviews-2023), website.
# We implement all paper-listed datasets below with subsampling (EXPERIMENT0_DATASET_SAMPLES per source).
# ---------------------------------------------------------------------------

# ArmelR/the-pile-splitted: Parquet format. Configs: ArXiv, Github, HackerNews, DM Mathematics, Pile-CC, PubMed Central, Wikipedia (en), etc.
# MIMIR (iamgroot42/mimir) uses deprecated loading scripts - use ArmelR instead
PILE_CONFIG_TO_ARMELR = {
    "hackernews": "HackerNews",
    "wikipedia": "Wikipedia (en)",
    "github": "Github",
    "arxiv": "ArXiv",
    "dm_mathematics": "DM Mathematics",
    "pile_cc": "Pile-CC",
    "pubmed_central": "PubMed Central",
    #"full_pile": "default",  # full dataset when subset = "default"
}

# Subsampling: max rows to consume per Pile subset (streaming = no full download)
MAX_PILE_ROWS_PER_SUBSET = 500

# Experiment 0: all paper datasets (Pile seen + unseen); we only subsample per dataset.
# Paper: "For each dataset, we selected 1 000 random samples for evaluation."
# Continuous text (book/article): "we split it into 600-characters chunks and treated as a dataset."
EXPERIMENT0_DATASET_SAMPLES = 1000  # paper full run: 1000; quick test: 5–20
PAPER_CHUNK_SIZE_CHARS = 600  # paper: continuous stream split into 600-char chunks
# Paper: "we selected 1 000 random samples" — load extra then random.sample to 1000
EXPERIMENT0_RANDOM_SAMPLE_POOL = 5000  # load up to this many, then random sample to EXPERIMENT0_DATASET_SAMPLES
# For flaky API/model responses (timeouts, context errors), evaluate more candidates
# and stop once n_samples successful scores are collected.
EXPERIMENT0_CANDIDATE_MULTIPLIER = 10

def load_pile_chunks(config_name, n_samples=20, split="train", max_rows=None):
    """Loads a subsample of text chunks from ArmelR/the-pile-splitted. No fallbacks."""
    subset_name = PILE_CONFIG_TO_ARMELR.get(config_name, config_name)
    max_rows = max_rows if max_rows is not None else MAX_PILE_ROWS_PER_SUBSET

    # 1) Try streaming: only downloads data as we iterate; we stop after enough chunks
    max_rows = max(max_rows, n_samples)  # request at least n_samples rows when streaming
    try:
        print(f"Loading Pile subset: {subset_name} (streaming, max {max_rows} rows)...")
        dataset = load_dataset(
            "ArmelR/the-pile-splitted",
            subset_name,
            split=split,
            streaming=True,
        )
        # Do NOT shuffle: streaming default order is deterministic; shuffle can vary shard order across runs
        chunks = []
        seen = 0
        for item in dataset:
            seen += 1
            if seen > max_rows:
                break
            text = item.get("text", "") or item.get("content", "") or ""
            if text and text.strip() and len(text.strip()) > 100:
                chunks.append(text.strip())
                if len(chunks) >= n_samples:
                    break
        if chunks:
            print(f"Loaded {len(chunks)} samples from Pile ({subset_name}, streaming)")
            return chunks
    except Exception as e:
        print(f"Error loading Pile subset {subset_name}: {e}")
    return []


def load_and_chunk(filepath, chunk_size_words=100, chunk_size_chars=None):
    """Loads text file and splits into chunks.
    
    Paper: for continuous stream (e.g. book or article), "we split it into 600-characters
    chunks and treated as a dataset." Use chunk_size_chars=600 for that. Otherwise chunks by words.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            return []
        if chunk_size_chars is not None:
            # Paper: 600-character chunks for continuous text
            chunks = [text[i:i + chunk_size_chars].strip() for i in range(0, len(text), chunk_size_chars)]
            return [c for c in chunks if len(c) >= chunk_size_chars // 2]  # drop very short tail
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size_words):
            chunk = words[i:i + chunk_size_words]
            if len(chunk) == chunk_size_words:
                chunks.append(" ".join(chunk))
        return chunks
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def _run_experiment_parallel(dataset_name, chunks, model, model_name, n_samples, n_context, ignore_prefix_tokens, n_seeds, random_seed, parallel_workers):
    """Parallel vLLM scoring: baseline and ICL calls via ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    random.seed(random_seed)

    def _extract_token_logprobs(token_data):
        if not token_data:
            return [], []
        return [t[0] for t in token_data], [t[1] for t in token_data]

    candidate_chunks = [c.strip() for c in chunks if c and c.strip()]
    if len(candidate_chunks) > 1:
        rng = random.Random(random_seed)
        rng.shuffle(candidate_chunks)
    # Try more than n_samples candidates, but stop as soon as n_samples successes are collected.
    max_candidates = min(len(candidate_chunks), max(n_samples, n_samples * EXPERIMENT0_CANDIDATE_MULTIPLIER))
    candidate_chunks = candidate_chunks[:max_candidates]

    def do_baseline(item):
        i, target_text, _ = item
        if not target_text:
            return (i, None)
        res = utils.get_text_logprobs(model, target_text, model_name=model_name)
        return (i, res)

    def do_icl(args):
        i, target_text, context_pool, seed_idx = args
        if not target_text or not context_pool:
            return (i, seed_idx, None)
        random.seed(random_seed + i * 1000 + seed_idx)
        k = min(n_context, len(context_pool))
        context_samples = random.sample(context_pool, k) if k > 0 else []
        context_str = "\n\n".join([cs.strip() for cs in context_samples])
        full_text = (context_str + "\n\n" + target_text) if context_str else target_text
        res = utils.get_text_logprobs(model, full_text, model_name=model_name, score_only_suffix=target_text)
        return (i, seed_idx, res)

    results = []
    # Batch processing prevents scheduling expensive calls for all candidates up-front.
    # We run only what is needed to reach n_samples successful items.
    batch_size = max(16, min(64, n_samples))
    offset = 0
    while len(results) < n_samples and offset < len(candidate_chunks):
        batch = candidate_chunks[offset:offset + batch_size]
        offset += batch_size

        context_pools = []
        for i, chunk in enumerate(batch):
            pool = [c for c in candidate_chunks if c != chunk and c and c.strip()]
            context_pools.append((i, chunk, pool))

        baseline_results = {}
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(do_baseline, (i, t, p)): i for i, t, p in context_pools}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    i, res = fut.result()
                    baseline_results[i] = res
                except Exception:
                    baseline_results[idx] = None

        icl_tasks = []
        for i, target_text, context_pool in context_pools:
            for seed_idx in range(max(1, n_seeds)):
                icl_tasks.append((i, target_text, context_pool, seed_idx))

        icl_results = {}
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(do_icl, t): t for t in icl_tasks}
            for fut in as_completed(futures):
                try:
                    i, seed_idx, res = fut.result()
                    icl_results[(i, seed_idx)] = res
                except Exception:
                    pass

        for i, target_text, context_pool in context_pools:
            if len(results) >= n_samples:
                break
            base_res = baseline_results.get(i)
            if not base_res:
                continue
            _, base_avg_logprob, base_tokens_data = base_res
            base_tokens, base_lps = _extract_token_logprobs(base_tokens_data)
            if ignore_prefix_tokens > 0 and len(base_lps) > ignore_prefix_tokens:
                base_lps_scored = base_lps[ignore_prefix_tokens:]
            else:
                base_lps_scored = base_lps
            if not base_lps_scored:
                continue
            base_mean = sum(base_lps_scored) / len(base_lps_scored)

            deltas = []
            icl_means = []
            for seed_idx in range(max(1, n_seeds)):
                icl_res = icl_results.get((i, seed_idx))
                if not icl_res:
                    continue
                _, _, icl_tokens_data = icl_res
                icl_tokens, icl_lps = _extract_token_logprobs(icl_tokens_data)
                if ignore_prefix_tokens > 0 and len(icl_lps) > ignore_prefix_tokens:
                    icl_lps_scored = icl_lps[ignore_prefix_tokens:]
                else:
                    icl_lps_scored = icl_lps
                if not icl_lps_scored:
                    continue
                icl_mean = sum(icl_lps_scored) / len(icl_lps_scored)
                deltas.append(icl_mean - base_mean)
                icl_means.append(icl_mean)

            if not deltas:
                continue
            results.append({
                "delta_logprob": sum(deltas) / len(deltas),
                "base_mean_logprob": base_mean,
                "icl_mean_logprob": sum(icl_means) / len(icl_means),
                "n_context": min(n_context, len(context_pool)),
                "n_seeds": len(deltas),
            })

    if len(results) < n_samples:
        print(
            f"  Warning: collected {len(results)}/{n_samples} successful samples "
            f"for {dataset_name} (parallel mode)."
        )

    return results


def run_experiment(dataset_name, chunks, model, model_name, n_samples=1000, n_context=3, ignore_prefix_tokens=0, n_seeds=1, random_seed=42, parallel_workers=0):
    """Runs a paper-faithful CoDeC-style experiment.

    CoDeC measures the change in average token log-likelihood of a FIXED target text x
    when prepending n_context in-context samples drawn from the same dataset.

    For each chunk x:
      - baseline: log p(x)
      - in-context: log p(x | x1..xn)
      - delta = in_context - baseline

    Paper (arxiv:2510.27055) alignment:
      - n_context=3: larger context for stronger signal (paper Section 3.4 shows more context improves separation)
      - ignore_prefix_tokens=0: "average log-likelihood on the consecutive tokens of x" (Section 2.3)
      - n_samples=1000: "around 100 examples already give stable estimates" (Section 3.4)

    When parallel_workers > 0 and model is vLLM or Gemini (API), runs baseline and in-context scoring in parallel.

    Returns a list of dicts with delta_logprob and the raw baseline/in-context averages.
    """
    is_vllm = isinstance(model, dict) and model.get("model_type") == "vllm"
    is_gemini = model_name and "gemini" in model_name.lower()
    use_parallel = parallel_workers > 0 and (is_vllm or is_gemini)

    if use_parallel:
        print(f"\n--- Running CoDeC Experiment for: {dataset_name} (Model: {model_name}, parallel workers={parallel_workers}) ---")
        return _run_experiment_parallel(dataset_name, chunks, model, model_name, n_samples, n_context, ignore_prefix_tokens, n_seeds, random_seed, parallel_workers)

    print(f"\n--- Running CoDeC Experiment for: {dataset_name} (Model: {model_name}) ---")
    random.seed(random_seed)

    def _extract_token_logprobs(token_data):
        if not token_data:
            return [], []
        toks = [t[0] for t in token_data]
        lps = [t[1] for t in token_data]
        return toks, lps

    results = []
    candidate_chunks = [c.strip() for c in chunks if c and c.strip()]
    if len(candidate_chunks) > 1:
        rng = random.Random(random_seed)
        rng.shuffle(candidate_chunks)
    max_candidates = min(len(candidate_chunks), max(n_samples, n_samples * EXPERIMENT0_CANDIDATE_MULTIPLIER))
    test_chunks = candidate_chunks[:max_candidates]

    for i, chunk in enumerate(test_chunks):
        if len(results) >= n_samples:
            break
        target_text = chunk.strip()
        if not target_text:
            continue

        # 1) Baseline: score the fixed target text
        print(f"Sample {i+1}: Baseline (no context)...")
        base_res = utils.get_text_logprobs(model, target_text, model_name=model_name)
        if not base_res:
            print("  Baseline scoring failed; skipping.")
            continue

        _, base_avg_logprob, base_tokens_data = base_res
        base_tokens, base_lps = _extract_token_logprobs(base_tokens_data)

        # Apply token ignore window (on target tokens)
        if ignore_prefix_tokens > 0 and len(base_lps) > ignore_prefix_tokens:
            base_lps_scored = base_lps[ignore_prefix_tokens:]
        else:
            base_lps_scored = base_lps

        if not base_lps_scored:
            print("  Not enough tokens after ignore window; skipping.")
            continue

        # In case utils.get_text_logprobs returns an average already, we still recompute to be safe
        base_mean = sum(base_lps_scored) / len(base_lps_scored)

        # 2) In-context: sample n_context other chunks and prepend, score ONLY the target text tokens
        # Repeat for multiple seeds if requested and average deltas
        deltas = []
        icl_means = []

        context_pool = [c for c in test_chunks if c != chunk and c and c.strip()]
        if not context_pool:
            print("  No context pool available; skipping.")
            continue

        for seed_idx in range(max(1, n_seeds)):
            # Use deterministic seed: global_seed + sample_index + seed_index
            # This ensures reproducibility while allowing multiple seeds per sample
            sample_seed = random_seed + i * 1000 + seed_idx
            random.seed(sample_seed)
            k = min(n_context, len(context_pool))
            context_samples = random.sample(context_pool, k) if k > 0 else []
            context_str = "\n\n".join([cs.strip() for cs in context_samples])

            # Build full input: context + separator + target
            if context_str:
                full_text = context_str + "\n\n" + target_text
            else:
                full_text = target_text

            print(f"Sample {i+1}: In-Context (k={k}, seed_idx={seed_idx})...")
            icl_res = utils.get_text_logprobs(model, full_text, model_name=model_name, score_only_suffix=target_text)
            if not icl_res:
                print("  ICL scoring failed for this seed; skipping seed.")
                continue

            _, _, icl_tokens_data = icl_res
            icl_tokens, icl_lps = _extract_token_logprobs(icl_tokens_data)

            # Ignore first tokens in the TARGET portion only
            if ignore_prefix_tokens > 0 and len(icl_lps) > ignore_prefix_tokens:
                icl_lps_scored = icl_lps[ignore_prefix_tokens:]
            else:
                icl_lps_scored = icl_lps

            if not icl_lps_scored:
                print("  Not enough ICL tokens after ignore window; skipping seed.")
                continue

            icl_mean = sum(icl_lps_scored) / len(icl_lps_scored)
            delta = icl_mean - base_mean
            deltas.append(delta)
            icl_means.append(icl_mean)
            print(f"  Baseline mean LP: {base_mean:.4f} | ICL mean LP: {icl_mean:.4f} | Delta: {delta:.4f}")

        if not deltas:
            print("  No successful ICL seeds; skipping sample.")
            continue

        delta_logprob = sum(deltas) / len(deltas)
        icl_mean_avg = sum(icl_means) / len(icl_means)

        results.append({
            "delta_logprob": delta_logprob,
            "base_mean_logprob": base_mean,
            "icl_mean_logprob": icl_mean_avg,
            "n_context": min(n_context, len(context_pool)),
            "n_seeds": len(deltas)
        })

    if len(results) < n_samples:
        print(
            f"  Warning: collected {len(results)}/{n_samples} successful samples "
            f"for {dataset_name}."
        )

    return results


def calculate_contamination_score(results):
    """
    Computes the CoDeC Contamination Score:
    S_CoDeC(D) = (1 / N) * Sum( 1[Delta(x_i) < 0] )

    where Delta(x_i) is the change in mean token log-likelihood of the FIXED target text x_i
    when adding in-context samples from the same dataset.

    returns: score (float between 0 and 1)
    """
    if not results:
        return 0.0
    count_negative = sum(1 for r in results if r['delta_logprob'] < 0)
    return count_negative / len(results)

def plot_contamination_score(seen_score, seen_name, unseen_score, unseen_name, output_file="contamination_score.png"):
    """
    Plots the Contamination Score (S_CoDeC) for Seen vs Unseen datasets.
    """
    plt.figure(figsize=(8, 6))

    datasets = [seen_name, unseen_name]
    scores = [seen_score * 100, unseen_score * 100] # Convert to percentage
    colors = ['#d62728', '#1f77b4'] # Red for Seen (High Contamination), Blue for Unseen (Low)

    ax = plt.gca()
    bars = ax.bar(datasets, scores, color=colors)

    # Add value labels
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')

    plt.title("CoDeC Contamination Score (Lower is Better/Unseen)", fontsize=14)
    plt.ylabel("Contamination Score (%) \n(% samples where Context WORSENED performance)", fontsize=12)
    plt.ylim(0, 105) # 0 to 100%

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nContamination Score Plot saved to {output_file}")


def plot_results(seen_results, unseen_results, output_file="results_plot.png"):
    """Generates and saves a boxplot comparing Delta LogProb."""
    data = []

    for r in seen_results:
        data.append({"Dataset": "Seen Data", "Delta LogProb": r["delta_logprob"]})
    for r in unseen_results:
        data.append({"Dataset": "Unseen Data", "Delta LogProb": r["delta_logprob"]})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    seen_vals = df[df["Dataset"] == "Seen Data"]["Delta LogProb"].tolist()
    unseen_vals = df[df["Dataset"] == "Unseen Data"]["Delta LogProb"].tolist()
    ax.boxplot([seen_vals, unseen_vals], labels=["Seen Data", "Unseen Data"], showfliers=True)

    # overlay points
    for j, vals in enumerate([seen_vals, unseen_vals], start=1):
        xs = [j + (random.random() - 0.5) * 0.1 for _ in vals]
        ax.scatter(xs, vals, alpha=0.6)

    plt.title("Impact of Context on Model Confidence (Delta LogProb)", fontsize=16)
    plt.ylabel("Delta LogProb (ICL - Baseline)", fontsize=12)
    plt.xlabel("", fontsize=12)

    # Add zero line
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label="No Change")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nPlot saved to {output_file}")


def plot_token_diff(tokens, base_logprobs, icl_logprobs, title, output_file):
    """
    Plots token-level logprob differences similar to the paper's figure.
    """
    import numpy as np
    
    # Debug: Check input lengths
    print(f"    Plot function: {len(tokens)} tokens, {len(base_logprobs)} base logprobs, {len(icl_logprobs)} icl logprobs")
    
    # Align lengths - simplify by truncating to min length
    min_len = min(len(base_logprobs), len(icl_logprobs), len(tokens))
    if min_len == 0:
        print(f"    Skipping plot {output_file}: Not enough tokens (min_len={min_len}).")
        return
    
    if min_len < 5:
        print(f"    Warning: Very few tokens ({min_len}) for plotting. This may indicate a token extraction issue.")
        
    # Limit to max 60 tokens for readability and consistency
    max_tokens = 60
    if min_len > max_tokens:
        min_len = max_tokens
        print(f"    Limiting to {max_tokens} tokens for readability")

    # Use the tokens from the ICL generation (assuming it's the 'better' one or just using one for X-axis labels)
    # Ideally they are the same tokens if the model generates the same text.
    plot_tokens = tokens[:min_len]
    base_lp = np.array(base_logprobs[:min_len])
    icl_lp = np.array(icl_logprobs[:min_len])
    
    diff = icl_lp - base_lp
    cumulative_diff = np.cumsum(diff)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # X-axis
    x = np.arange(len(plot_tokens))
    
    # 1. Grey bars: Logprob without context (Base)
    # The paper plots 'Logprob without context' as grey bars.
    # These are typically small (close to 0) and represent the baseline logprob
    ax1.bar(x, base_lp, color='lightgrey', label='Logprob without context', alpha=0.8, width=0.8)
    
    # 2. Red/Green bars: Logprob difference with context
    # These bars show the change when context is added
    # Green if diff > 0 (ICL improved confidence), Red if diff < 0 (ICL worsened confidence)
    # Stacked on top of the gray bars (or extending below if negative)
    for i in range(len(x)):
        if diff[i] >= 0:
            # Positive difference: green bar extending upward from base
            ax1.bar(x[i], diff[i], bottom=base_lp[i], color='green', alpha=0.7, width=0.8)
        else:
            # Negative difference: red bar extending downward from base
            ax1.bar(x[i], diff[i], bottom=base_lp[i], color='red', alpha=0.7, width=0.8)
            
    # Proxy artists for legend
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple
    
    green_patch = mpatches.Patch(color='green', label='Logprob difference with context')
    red_patch = mpatches.Patch(color='red', label='Logprob difference with context')
            
    ax1.set_ylabel('Log Probability')
    ax1.set_xlabel('Token')
    
    # Rotate x labels - match paper style
    ax1.set_xticks(x)
    # Truncate long tokens for display
    display_tokens = [t[:12] if len(t) > 12 else t for t in plot_tokens]
    # Adjust font size and rotation based on number of tokens
    if len(plot_tokens) > 25:
        fontsize = 7
        rotation = 90
    elif len(plot_tokens) > 15:
        fontsize = 8
        rotation = 75
    elif len(plot_tokens) > 10:
        fontsize = 9
        rotation = 60
    else:
        fontsize = 10
        rotation = 45
    ax1.set_xticklabels(display_tokens, rotation=rotation, ha='right', fontsize=fontsize)
    
    # Set reasonable y-axis limits for log probability
    y_min = min(np.min(base_lp), np.min(icl_lp)) - 0.1
    y_max = max(np.max(base_lp), np.max(icl_lp)) + 0.1
    ax1.set_ylim(y_min, y_max)
    
    # Right Axis: Cumulative difference
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_diff, color='tab:blue', linewidth=2.5, label='Cumulative difference', marker='o', markersize=3)
    ax2.set_ylabel('Cumulative difference', color='tab:blue', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Set reasonable y-axis limits for cumulative difference
    cum_min = np.min(cumulative_diff) - 0.1
    cum_max = np.max(cumulative_diff) + 0.1
    ax2.set_ylim(cum_min, cum_max)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Combine specifically to match paper:
    # 1. Logprob without context (Grey)
    # 2. Logprob difference with context (Green/Red Tuple)
    # 3. Cumulative difference (Blue Line)
    
    # 'lines' contains the Grey bars (first call to bar). 
    # We construct the handles list manually to ensure order.
    
    grey_bar_handle = lines[0] # The first bar chart
    blue_line_handle = lines2[0] # The line plot
    
    ax1.legend(
        [grey_bar_handle, (green_patch, red_patch), blue_line_handle],
        ['Logprob without context', 'Logprob difference with context', 'Cumulative difference'],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='best',
        fontsize=10,
        framealpha=0.9
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Generated {output_file} with {len(plot_tokens)} tokens")


def run_token_diff_experiment(dataset_name, chunks, model, output_file_prefix, model_name, n_plots=3, random_seed=42):
    """Runs experiments on multiple samples to capture token-level dynamics."""
    print(f"Running Token Analysis for {dataset_name} (Model: {model_name}, Generating {n_plots} plots)...")
    
    # Set seed for reproducibility
    random.seed(random_seed)

    for i in range(min(n_plots, len(chunks))):
        chunk = chunks[i]
        if not chunk:
            continue

        print(f"  Generating plot {i+1}/{n_plots}...")

        target_text = chunk.strip()
        if not target_text:
            print(f"  Skipping chunk {i} (empty)")
            continue
        if len(target_text.split()) < 20:
            print(f"  Warning: Chunk {i} is short ({len(target_text.split())} words), but proceeding anyway")

        # Context
        context_pool = [c for c in chunks if c != chunk and c and c.strip()]
        if not context_pool:
            print("  Not enough chunks for context.")
            continue

        # Use deterministic seed for each plot
        random.seed(random_seed + i * 1000)
        context_samples = random.sample(context_pool, min(3, len(context_pool)))
        context_str = "\n\n".join([cs.strip() for cs in context_samples])
        if context_str:
            full_text = context_str + "\n\n" + target_text
        else:
            full_text = target_text

        # Run Base
        base_res = utils.get_text_logprobs(model, target_text, model_name=model_name)
        if not base_res:
            print(f"  Warning: Baseline scoring failed for sample {i+1}")
            continue
        _, _, base_tokens_data = base_res
        tokens_base = [t[0] for t in base_tokens_data]
        tokens_base_lp = [t[1] for t in base_tokens_data]
        print(f"  Baseline: {len(tokens_base)} tokens")

        # Run ICL
        icl_res = utils.get_text_logprobs(model, full_text, model_name=model_name, score_only_suffix=target_text)
        if not icl_res:
            print(f"  Warning: ICL scoring failed for sample {i+1}")
            continue
        _, _, icl_tokens_data = icl_res
        tokens_icl = [t[0] for t in icl_tokens_data]
        tokens_icl_lp = [t[1] for t in icl_tokens_data]
        print(f"  ICL: {len(tokens_icl)} tokens")

        # Use tokens from the scored suffix (should match target_text)
        # Apply ignore window (first 10 tokens, but adjust if we have very few tokens)
        ignore_prefix_tokens = 10
        
        # Adjust ignore window if we have very few tokens
        if len(tokens_base) < 20 or len(tokens_icl) < 20:
            ignore_prefix_tokens = min(3, len(tokens_base) // 4, len(tokens_icl) // 4)
            print(f"  Adjusted ignore window to {ignore_prefix_tokens} tokens (had {len(tokens_base)} base, {len(tokens_icl)} icl tokens)")
        
        # Extract tokens and logprobs after ignore window
        # For baseline: tokens and logprobs should align
        if len(tokens_base) > ignore_prefix_tokens:
            tokens_base_plot = tokens_base[ignore_prefix_tokens:]
            tokens_base_lp_vals = tokens_base_lp[ignore_prefix_tokens:]
        else:
            tokens_base_plot = tokens_base
            tokens_base_lp_vals = tokens_base_lp
        
        # For ICL: tokens and logprobs should align (these are from the scored suffix)
        if len(tokens_icl) > ignore_prefix_tokens:
            tokens_icl_plot = tokens_icl[ignore_prefix_tokens:]
            tokens_icl_lp_plot = tokens_icl_lp[ignore_prefix_tokens:]
        else:
            tokens_icl_plot = tokens_icl
            tokens_icl_lp_plot = tokens_icl_lp
        
        print(f"  After ignore window: {len(tokens_base_plot)} base tokens, {len(tokens_icl_plot)} icl tokens")
        
        # Use ICL tokens for plotting (they represent the actual scored text)
        # But we need to ensure base and ICL are aligned
        # Since both should represent the same target_text, they should be similar
        # Use the shorter length to ensure alignment
        min_len = min(len(tokens_base_plot), len(tokens_icl_plot))
        if min_len == 0:
            print(f"  Warning: No tokens after ignore window, skipping plot")
            continue
        
        if min_len < 5:
            print(f"  Warning: Very few tokens ({min_len}) after processing. This may indicate an issue with token extraction.")
        
        tokens_text = tokens_icl_plot[:min_len]
        tokens_base_lp_vals = tokens_base_lp_vals[:min_len]
        tokens_icl_lp_plot = tokens_icl_lp_plot[:min_len]

        # For box/bar plotting, we want the logprobs
        plot_filename = f"{output_file_prefix}_{i+1}.png"
        print(f"  Plotting {len(tokens_text)} tokens...")
        plot_token_diff(tokens_text, tokens_base_lp_vals, tokens_icl_lp_plot, f"Changes in confidence: {dataset_name} (Sample {i+1})", plot_filename)


def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Run Contamination Experiment")
    parser.add_argument("--seen", type=str, default=None, 
                        choices=["romeo", "humaneval", "alice", "mbpp", "gsm8k", "dolma", "dolma-wiki"], 
                        help="Dataset to use as 'Seen' (known to model). 'dolma' uses Wikipedia (part of Dolma).")
    parser.add_argument("--unseen", type=str, default=None, 
                        choices=["jibby", "livecodebench", "ladn", "bigobench"], 
                        help="Dataset to use as 'Unseen' (unknown to model). Auto-selected for OLMo models (livecodebench).")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use. For --experiment 0 (Pile), defaults to pythia-410m. Otherwise defaults to gemini-2.5-flash. "
                             "Options: gemini-2.5-flash, gemini-2.0-flash, pythia-410m, pythia-1b, olmo-7b, vllm:pythia-410m, etc.")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vLLM server (default: http://localhost:8000/v1)")
    parser.add_argument("--benchmark", action="store_true", help="Run aggregate benchmark on all datasets")
    parser.add_argument("--experiment", type=int, default=None, choices=[0],
                        help="Experiment 0: Run on Pile subsets (HackerNews, GitHub, ArXiv) + unseen (GSM8K, HumanEval, MBPP)")
    parser.add_argument("--all-models", action="store_true",
                        help="With --experiment 0: run all models, then output AUC table and contamination score figure")
    parser.add_argument("--list-models-datasets", action="store_true",
                        help="Print list of models and their SEEN/UNSEEN datasets (for verification), then exit")
    parser.add_argument("--parallel", type=int, default=4,
                        help="When > 0 and using vLLM, run logprob scoring in parallel with N workers (default: 4)")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Use Hugging Face (transformers) instead of vLLM for Experiment 0. Useful when vLLM server is not running.")
    parser.add_argument("--spawn-vllm", action="store_true",
                        help="With --experiment 0 --all-models: automatically start vLLM server per model, run experiment, then stop server.")
    parser.add_argument("--vllm-screen", action="store_true",
                        help="With --experiment 0 --all-models: run vLLM in GNU screen (screen -r vllmserver). Resume screen, run server, detach, run experiment.")
    parser.add_argument("--vllm-screen-name", type=str, default="vllmserver",
                        help="Screen session name for --vllm-screen (default: vllmserver)")
    parser.add_argument("--vllm-tensor-parallel", type=int, default=2,
                        help="vLLM --tensor-parallel-size when using --spawn-vllm (default: 2)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.5,
                        help="vLLM --gpu-memory-utilization when using --spawn-vllm (default: 0.5)")
    parser.add_argument("--vllm-max-model-len", type=int, default=4000,
                        help="vLLM --max-model-len when using --spawn-vllm (default: 4000)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save per-dataset results (deltas, scores) to results/<model>/ for later comparison")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory for saved results (default: results)")
    args = parser.parse_args()

    # List models and datasets (for verification) then exit
    if args.list_models_datasets:
        print_models_and_datasets()
        return

    # Default model: Pythia for experiment 0 (Pile), Gemini otherwise
    # Note: gemini-2.0-flash is deprecated (404); use gemini-2.5-flash
    if args.model is None:
        args.model = "pythia-410m" if args.experiment == 0 else "gemini-2.5-flash"
        print(f"Using default model for {'Experiment 0 (Pile)' if args.experiment == 0 else 'standard experiment'}: {args.model}")

    # Map model names to actual model identifiers (paper-aligned: Pile, Dolma, Nemotron-CC)
    model_map = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        # Pile-trained (Peng et al., 2023b): Pythia, GPT-Neo, RWKV-4
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia-1.4b": "EleutherAI/pythia-1.4b",
        "pythia-12b": "EleutherAI/pythia-12b",
        "pythia-1b": "EleutherAI/pythia-1b",
        "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
        #"gpt-neo-20b": "EleutherAI/gpt-neox-20b",
        "rwkv-4-430m": "BlinkDL/rwkv-4-pile-430m",
        "rwkv-4-3b": "BlinkDL/rwkv-4-pile-7b",  # paper "3B"; HF has 1.5B/7B/14B, use 7B as proxy
        "rwkv-4-14b": "BlinkDL/rwkv-4-pile-14b",
        # Dolma-trained (Soldaini et al., 2024): OLMo
        "olmo-1b": "allenai/OLMo-1B",
        "allenai/OLMo-1B": "allenai/OLMo-1B",
        "olmo-7b": "allenai/OLMo-7B",
        "olmo-7b-instruct": "allenai/OLMo-7B-Instruct",
        "olmo-13b": "allenai/OLMo-13B",
        # Nemotron-CC (NVIDIA et al., 2025b): Nemotron-v2, Nemotron-H
        "nemotron-v2-9b": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base",
        "nemotron-h-8b": "nvidia/Nemotron-H-8B-Base-8K",
        #"nemotron-h-56b": "nvidia/Nemotron-H-56B-Base-8K",
        # vLLM models - the actual model name should match what's loaded in vLLM server
        "vllm:olmo-1b": "vllm:olmo-1b",
        "vllm:olmo-7b": "vllm:olmo-7b",
        "vllm:olmo-7b-instruct": "vllm:olmo-7b-instruct",
        "vllm:olmo-13b": "vllm:olmo-13b",
        "vllm:pythia-410m": "vllm:pythia-410m",
        "vllm:nemotron-h-8b": "vllm:nemotron-h-8b",
        "vllm:nemotron-v2-9b": "vllm:nemotron-v2-9b",
    }
    
    actual_model_name = model_map.get(args.model, args.model)

    # Nemotron-H uses Mamba architecture; mamba-ssm requires CUDA (Linux+GPU). Use vLLM instead.
    if args.no_vllm and ("nemotron-h" in args.model.lower() or "nemotron-h" in actual_model_name.lower()):
        print("\n*** Nemotron-H requires Mamba (mamba-ssm), which needs CUDA. Use vLLM instead: ***")
        print("  1. Start vLLM (on a machine with NVIDIA GPU):")
        print("     vllm serve nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --mamba_ssm_cache_dtype float32 --served-model-name nemotron-h-8b")
        print("  2. Run: python experiment.py --experiment 0 --model vllm:nemotron-h-8b --parallel 5")
        print("     (Remove --no-vllm to use vLLM)\n")
        return

    # Require auth for Gemini models: GOOGLE_API_KEY (2.0) or GOOGLE_CLOUD_PROJECT (2.5 Vertex)
    if args.model.startswith("gemini"):
        needs_vertex = "gemini-2.5" in args.model.lower()
        has_auth = (
            utils.GOOGLE_CLOUD_PROJECT if needs_vertex
            else utils.API_KEY != "YOUR_API_KEY_HERE"
        )
        if not has_auth:
            if needs_vertex:
                print("CRITICAL ERROR: Gemini 2.5 requires Vertex AI. Set GOOGLE_CLOUD_PROJECT and run: gcloud auth application-default login")
            else:
                print("CRITICAL ERROR: API Key not set. Please set GOOGLE_API_KEY env var for Gemini 2.0.")
            return

    # Experiment 0 all-models: run multiple models, produce AUC table + contamination figure (no single model load)
    if args.experiment == 0 and args.all_models:
        run_experiment_five_all_models(
            model_map,
            vllm_url=args.vllm_url,
            parallel_workers=args.parallel,
            use_vllm=not args.no_vllm,
            spawn_vllm=args.spawn_vllm,
            vllm_screen=args.vllm_screen,
            vllm_screen_name=args.vllm_screen_name,
            vllm_tensor_parallel=args.vllm_tensor_parallel,
            vllm_gpu_memory=args.vllm_gpu_memory,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        return

    # Setup model: Experiment 0 uses vLLM by default, or HF when --no-vllm
    if args.experiment == 0:
        if args.no_vllm:
            model = utils.setup_model(actual_model_name)
        else:
            model = utils.setup_model("vllm:" + args.model, vllm_base_url=args.vllm_url)
    elif args.model.startswith("vllm:"):
        model = utils.setup_model(actual_model_name, vllm_base_url=args.vllm_url)
    else:
        model = utils.setup_model(actual_model_name)
    
    # Determine Cutoff Date based on model (only for Gemini models)
    if args.model == "gemini-2.5-flash":
        cutoff_date = datetime(2025, 6, 24)
        print(f"Selected Model: {args.model} (Cutoff: 2025-06-24)")
    elif args.model.startswith("gemini"):
        # Default for Gemini 2.0 Flash
        cutoff_date = datetime(2025, 2, 5)
        print(f"Selected Model: {args.model} (Cutoff: 2025-02-05)")
    elif args.model.startswith("vllm:"):
        # vLLM models - use a date before training (OLMo/Pythia training dates)
        cutoff_date = datetime(2024, 2, 1)  # Approximate OLMo training cutoff
        print(f"Selected Model: {args.model} (vLLM server, Cutoff: 2024-02-01)")
    elif "pythia" in args.model.lower() or "gpt-neo" in args.model.lower() or "rwkv" in args.model.lower():
        # Pile-trained (Peng et al., 2023b): Pythia, GPT-Neo, RWKV-4
        cutoff_date = datetime(2022, 1, 1)
        print(f"Selected Model: {args.model} (Pile-trained, Cutoff: 2022-01-01)")
    elif "nemotron" in args.model.lower():
        # Nemotron-CC (NVIDIA et al., 2025b)
        cutoff_date = datetime(2024, 9, 1)
        print(f"Selected Model: {args.model} (Nemotron-CC, Cutoff: 2024-09-01)")
    else:
        # OLMo / Dolma (Soldaini et al., 2024)
        cutoff_date = datetime(2024, 2, 1)
        print(f"Selected Model: {args.model} (Dolma-trained, Cutoff: 2024-02-01)")
        
    if args.benchmark:
        run_aggregate_benchmark(model, actual_model_name, cutoff_date)
        return

    # Experiment 0: paper training (seen) + unseen per model family (Pile / Dolma / Nemotron-CC)
    if args.experiment == 0:
        seen_source = _seen_source_for_experiment0_model(args.model)
        # Use parallel for vLLM and Gemini (API); not for local HF (Pythia/OLMo with --no-vllm)
        pw = args.parallel if (not args.no_vllm or "gemini" in args.model.lower()) else 0
        run_experiment_five(
            model, actual_model_name, cutoff_date,
            seen_source=seen_source,
            parallel_workers=pw,
            save_results=args.save_results,
            results_dir=args.results_dir,
        )
        return

    # Auto-select datasets for OLMo models
    if args.model.startswith("olmo") or args.model.startswith("vllm:"):
        if args.seen is None:
            args.seen = "dolma"
            print("Auto-selected 'dolma' as seen dataset for OLMo model")
        if args.unseen is None:
            args.unseen = "livecodebench"
            print("Auto-selected 'livecodebench' as unseen dataset for OLMo model")
    else:
        # Default datasets for non-OLMo models
        if args.seen is None:
            args.seen = "romeo"
        if args.unseen is None:
            args.unseen = "jibby"

    # Dataset Loading Logic
    seen_chunks = []
    unseen_chunks = []
    seen_name = ""
    unseen_name = ""

    print(f"Configuring Experiment...")
    
    # Load SEEN
    if args.seen == "romeo":
        seen_file = os.path.join(DATA_DIR, "romeo_and_juliet.txt")
        seen_chunks = load_and_chunk(seen_file)
        seen_name = "SEEN (Romeo & Juliet)"
    elif args.seen == "alice":
        seen_file = os.path.join(DATA_DIR, "alice_in_wonderland.txt")
        seen_chunks = load_and_chunk(seen_file)
        seen_name = "SEEN (Alice in Wonderland)"

    elif args.seen == "humaneval":
        seen_chunks = load_humaneval_chunks(n_samples=60)
        seen_name = "SEEN (HumanEval)"
    elif args.seen == "mbpp":
        seen_chunks = load_mbpp_chunks(n_samples=60)
        seen_name = "SEEN (MBPP)"
    elif args.seen == "gsm8k":
        seen_chunks = load_gsm8k_chunks(n_samples=60)
        seen_name = "SEEN (GSM8K)"
    elif args.seen == "dolma" or args.seen == "dolma-wiki":
        # Use Wikipedia from Dolma (much smaller and easier to load)
        seen_chunks = load_wikipedia_wikibooks_chunks(n_samples=60)
        seen_name = "SEEN (Dolma - Wikipedia)"

    # Load UNSEEN
    if args.unseen == "jibby":
        # Jibby Jones file
        unseen_file = os.path.join(DATA_DIR, "jibby_jones.txt")
        unseen_chunks = load_and_chunk(unseen_file)
        unseen_name = "UNSEEN (Jibby Jones)"

    elif args.unseen == "livecodebench":
        # Use v5 for OLMo models, otherwise use default
        lcb_version = "v5" if (args.model.startswith("olmo") or args.model.startswith("vllm:")) else "lite"
        unseen_chunks = load_livecodebench_chunks(cutoff_date, n_samples=60, version=lcb_version)
        unseen_name = f"UNSEEN (LiveCodeBench {lcb_version.upper()})"
    elif args.unseen == "ladn":
        unseen_file = os.path.join(DATA_DIR, "ladn_hbm4_memory.txt")
        unseen_chunks = load_and_chunk(unseen_file)
        unseen_name = "UNSEEN (LADN HBM4 Memory)"
    elif args.unseen == "bigobench":
        unseen_chunks = load_bigobench_chunks(n_samples=60)
        unseen_name = "UNSEEN (BigOBench)"
    
    print(f"Loaded {len(seen_chunks)} chunks for {seen_name}.")
    print(f"Loaded {len(unseen_chunks)} chunks for {unseen_name}.")
    
    if not seen_chunks or not unseen_chunks:
        print("Error: One or both datasets failed to load. Exiting.")
        return

    # Run Experiment
    # Using 100 samples by default for more stable CoDeC estimates
    seen_results = run_experiment(seen_name, seen_chunks, model, actual_model_name, n_samples=100, n_context=3, ignore_prefix_tokens=0, n_seeds=1)
    unseen_results = run_experiment(unseen_name, unseen_chunks, model, actual_model_name, n_samples=100, n_context=3, ignore_prefix_tokens=0, n_seeds=1)
    
    # Verify Hypothesis
    # Hypothesis: Seen data should show drops (negative delta) in confidence/match when context is added.
    # Unseen data should show increase (positive delta) or neutral.
    
    # Calculate CoDeC Contamination Scores
    score_seen = calculate_contamination_score(seen_results)
    score_unseen = calculate_contamination_score(unseen_results)
    
    avg_delta_seen = sum(r['delta_logprob'] for r in seen_results) / len(seen_results) if seen_results else 0
    avg_delta_unseen = sum(r['delta_logprob'] for r in unseen_results) / len(unseen_results) if unseen_results else 0
    
    print(f"\n--- FINAL RESULTS ---")
    print(f"Avg Delta LogProb SEEN: {avg_delta_seen:.4f}")
    print(f"Avg Delta LogProb UNSEEN: {avg_delta_unseen:.4f}")
    print(f"--------------------------------------------------")
    print(f"CoDeC Contamination Score SEEN:   {score_seen*100:.1f}%")
    print(f"CoDeC Contamination Score UNSEEN: {score_unseen*100:.1f}%")
    
    # Generate Plots
    plot_results(seen_results, unseen_results, output_file=f"results_plot_{args.seen}_{args.unseen}_{args.model}.png")
    plot_contamination_score(score_seen, args.seen, score_unseen, args.unseen, output_file=f"contamination_score_{args.seen}_{args.unseen}_{args.model}.png")
    
    # Generate Token-Level Figures
    run_token_diff_experiment(seen_name, seen_chunks, model, f"figure_seen_{args.seen}", actual_model_name, random_seed=42)
    run_token_diff_experiment(unseen_name, unseen_chunks, model, f"figure_unseen_{args.unseen}", actual_model_name, random_seed=42)
    
    if avg_delta_seen < avg_delta_unseen:
        print("SUCCESS: Hypothesis confirmed! Seen data had lower/negative boost directly compared to unseen data.")


def load_dataset_by_name(name, cutoff_date=None):
    """Dispatcher to load dataset chunks by simple name."""
    if name == "romeo":
        return load_and_chunk(os.path.join(DATA_DIR, "romeo_and_juliet.txt")), "Romeo & Juliet"
    elif name == "humaneval":
        return load_humaneval_chunks(n_samples=60), "HumanEval"
    elif name == "mbpp":
        return load_mbpp_chunks(n_samples=60), "MBPP"
    elif name == "dolma":
        # Use Wikipedia from Dolma (easier to load)
        return load_wikipedia_wikibooks_chunks(n_samples=60), "Dolma (Wikipedia)"
    elif name == "jibby":
        return load_and_chunk(os.path.join(DATA_DIR, "jibby_jones.txt")), "Jibby Jones"
    elif name == "livecodebench":
        # Use v5 version for LiveCodeBench
        return load_livecodebench_chunks(cutoff_date, n_samples=60, version="v5"), "LiveCodeBench v5"
    # Fallback/Others

    elif name == "ladn":
        return load_and_chunk(os.path.join(DATA_DIR, "ladn_hbm4_memory.txt")), "LADN HBM4 Memory"
    elif name == "bigobench":
        return load_bigobench_chunks(n_samples=60), "BigOBench"
    return [], name


def plot_aggregate_scores(results_data, model_name, output_file="aggregate_scores.png"):
    """Plots contamination scores for multiple datasets as boxplots (Paper Figure 2 style)."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    if not results_data:
        print(f"Skipping aggregate score plot for {model_name}: no results.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Data is list of dicts: {"Dataset": name, "Type": "Seen/Unseen", "Score": score}
    df = pd.DataFrame(results_data)
    
    # Add Model Name column for X-axis grouping
    df["Model"] = model_name
    
    # Convert score to percentage
    df["Score"] = df["Score"] * 100
    
    # Plot
    # X: Model, Y: Score, Hue: Type
    # This creates the side-by-side boxplots for Seen (Red) and Unseen (Blue)
    palette = {"Seen": "#d62728", "Seen (Pile)": "#d62728", "Seen (Dolma)": "#d62728", "Seen (Nemotron-CC)": "#d62728", "Unseen": "#1f77b4"}
    ax = sns.boxplot(x="Model", y="Score", hue="Type", data=df, palette=palette, width=0.5)
    
    # Add points (Strip plot) to show individual datasets
    sns.stripplot(x="Model", y="Score", hue="Type", data=df, dodge=True, palette="dark:.25", alpha=0.6, jitter=True, legend=False)
    
    # Annotate points with dataset names
    # This is tricky with dodge, but we can try simple annotation or skip if too crowded.
    # Given we have few points, let's print them.
        
    plt.title("CoDeC Contamination Score Distribution", fontsize=16)
    plt.ylabel("Contamination Score (%)", fontsize=12)
    plt.ylim(-5, 105) # Allow some buffer
    plt.legend(title="Dataset Type", loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Aggregate Score Plot saved to {output_file}")



def plot_contamination_score_per_dataset(results_data, model_name, output_file="contamination_score_per_dataset.png"):
    """Plots contamination scores for each dataset individually (Bar Chart)."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    if not results_data:
        print(f"Skipping per-dataset contamination plot for {model_name}: no results.")
        return
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Data is list of dicts: {"Dataset": name, "Type": "Seen/Unseen", "Score": score}
    df = pd.DataFrame(results_data)
    
    # Convert score to percentage
    df["Score"] = df["Score"] * 100
    
    # Plot
    # Hue by "Type" to distinguish Seen vs Unseen
    palette = {"Seen": "#d62728", "Seen (Pile)": "#d62728", "Seen (Dolma)": "#d62728", "Seen (Nemotron-CC)": "#d62728", "Unseen": "#1f77b4"}
    ax = sns.barplot(x="Dataset", y="Score", hue="Type", data=df, palette=palette)
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')
        
    plt.title(f"CoDeC Contamination Scores Across Datasets ({model_name})", fontsize=16)
    plt.ylabel("Contamination Score (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(title="Dataset Type")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Per-Dataset Contamination Score Plot saved to {output_file}")



def plot_aggregate_boxplots(all_results_data, output_file="aggregate_boxplots.png"):
    """Plots boxplots of Delta LogProb for all datasets."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Flatten data for boxplot
    # all_results_data is a list of (DatasetName, dict_result) tuples or similar
    # Actually, we need to restructure it.
    
    # Let's reconstruct the DataFrame
    plot_data = []
    for dataset_name, results_list, dataset_type in all_results_data:
        for r in results_list:
            plot_data.append({
                "Dataset": dataset_name,
                "Delta LogProb": r["delta_logprob"],
                "Type": dataset_type
            })
            
    df = pd.DataFrame(plot_data)
    
    # Boxplot
    palette = {"Seen": "#d62728", "Seen (Pile)": "#d62728", "Seen (Dolma)": "#d62728", "Seen (Nemotron-CC)": "#d62728", "Unseen": "#1f77b4"}
    ax = sns.boxplot(x="Dataset", y="Delta LogProb", hue="Type", data=df, palette=palette)
    sns.stripplot(x="Dataset", y="Delta LogProb", hue="Type", data=df, dodge=True, palette="dark:.25", alpha=0.4, legend=False)
    
    plt.title("Impact of Context on Model Confidence (Delta LogProb) - Aggregate", fontsize=16)
    plt.ylabel("Delta LogProb (ICL - Baseline)", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label="No Change")
    plt.xticks(rotation=15)
    plt.legend(title="Dataset Type")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Aggregate Boxplot saved to {output_file}")


# Experiment 0: Paper training (seen) datasets — The Pile (full list from Appendix B)
# HackerNews, Wikipedia, GitHub, ArXiv, DM Mathematics, Pile CommonCrawl, PubMed Central, Full Pile
# (Wikipedia Music, GitHub Licenses = low-diversity; omit if not in ArmelR)
PILE_EXPERIMENT_FIVE_CONFIGS = [
    ("hackernews", "HackerNews"),
    ("wikipedia", "Wikipedia"),
    ("github", "GitHub"),
    ("arxiv", "ArXiv"),
    ("dm_mathematics", "DM Mathematics"),
    ("pile_cc", "Pile CommonCrawl"),
    ("pubmed_central", "PubMed Central"),
    #("full_pile", "Full Pile"),
]

# Paper Appendix B: Dolma training (seen) — C4, Wikipedia, Pes2o v2, Reddit v5, CommonCrawl head/middle/tail
# Each entry: (key, loader_fn, display_name); loader_fn(n_samples=n) returns list of text chunks
DOLMA_SEEN_CONFIGS = [
    ("c4", load_c4_chunks, "C4"),
    ("wikipedia", load_wikipedia_wikibooks_chunks, "Wikipedia"),
    ("pes2o_v2", lambda n_samples=5000: load_dolma_chunks(n_samples=n_samples, source="pes2o"), "Pes2o v2"),
    ("reddit_v5", lambda n_samples=5000: load_dolma_chunks(n_samples=n_samples, source="reddit"), "Reddit v5"),
    ("commoncrawl_head", lambda n_samples=5000: load_dolma_chunks(n_samples=n_samples, source="commoncrawl"), "CommonCrawl head"),
    ("commoncrawl_middle", lambda n_samples=5000: load_dolma_chunks(n_samples=n_samples, source="commoncrawl"), "CommonCrawl middle"),
    ("commoncrawl_tail", lambda n_samples=5000: load_dolma_chunks(n_samples=n_samples, source="commoncrawl"), "CommonCrawl tail"),
]

# Paper Appendix B: Nemotron-CC training (seen) — Wikipedia, ArXiv, CommonCrawl (7 quality buckets)
# Quality buckets: 2024 high, 2020 high/low, 2019 medium, 2016 medium, 2014 low, 2013 high
NEMOTRON_CC_SEEN_CONFIGS = [
    ("wikipedia", load_wikipedia_wikibooks_chunks, "Wikipedia"),
    ("arxiv", lambda n_samples=5000: load_pile_chunks("arxiv", n_samples=n_samples), "ArXiv"),
    ("cc_2024_high", load_c4_chunks, "CommonCrawl 2024 (high)"),
    ("cc_2020_high", load_c4_chunks, "CommonCrawl 2020 (high)"),
    ("cc_2020_low", load_c4_chunks, "CommonCrawl 2020 (low)"),
    ("cc_2019_medium", load_c4_chunks, "CommonCrawl 2019 (medium)"),
    ("cc_2016_medium", load_c4_chunks, "CommonCrawl 2016 (medium)"),
    ("cc_2014_low", load_c4_chunks, "CommonCrawl 2014 (low)"),
    ("cc_2013_high", load_c4_chunks, "CommonCrawl 2013 (high)"),
]

# Paper Appendix B.2: Unseen datasets — per model family (paper lists)
# Pile-trained: gsm8k, GPQA Diamond, IFEval, HumanEval, FRAMES, AIME 2024/2025, LiveCodeBench v1/v5, BFCL v3, BBQ, RewardBench v1/v2, MATH 500
# + Project Gutenberg (3 books), HuggingFace (global-news, Amazon-Reviews-2023), website (Ukraine updates)
# OLMo: FRAMES, AIME 2024, AIME 2025, LiveCodeBench v5, BFCL v3, RewardBench v1, RewardBench v2 + Gutenberg + HF + website
# Nemotron-H: only AIME 2025 (paper)
def _lcb(cutoff, version, n_samples):
    return load_livecodebench_chunks(cutoff, n_samples=n_samples, version=version)


def _make_lcb_loader(version, default_cutoff=None):
    """Loader that accepts optional cutoff_date (uses model cutoff when provided)."""
    from datetime import datetime
    default = default_cutoff or datetime(2022, 1, 1)
    def loader(n_samples=20, cutoff_date=None):
        c = cutoff_date if cutoff_date is not None else default
        return load_livecodebench_chunks(c, n_samples=n_samples, version=version)
    return loader


UNSEEN_FOR_PILE = [
    ("gsm8k", load_gsm8k_chunks, "GSM8K"),
    ("gpqa_diamond", load_gpqa_diamond_chunks, "GPQA Diamond"),
    ("ifeval", load_ifeval_chunks, "IFEval"),
    ("humaneval", load_humaneval_chunks, "HumanEval"),
    ("frames", load_frames_chunks, "FRAMES"),
    ("aime_2024", load_aime_2024_chunks, "AIME 2024"),
    ("aime_2025", load_aime_2025_chunks, "AIME 2025"),
    ("livecodebench_v1", _make_lcb_loader("v1"), "LiveCodeBench v1"),
    ("livecodebench_v5", _make_lcb_loader("v5"), "LiveCodeBench v5"),
    ("bfcl_v3", load_bfcl_v3_chunks, "BFCL v3"),
    ("bbq", load_bbq_chunks, "BBQ"),
    ("rewardbench_v1", load_rewardbench_v1_chunks, "RewardBench v1"),
    ("rewardbench_v2", load_rewardbench_v2_chunks, "RewardBench v2"),
    ("math_500", load_math_500_chunks, "MATH 500"),
    ("gutenberg_colonial", lambda n_samples=20: load_gutenberg_book_chunks("colonial_memories", n_samples), "Gutenberg: Colonial Memories"),
    ("gutenberg_jibby", lambda n_samples=20: load_gutenberg_book_chunks("jibby_jones", n_samples), "Gutenberg: Jibby Jones"),
    ("gutenberg_corbin", lambda n_samples=20: load_gutenberg_book_chunks("corbin_necklace", n_samples), "Gutenberg: The Corbin necklace"),
    ("global_news", load_global_news_chunks, "NickyNicky/global-news-dataset"),
    ("amazon_reviews_2023", load_amazon_reviews_2023_chunks, "McAuley-Lab/Amazon-Reviews-2023"),
    # Ukraine article removed: live webpage whose content changes daily; not reproducible across runs
]
UNSEEN_FOR_DOLMA = [
    ("frames", load_frames_chunks, "FRAMES"),
    ("aime_2024", load_aime_2024_chunks, "AIME 2024"),
    ("aime_2025", load_aime_2025_chunks, "AIME 2025"),
    ("livecodebench_v5", _make_lcb_loader("v5", default_cutoff=datetime(2024, 2, 1)), "LiveCodeBench v5"),
    ("bfcl_v3", load_bfcl_v3_chunks, "BFCL v3"),
    ("rewardbench_v1", load_rewardbench_v1_chunks, "RewardBench v1"),
    ("rewardbench_v2", load_rewardbench_v2_chunks, "RewardBench v2"),
    ("gutenberg_colonial", lambda n_samples=20: load_gutenberg_book_chunks("colonial_memories", n_samples), "Gutenberg: Colonial Memories"),
    ("gutenberg_jibby", lambda n_samples=20: load_gutenberg_book_chunks("jibby_jones", n_samples), "Gutenberg: Jibby Jones"),
    ("gutenberg_corbin", lambda n_samples=20: load_gutenberg_book_chunks("corbin_necklace", n_samples), "Gutenberg: The Corbin necklace"),
    ("global_news", load_global_news_chunks, "NickyNicky/global-news-dataset"),
    ("amazon_reviews_2023", load_amazon_reviews_2023_chunks, "McAuley-Lab/Amazon-Reviews-2023"),
    # Ukraine article removed: live webpage whose content changes daily; not reproducible across runs
]
UNSEEN_FOR_NEMOTRON_CC = [
    ("aime_2025", load_aime_2025_chunks, "AIME 2025"),
    ("gpqa_diamond", load_gpqa_diamond_chunks, "GPQA Diamond"),
    ("livecodebench_v5", _make_lcb_loader("v5", default_cutoff=datetime(2023, 6, 1)), "LiveCodeBench v5"),
]
# Legacy: single list used when per-family unseen was not implemented (default to Pile unseen)
UNSEEN_DATASETS_FOR_EXPERIMENT_5 = UNSEEN_FOR_PILE


def compute_auc_seen_vs_unseen(results_score_data):
    """
    Computes dataset-level AUC for separating seen vs unseen (paper Table 1).
    Higher CoDeC score = more contaminated = seen. Labels: 1=seen, 0=unseen.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  Warning: sklearn not found. Install with: pip install scikit-learn")
        return None

    scores = []
    labels = []  # 1 = seen, 0 = unseen
    for r in results_score_data:
        scores.append(r["Score"])
        t = r.get("Type", "")
        labels.append(1 if ("Seen" in t or "Pile" in t or "Dolma" in t or "Nemotron" in t) else 0)

    if len(set(labels)) < 2:
        print("  Warning: Need both seen and unseen datasets for AUC.")
        return None

    auc = roc_auc_score(labels, scores)
    return auc


def _set_reproducibility_seeds(seed=42):
    """Set all relevant RNG seeds for reproducible experiments."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from datasets import set_seed
        set_seed(seed)
    except (ImportError, AttributeError):
        pass


def _sanitize_for_filename(s: str) -> str:
    """Replace chars unsafe for filenames."""
    return re.sub(r'[^\w\-.]', '_', (s or "").strip())


def _save_dataset_results(results_dir: str, model_name: str, dataset_name: str, dataset_type: str, score: float, exp_results: list, n_samples: int, n_context: int = 3):
    """Save per-dataset CoDeC results to JSON for later comparison."""
    safe_model = _sanitize_for_filename(model_name.replace("/", "_"))
    safe_dataset = _sanitize_for_filename(dataset_name)
    out_dir = os.path.join(results_dir, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_dataset}.json")
    payload = {
        "model": model_name,
        "dataset": dataset_name,
        "type": dataset_type,
        "score": score,
        "n_samples": n_samples,
        "n_context": n_context,
        "samples": [
            {
                "delta_logprob": r["delta_logprob"],
                "base_mean_logprob": r.get("base_mean_logprob"),
                "icl_mean_logprob": r.get("icl_mean_logprob"),
            }
            for r in exp_results
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {out_path}")


def run_experiment_five(model, model_name, cutoff_date, n_samples=None, skip_token_figures=False, seen_source="pile", parallel_workers=0, save_results=False, results_dir="results", seen_datasets_limit=None, unseen_datasets_limit=None, unseen_dataset_keys=None, unseen_only=False, seen_dataset_keys=None):
    """
    Runs Experiment 0: paper training (seen) + unseen datasets with subsampling only.
    seen_source: "pile" | "dolma" | "nemotron_cc" — selects which training (seen) datasets to use per model family.
    parallel_workers: when > 0 and model is vLLM, run logprob scoring in parallel with N workers.
    seen_datasets_limit: if set, use only first N seen datasets (e.g. 2 for quick runs).
    unseen_datasets_limit: if set, use only first N unseen datasets (e.g. 2 for quick runs).
    unseen_dataset_keys: if set, use only these unseen dataset keys (e.g. ["gpqa_diamond", "livecodebench_v5"]).
    unseen_only: if True, skip seen datasets and run only unseen (no AUC).
    seen_dataset_keys: if set, use only these seen dataset keys (e.g. ["hackernews", "wikipedia"]). Empty list = skip seen.
    Uses EXPERIMENT0_DATASET_SAMPLES per dataset by default.
    Computes and reports AUC for separating seen vs unseen (paper Table 1).
    Returns (auc, results_score_data) for aggregation when running all models.
    """
    _set_reproducibility_seeds(42)
    n_samples = n_samples if n_samples is not None else EXPERIMENT0_DATASET_SAMPLES
    results_score_data = []
    all_raw_results = []

    # Resolve seen type label and description by source
    if seen_source == "pile":
        seen_type_label = "Seen (Pile)"
        seen_desc = "Pile: HackerNews, Wikipedia, GitHub, ArXiv, DM Mathematics, Pile CommonCrawl, PubMed Central, Full Pile"
    elif seen_source == "dolma":
        seen_type_label = "Seen (Dolma)"
        seen_desc = "Dolma: C4, Wikipedia, Pes2o v2, Reddit v5, CommonCrawl head/middle/tail"
    elif seen_source == "nemotron_cc":
        seen_type_label = "Seen (Nemotron-CC)"
        seen_desc = "Nemotron-CC: Wikipedia, ArXiv, CommonCrawl (7 quality buckets)"
    else:
        seen_type_label = "Seen"
        seen_desc = "(none)"

    # Unseen list per model family (paper Appendix B.2)
    unseen_list_full = UNSEEN_FOR_PILE if seen_source == "pile" else (UNSEEN_FOR_DOLMA if seen_source == "dolma" else UNSEEN_FOR_NEMOTRON_CC)
    if unseen_dataset_keys is not None:
        key_set = set(unseen_dataset_keys)
        unseen_list = [d for d in unseen_list_full if d[0] in key_set]
    else:
        unseen_list = unseen_list_full[:unseen_datasets_limit] if unseen_datasets_limit else unseen_list_full
    if unseen_dataset_keys is not None and not unseen_list:
        print(
            f"Warning: no unseen datasets matched for seen_source='{seen_source}'. "
            f"Requested={list(unseen_dataset_keys)}, allowed={[d[0] for d in unseen_list_full]}"
        )
    unseen_desc = ", ".join(d[2] for d in unseen_list[:5]) + (" ..." if len(unseen_list) > 5 else "")

    # seen_dataset_keys: explicit list; empty = skip seen
    if seen_dataset_keys is not None and len(seen_dataset_keys) == 0:
        unseen_only = True

    print("\n========== EXPERIMENT 5: Paper Datasets (subsampling only) ==========")
    if unseen_only:
        print("UNSEEN ONLY (skipping seen datasets)")
    print(f"SEEN ({seen_source}): {seen_desc}" if not unseen_only else "SEEN: (skipped)")
    print(f"UNSEEN ({seen_source}): {unseen_desc}")
    print(f"Subsampling: {n_samples} samples per dataset.")

    # Paper: "we selected 1 000 random samples" — load pool then random sample
    # Sort chunks by content hash so that streaming load order does not affect sampling.
    # This ensures reproducibility when HuggingFace streaming returns different order across runs.
    def _random_sample_chunks(chunks, n, display_name):
        if len(chunks) <= n:
            return chunks[:n]
        # Deterministic sort: same chunks -> same order regardless of load order
        sorted_chunks = sorted(chunks, key=lambda c: hashlib.sha256((c or "").encode("utf-8")).hexdigest())
        rng = random.Random(42 + (hash(display_name) % (2**31)))
        return rng.sample(sorted_chunks, n)

    load_n = min(EXPERIMENT0_RANDOM_SAMPLE_POOL, max(n_samples, 2000))  # pool for random sampling
    # Process SEEN (model-family–specific training datasets; load one at a time to save memory)
    if not unseen_only and seen_source == "pile":
        if seen_dataset_keys:
            key_set = set(seen_dataset_keys)
            pile_configs = [c for c in PILE_EXPERIMENT_FIVE_CONFIGS if c[0] in key_set]
        else:
            pile_configs = PILE_EXPERIMENT_FIVE_CONFIGS[:seen_datasets_limit] if seen_datasets_limit else PILE_EXPERIMENT_FIVE_CONFIGS
        for config_name, display_name in pile_configs:
            chunks = load_pile_chunks(config_name, n_samples=load_n)
            if not chunks:
                print(f"  Skipping {display_name} (failed to load)")
                continue
            candidate_n = min(len(chunks), max(n_samples, n_samples * EXPERIMENT0_CANDIDATE_MULTIPLIER))
            chunks = _random_sample_chunks(chunks, candidate_n, display_name)
            print(f"\n--- Processing SEEN: {display_name} ---")
            exp_results = run_experiment(display_name, chunks, model, model_name, n_samples=min(n_samples, len(chunks)), n_context=3, ignore_prefix_tokens=0, n_seeds=1, parallel_workers=parallel_workers)
            score = calculate_contamination_score(exp_results)
            results_score_data.append({"Dataset": display_name, "Type": seen_type_label, "Score": score})
            all_raw_results.append((display_name, exp_results, seen_type_label))
            if save_results:
                _save_dataset_results(results_dir, model_name, display_name, seen_type_label, score, exp_results, n_samples, n_context=3)
            if not skip_token_figures:
                run_token_diff_experiment(display_name, chunks, model, f"figure_experiment0_pile_{config_name}", model_name, n_plots=2, random_seed=42)
    elif not unseen_only and seen_source in ("dolma", "nemotron_cc"):
        configs_full = DOLMA_SEEN_CONFIGS if seen_source == "dolma" else NEMOTRON_CC_SEEN_CONFIGS
        if seen_dataset_keys:
            key_set = set(seen_dataset_keys)
            configs = [c for c in configs_full if c[0] in key_set]
        else:
            configs = configs_full[:seen_datasets_limit] if seen_datasets_limit else configs_full
        for config_key, loader_fn, display_name in configs:
            chunks = loader_fn(n_samples=load_n)
            if not chunks:
                print(f"  Skipping {display_name} (failed to load)")
                continue
            candidate_n = min(len(chunks), max(n_samples, n_samples * EXPERIMENT0_CANDIDATE_MULTIPLIER))
            chunks = _random_sample_chunks(chunks, candidate_n, display_name)
            print(f"\n--- Processing SEEN: {display_name} ---")
            exp_results = run_experiment(display_name, chunks, model, model_name, n_samples=min(n_samples, len(chunks)), n_context=3, ignore_prefix_tokens=0, n_seeds=1, parallel_workers=parallel_workers)
            score = calculate_contamination_score(exp_results)
            results_score_data.append({"Dataset": display_name, "Type": seen_type_label, "Score": score})
            all_raw_results.append((display_name, exp_results, seen_type_label))
            if save_results:
                _save_dataset_results(results_dir, model_name, display_name, seen_type_label, score, exp_results, n_samples, n_context=3)
            if not skip_token_figures:
                run_token_diff_experiment(display_name, chunks, model, f"figure_experiment0_{seen_source}_{config_key}", model_name, n_plots=2, random_seed=42)

    # Process UNSEEN (paper Appendix B.2, per-family list)
    # Use same n_context=3 as SEEN (paper: same k for all datasets; was bug: unseen used 1)
    LCB_KEYS = ("livecodebench_v1", "livecodebench_v5")
    for key, loader_fn, display_name in unseen_list:
        try:
            chunks = loader_fn(n_samples=load_n, cutoff_date=cutoff_date) if key in LCB_KEYS else loader_fn(n_samples=load_n)
        except TypeError:
            chunks = loader_fn(n_samples=load_n)
        if not chunks:
            print(f"  Skipping {display_name} (failed to load)")
            continue
        candidate_n = min(len(chunks), max(n_samples, n_samples * EXPERIMENT0_CANDIDATE_MULTIPLIER))
        chunks = _random_sample_chunks(chunks, candidate_n, display_name)
        print(f"\n--- Processing UNSEEN: {display_name} ---")
        exp_results = run_experiment(
            display_name, chunks, model, model_name,
            n_samples=min(n_samples, len(chunks)), n_context=3, ignore_prefix_tokens=0, n_seeds=1,
            parallel_workers=parallel_workers
        )
        score = calculate_contamination_score(exp_results)
        results_score_data.append({"Dataset": display_name, "Type": "Unseen", "Score": score})
        all_raw_results.append((display_name, exp_results, "Unseen"))
        if save_results:
            _save_dataset_results(results_dir, model_name, display_name, "Unseen", score, exp_results, n_samples, n_context=3)

        if not skip_token_figures:
            run_token_diff_experiment(
                display_name, chunks, model, f"figure_experiment0_unseen_{key}",
                model_name, n_plots=2, random_seed=42
            )

    # Compute AUC for separating seen vs unseen (paper Table 1)
    auc = compute_auc_seen_vs_unseen(results_score_data) if not unseen_only and results_score_data else None

    print("\n--- EXPERIMENT 5 RESULTS ---")
    for r in results_score_data:
        print(f"[{r['Type']}] {r['Dataset']}: {r['Score']*100:.1f}%")
    if auc is not None:
        print(f"\n*** AUC (seen vs unseen): {auc*100:.1f}% ***")
        print("(Paper Table 1: CoDeC achieves ~99.9% dataset-level AUC)")

    # Sanitize model name for filenames (replace / with _)
    safe_model = model_name.replace("/", "_")

    # Plots
    plot_aggregate_scores(results_score_data, model_name, f"experiment0_aggregate_scores_{safe_model}.png")
    plot_contamination_score_per_dataset(results_score_data, model_name, f"experiment0_contamination_per_dataset_{safe_model}.png")

    # Save AUC to file
    if auc is not None:
        with open(f"experiment0_auc_{safe_model}.txt", "w") as f:
            f.write(f"AUC (seen vs unseen): {auc*100:.1f}%\n")

    # Save full summary for later comparison (when --save-results)
    if save_results:
        out_dir = os.path.join(results_dir, safe_model)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.json")
        summary = {
            "model": model_name,
            "auc": auc,
            "scores": results_score_data,
            "n_samples": n_samples,
            "seen_source": seen_source,
            "timestamp": datetime.now().isoformat(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")

    print("\nExperiment 0 complete. Outputs: experiment0_*.png, experiment0_auc_*.txt" + (" + results/" if save_results else ""))
    return auc, results_score_data


# Models to run for --experiment 0 --all-models (paper Table 1: same models/datasets, subsampling only)
# Pile: Pythia, GPT-Neo, RWKV-4 (Peng et al., 2023b); Dolma: OLMo (Soldaini et al., 2024); Nemotron-CC: Nemotron (NVIDIA et al., 2025b)
EXPERIMENT0_ALL_MODEL_KEYS = [
    "pythia-410m", "pythia-1.4b", "pythia-12b",
    "gpt-neo-1.3b", "gpt-neo-20b",
    "rwkv-4-430m", "rwkv-4-3b", "rwkv-4-14b",
    "olmo-1b", "olmo-7b",
    "nemotron-v2-9b", "nemotron-h-8b", "nemotron-h-56b",
]
EXPERIMENT0_MODEL_DISPLAY = {
    "pythia-410m": "Pythia 410M",
    "pythia-1.4b": "Pythia 1.4B",
    "pythia-12b": "Pythia 12B",
    "gpt-neo-1.3b": "GPT-Neo 1.3B",
    "gpt-neo-20b": "GPT-Neo 20B",
    "rwkv-4-430m": "RWKV-4 430M",
    "rwkv-4-3b": "RWKV-4 3B",
    "rwkv-4-14b": "RWKV-4 14B",
    "olmo-1b": "OLMo 1B",
    "olmo-7b": "OLMo 7B",
    "nemotron-v2-9b": "Nemotron v2 9B",
    "nemotron-h-8b": "Nemotron-H 8B",
    "nemotron-h-56b": "Nemotron-H 56B",
}


def _cutoff_for_experiment0_model(model_key):
    """Cutoff date per paper: Pile (2022-01-01), Dolma (2024-02-01), Nemotron-CC (2024-09-01)."""
    from datetime import datetime
    k = model_key.lower()
    if "pythia" in k or "gpt-neo" in k or "rwkv" in k:
        return datetime(2022, 1, 1)   # Pile-trained (Peng et al., 2023b)
    if "olmo" in k:
        return datetime(2024, 2, 1)   # Dolma-trained (Soldaini et al., 2024)
    if "nemotron" in k:
        return datetime(2024, 9, 1)   # Nemotron-CC (NVIDIA et al., 2025b; pretraining cutoff ~Sept 2024)
    return datetime(2024, 2, 1)


def _seen_source_for_experiment0_model(model_key):
    """Training (seen) datasets per paper: Pile, Dolma, or Nemotron-CC."""
    k = model_key.lower()
    if "pythia" in k or "gpt-neo" in k or "rwkv" in k:
        return "pile"
    if "olmo" in k:
        return "dolma"
    if "nemotron" in k:
        return "nemotron_cc"
    if "davinci" in k or "babbage" in k or k.startswith("openai:"):
        return "pile"
    return "pile"


def print_models_and_datasets():
    """Print list of models and their SEEN/UNSEEN datasets for verification (see also MODELS_AND_DATASETS.md)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: MODELS AND THEIR DATASETS (for verification)")
    print("Subsampling: {} samples per dataset".format(EXPERIMENT0_DATASET_SAMPLES))
    print("=" * 70)

    seen_names = {
        "pile": [d[1] for d in PILE_EXPERIMENT_FIVE_CONFIGS],
        "dolma": [d[2] for d in DOLMA_SEEN_CONFIGS],
        "nemotron_cc": [d[2] for d in NEMOTRON_CC_SEEN_CONFIGS],
    }
    unseen_lists = {
        "pile": [d[2] for d in UNSEEN_FOR_PILE],
        "dolma": [d[2] for d in UNSEEN_FOR_DOLMA],
        "nemotron_cc": [d[2] for d in UNSEEN_FOR_NEMOTRON_CC],
    }

    for model_key in EXPERIMENT0_ALL_MODEL_KEYS:
        display = EXPERIMENT0_MODEL_DISPLAY.get(model_key, model_key)
        source = _seen_source_for_experiment0_model(model_key)
        print("\n--- {} ---".format(display))
        print("  Training (SEEN) source: {}".format(source))
        print("  SEEN datasets: {}".format(", ".join(seen_names.get(source, []))))
        print("  UNSEEN datasets: {}".format(", ".join(unseen_lists.get(source, []))))

    print("\n" + "-" * 70)
    print("SEEN by source:")
    print("  pile:        " + ", ".join(seen_names["pile"]))
    print("  dolma:       " + ", ".join(seen_names["dolma"]))
    print("  nemotron_cc: " + ", ".join(seen_names["nemotron_cc"]))
    print("UNSEEN by source (paper Appendix B.2):")
    print("  pile:        " + ", ".join(unseen_lists["pile"]))
    print("  dolma:       " + ", ".join(unseen_lists["dolma"]))
    print("  nemotron_cc: " + ", ".join(unseen_lists["nemotron_cc"]))
    print("=" * 70 + "\n")


def _spawn_vllm_server(model_id, tensor_parallel_size=2, gpu_memory_utilization=0.5, max_model_len=4000):
    """Spawn vLLM server as subprocess. Returns Popen process handle."""
    cmd = [
        "vllm", "serve", model_id,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    return proc


def _wait_for_vllm(base_url="http://localhost:8000/v1", proc=None, timeout=300, poll_interval=5):
    """Poll until vLLM server is ready or timeout. If proc is given, return False if it exits early."""
    # vLLM health at root /health (base_url is typically .../v1)
    base = base_url.rstrip("/").rsplit("/", 1)[0] if "/v1" in base_url else base_url.rstrip("/")
    health_url = base + "/health"
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            print("\nvLLM server process exited unexpectedly.")
            return False
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("\nvLLM server is ready.")
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)
        print(".", end="", flush=True)
    print("\nvLLM server did not become ready in time.")
    return False


def _stop_vllm_server(proc):
    """Terminate vLLM server process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("vLLM server stopped.")


def _ensure_screen_exists(screen_name="vllmserver"):
    """Create screen session if it does not exist."""
    r = subprocess.run(["screen", "-list"], capture_output=True, text=True)
    if screen_name in (r.stdout or "") or screen_name in (r.stderr or ""):
        return  # already exists
    subprocess.run(["screen", "-dmS", screen_name, "bash"], check=False)
    time.sleep(1)


def _start_vllm_in_screen(model_id, screen_name="vllmserver", tensor_parallel_size=2, gpu_memory_utilization=0.5, max_model_len=4000):
    """Start vLLM server in GNU screen session. Creates/resumes screen, runs server. Use screen -r vllmserver to attach."""
    _ensure_screen_exists(screen_name)
    cmd = (
        f"vllm serve {model_id} --trust-remote-code "
        f"--tensor-parallel-size {tensor_parallel_size} "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--max-model-len {max_model_len}"
    )
    subprocess.run(
        ["screen", "-S", screen_name, "-X", "stuff", cmd + "\n"],
        check=True,
    )
    print(f"Started vLLM in screen '{screen_name}': {cmd}")
    print(f"  Attach with: screen -r {screen_name}")


def _stop_vllm_in_screen(screen_name="vllmserver"):
    """Send Ctrl+C to screen to stop vLLM server."""
    subprocess.run(["screen", "-S", screen_name, "-X", "stuff", "\003"], check=False)
    print(f"Sent stop to vLLM in screen '{screen_name}'.")
    time.sleep(5)  # allow vLLM to shut down before next model


def run_experiment_five_all_models(model_map, vllm_url="http://localhost:8000/v1", parallel_workers=4, use_vllm=True,
                                   spawn_vllm=False, vllm_screen=False, vllm_screen_name="vllmserver",
                                   vllm_tensor_parallel=2, vllm_gpu_memory=0.5, vllm_max_model_len=4000):
    """
    Runs Experiment 0 for all paper models. Paper: "For each dataset, we selected 1 000 random samples."
    Produces: (1) Table of AUC (Model, CoDeC %), (2) Figure of contamination score (boxplot).
    Cutoff: Pile (Pythia, GPT-Neo, RWKV-4) 2022-01-01; Dolma (OLMo) 2024-02-01; Nemotron-CC 2024-09-01.
    use_vllm: if True, use vLLM server; if False, use Hugging Face (transformers) directly.
    spawn_vllm: if True and use_vllm, start vLLM server per model, run experiment, then stop server.
    vllm_screen: if True, use GNU screen (screen -r vllmserver) to run vLLM. Resumes screen, runs server, detaches, runs experiment.
    """
    from datetime import datetime
    import seaborn as sns

    n_samples = EXPERIMENT0_DATASET_SAMPLES
    table_rows = []
    all_results_for_plot = []
    pw = parallel_workers if use_vllm else 0
    vllm_proc = None

    for model_key in EXPERIMENT0_ALL_MODEL_KEYS:
        actual_model_name = model_map.get(model_key)
        if actual_model_name is None:
            print(f"Skipping {model_key} (not in model_map)")
            continue
        # Skip vLLM models in map (e.g. vllm:olmo-7b) - use HF ID for spawning
        hf_model_id = actual_model_name if not str(actual_model_name).startswith("vllm:") else None
        if use_vllm and (spawn_vllm or vllm_screen) and not hf_model_id:
            print(f"Skipping {model_key} (no HF model ID for spawn)")
            continue
        display_name = EXPERIMENT0_MODEL_DISPLAY.get(model_key, model_key)
        cutoff_date = _cutoff_for_experiment0_model(model_key)

        print(f"\n{'='*60}")
        print(f"Running Experiment 0 for {display_name} ({'vLLM' if use_vllm else 'Hugging Face'})")
        print(f"{'='*60}")

        if use_vllm and vllm_screen:
            # Stop any existing vLLM in screen, then start new one
            _stop_vllm_in_screen(vllm_screen_name)
            _start_vllm_in_screen(
                hf_model_id,
                screen_name=vllm_screen_name,
                tensor_parallel_size=vllm_tensor_parallel,
                gpu_memory_utilization=vllm_gpu_memory,
                max_model_len=vllm_max_model_len,
            )
            if not _wait_for_vllm(vllm_url, proc=None, timeout=600):
                print(f"Failed to start vLLM for {display_name}, skipping.")
                continue
        elif use_vllm and spawn_vllm:
            vllm_proc = _spawn_vllm_server(
                hf_model_id,
                tensor_parallel_size=vllm_tensor_parallel,
                gpu_memory_utilization=vllm_gpu_memory,
                max_model_len=vllm_max_model_len,
            )
            if not _wait_for_vllm(vllm_url, proc=vllm_proc, timeout=600):
                _stop_vllm_server(vllm_proc)
                print(f"Failed to start vLLM for {display_name}, skipping.")
                continue

        try:
            if use_vllm:
                vllm_model_name = "vllm:" + model_key.replace("vllm:", "")
                model = utils.setup_model(vllm_model_name, vllm_base_url=vllm_url)
                model_name = model_key
            else:
                model = utils.setup_model(actual_model_name)
                model_name = actual_model_name
        except Exception as e:
            if use_vllm and spawn_vllm and vllm_proc:
                _stop_vllm_server(vllm_proc)
            elif use_vllm and vllm_screen:
                _stop_vllm_in_screen(vllm_screen_name)
            print(f"Failed to load {display_name}: {e}")
            continue

        try:
            seen_source = _seen_source_for_experiment0_model(model_key)
            auc, results_score_data = run_experiment_five(
                model, model_name, cutoff_date,
                n_samples=n_samples, skip_token_figures=True, seen_source=seen_source,
                parallel_workers=pw
            )
        finally:
            if use_vllm and spawn_vllm and vllm_proc and vllm_proc.poll() is None:
                _stop_vllm_server(vllm_proc)
                vllm_proc = None
            elif use_vllm and vllm_screen:
                _stop_vllm_in_screen(vllm_screen_name)

        auc_pct = auc * 100 if auc is not None else float("nan")
        table_rows.append({"Model": display_name, "CoDeC": f"{auc_pct:.1f}%"})
        for r in results_score_data:
            all_results_for_plot.append({
                "Model": display_name,
                "Dataset": r["Dataset"],
                "Type": r["Type"],
                "Score": r["Score"] * 100,
            })

    # Table: AUC per model (paper Table 1)
    df_table = pd.DataFrame(table_rows)
    table_path = "experiment0_auc_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"\n--- AUC Table (CoDeC) ---")
    print(df_table.to_string(index=False))
    print(f"Saved to {table_path}")

    # Figure: Contamination score boxplot (paper style - red Seen, blue Unseen)
    if not all_results_for_plot:
        print("No results for figure.")
        return

    df_plot = pd.DataFrame(all_results_for_plot)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    palette = {"Seen": "#d62728", "Seen (Pile)": "#d62728", "Seen (Dolma)": "#d62728", "Seen (Nemotron-CC)": "#d62728", "Unseen": "#1f77b4"}
    ax = sns.boxplot(x="Model", y="Score", hue="Type", data=df_plot, palette=palette, width=0.6)
    sns.stripplot(x="Model", y="Score", hue="Type", data=df_plot, dodge=True, palette="dark:.25", alpha=0.5, size=4, legend=False)
    plt.title("CoDeC: Contamination Score", fontsize=16)
    plt.ylabel("Contamination Score (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(rotation=15)
    plt.legend(title="Type")
    plt.tight_layout()
    fig_path = "experiment0_contamination_score_all_models.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Contamination score figure saved to {fig_path}")


def run_aggregate_benchmark(model, model_name, cutoff_date):
    """Runs the experiment on the specific list of datasets requested."""
    # Include Dolma in seen list if using OLMo model (since OLMo was trained on Dolma)
    if "olmo" in model_name.lower() or model_name.startswith("vllm:"):
        seen_list = ["dolma", "mbpp", "humaneval"]
    else:
        seen_list = ["romeo", "mbpp", "humaneval"]
    unseen_list = ["jibby", "livecodebench", "ladn", "bigobench"]

    results_score_data = [] # For the bar chart
    all_raw_results = [] # For the boxplot: tuple (Name, ResultsList, Type)

    print("\n========== RUNNING AGGREGATE BENCHMARK ==========")

    # Process SEEN
    for name in seen_list:
        chunks, display_name = load_dataset_by_name(name, cutoff_date)
        if not chunks: continue

        exp_results = run_experiment(display_name, chunks, model, model_name, n_samples=100, n_context=1, ignore_prefix_tokens=0, n_seeds=1)
        score = calculate_contamination_score(exp_results)
        results_score_data.append({"Dataset": display_name, "Type": "Seen", "Score": score})
        all_raw_results.append((display_name, exp_results, "Seen"))

        # Generate Token Level Plots (Cumulative)
        run_token_diff_experiment(display_name, chunks, model, f"figure_seen_{name}", model_name, n_plots=2, random_seed=42)

    # Process UNSEEN
    for name in unseen_list:
        chunks, display_name = load_dataset_by_name(name, cutoff_date)
        if not chunks: continue

        exp_results = run_experiment(display_name, chunks, model, model_name, n_samples=100, n_context=1, ignore_prefix_tokens=0, n_seeds=1)
        score = calculate_contamination_score(exp_results)
        results_score_data.append({"Dataset": display_name, "Type": "Unseen", "Score": score})
        all_raw_results.append((display_name, exp_results, "Unseen"))

        # Generate Token Level Plots (Cumulative)
        run_token_diff_experiment(display_name, chunks, model, f"figure_unseen_{name}", model_name, n_plots=2, random_seed=42)

    print("\n--- AGGREGATE RESULTS ---")
    for r in results_score_data:
        print(f"[{r['Type']}] {r['Dataset']}: {r['Score']*100:.1f}%")

    safe_model = model_name.replace("/", "_")
    # 1. Contamination Score Box Plot (Figure 2 Style)
    plot_aggregate_scores(results_score_data, model_name, f"aggregate_scores_{safe_model}.png")

    # 2. Contamination Score Bar Plot (Per Dataset)
    plot_contamination_score_per_dataset(results_score_data, model_name, f"contamination_score_per_dataset_{safe_model}.png")

    # 3. Delta LogProb Box Plot
    #plot_aggregate_boxplots(all_raw_results, f"aggregate_boxplots_{safe_model}.png")

if __name__ == "__main__":
    main()
