#!/usr/bin/env python3
"""
prompt_fuzzing.py  --  Fuzz natural-language prompts in coding benchmarks.

Two perturbation modes, both standard in the robustness / memorization
literature (cf. ReCode, Wang et al. ACL 2023):

  1.  synonym   -- replace content words with WordNet synonyms at
                   probability --fuzz_level (float 0-1).
  2.  deadcode  -- insert dead Python code blocks into the prompt.

Usage
-----
  python prompt_fuzzing.py --dataset bigobench --mode synonym --fuzz_level 0.2
  python prompt_fuzzing.py --dataset mbpp --mode deadcode --num_blocks 3

Output goes to  datasets/<name>_<mode>_<tag>/
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

# ── Dataset configuration ────────────────────────────────────────────
DATASET_CONFIG = {
    "bigobench": {
        "src_dir": "BigOBench",
        "prompt_field": "description",
        "problem_id_field": "problem_name",
    },
    "effibench": {
        "src_dir": "Effibench",
        "prompt_field": "prompt",
        "problem_id_field": "id",
    },
    "mbpp": {
        "src_dir": "mbpp",
        "prompt_field": "prompt",
        "problem_id_field": "task_id",
    },
}

# ── Stop / skip sets ─────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

CODE_MARKERS = {"`", "``", "```", "\\", "$", "@", "#", "_", "==", "!=",
                ">=", "<=", ">>", "<<", "&&", "||", "->", "::", "//"}

def _is_code_token(tok):
    if any(c in tok for c in CODE_MARKERS):
        return True
    if "_" in tok:
        return True
    if tok.startswith("0x") or tok.startswith("0b"):
        return True
    if re.match(r"^[A-Z][a-z]+[A-Z]", tok):
        return True
    if re.match(r"^[a-z]+[A-Z]", tok):
        return True
    return False

# ── WordNet POS mapping ──────────────────────────────────────────────
_PENN_TO_WN = {"J": wordnet.ADJ, "V": wordnet.VERB,
               "N": wordnet.NOUN, "R": wordnet.ADV}

def _penn_to_wn(tag):
    return _PENN_TO_WN.get(tag[0])

# ── Synonym fuzzing ──────────────────────────────────────────────────

def _get_synonym(word, pos):
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    candidates = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                candidates.add(name)
    return random.choice(sorted(candidates)) if candidates else None


def fuzz_synonym(text, level, rng):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    out = []
    n_replaced = 0
    for tok, tag in tagged:
        if (len(tok) <= 2
                or tok.lower() in STOP_WORDS
                or not tok.isalpha()
                or _is_code_token(tok)):
            out.append(tok)
            continue
        if rng.random() >= level:
            out.append(tok)
            continue
        wn_pos = _penn_to_wn(tag)
        syn = _get_synonym(tok, wn_pos)
        if syn is None:
            out.append(tok)
        else:
            if tok[0].isupper():
                syn = syn.capitalize()
            out.append(syn)
            n_replaced += 1
    return _detokenize(out), n_replaced


def _detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"` +", "`", text)
    text = re.sub(r" +`", "`", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ── Dead-code insertion ──────────────────────────────────────────────

DEAD_CODE_BLOCKS = [
    '# [debug] temporary variable for edge-case logging\n_dbg_sentinel = None',
    'if False:\n    _unreachable = 42\n    print("this never runs")',
    'def _temp_helper_unused():\n    """Placeholder -- not called."""\n    return []',
    '_MAGIC_OFFSET = 0xDEAD  # legacy constant, safe to ignore',
    'for _i in range(0):\n    pass  # no-op loop',
    '# NOTE: retained for backward compatibility\ntry:\n    _ = None\nexcept Exception:\n    pass',
    '_chk = lambda *a, **k: None  # no-op checker stub',
    '# TODO(reviewer): verify this block is dead code\nclass _Unused:\n    pass',
    'import sys as _sys_unused  # unused import kept for compat',
    '_PADDING = [0] * 16  # pre-allocated buffer (unused)',
]


def insert_dead_code(text, num_blocks, rng):
    chosen = rng.sample(DEAD_CODE_BLOCKS, min(num_blocks, len(DEAD_CODE_BLOCKS)))
    block = "\n\n".join(chosen)
    sep = ("\n\n# ----- [context: supporting code / utilities] -----\n"
           "# The following helper code may or may not be relevant.\n\n")
    return text + sep + block

# ── Main processing ──────────────────────────────────────────────────

def process_dataset(dataset, mode, fuzz_level, num_blocks, seed, datasets_root):
    cfg = DATASET_CONFIG[dataset]
    src_dir = datasets_root / cfg["src_dir"]

    if mode == "synonym":
        tag = str(int(fuzz_level * 100)).zfill(2)
        out_name = f"{dataset}_synonym_{tag}"
    else:
        out_name = f"{dataset}_deadcode"

    out_dir = datasets_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.jsonl"))
    if not src_files:
        print(f"[ERROR] No .jsonl files in {src_dir}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    total_rows = 0
    total_replacements = 0

    for src_file in src_files:
        out_file = out_dir / src_file.name
        print(f"  {src_file.name} -> {out_file}")
        with open(src_file) as fin, open(out_file, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get(cfg["prompt_field"], "")

                if mode == "synonym":
                    fuzzed, n_rep = fuzz_synonym(prompt, fuzz_level, rng)
                    total_replacements += n_rep
                else:
                    fuzzed = insert_dead_code(prompt, num_blocks, rng)

                new_row = dict(row)
                new_row[cfg["prompt_field"]] = fuzzed
                new_row["_fuzz_method"] = mode
                new_row["_fuzz_level"] = fuzz_level if mode == "synonym" else num_blocks
                new_row["_fuzz_seed"] = seed
                fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                total_rows += 1

    print(f"  Wrote {total_rows} rows to {out_dir}/")
    if mode == "synonym":
        avg = total_replacements / max(total_rows, 1)
        print(f"  Average replacements per prompt: {avg:.1f}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="Fuzz coding-benchmark prompts (synonym / deadcode).")
    ap.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    ap.add_argument("--mode", choices=["synonym", "deadcode"], required=True)
    ap.add_argument("--fuzz_level", type=float, default=0.2,
                    help="P(replace) for synonym mode.  Default 0.2.")
    ap.add_argument("--num_blocks", type=int, default=3,
                    help="Dead-code blocks to insert.  Default 3.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets_root", type=str, default=None)
    args = ap.parse_args()

    if args.datasets_root:
        datasets_root = Path(args.datasets_root)
    else:
        datasets_root = Path(__file__).resolve().parent.parent / "datasets"

    if not datasets_root.is_dir():
        print(f"[ERROR] datasets root not found: {datasets_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {args.mode}  |  Dataset: {args.dataset}  |  "
          f"Level/blocks: {args.fuzz_level if args.mode == 'synonym' else args.num_blocks}  |  "
          f"Seed: {args.seed}")
    print(f"Datasets root: {datasets_root}")
    process_dataset(args.dataset, args.mode, args.fuzz_level,
                    args.num_blocks, args.seed, datasets_root)
    print("Done.")


if __name__ == "__main__":
    main()
