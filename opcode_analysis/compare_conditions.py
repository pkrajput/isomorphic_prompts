"""
compare_conditions.py -- JSD + L2 between orig/iso opcode centroids per problem.
"""

import argparse
import json

import numpy as np
from scipy.spatial.distance import euclidean, jensenshannon


def centroid_divergence(centroid_orig, centroid_iso):
    """JSD and L2 between two opcode distributions (dicts)."""
    if not centroid_orig or not centroid_iso:
        return None, None

    ops = sorted(set(centroid_orig) | set(centroid_iso))
    p = np.array([float(centroid_orig.get(op, 0)) for op in ops])
    q = np.array([float(centroid_iso.get(op, 0)) for op in ops])

    if p.sum() > 0:
        p /= p.sum()
    if q.sum() > 0:
        q /= q.sum()

    d = jensenshannon(p, q, base=2)
    jsd = d ** 2 if not np.isnan(d) else 0.0
    l2 = euclidean(p, q)
    return jsd, l2


def main():
    parser = argparse.ArgumentParser(
        description="Compare opcode centroids between original and ISO conditions.")
    parser.add_argument("--orig_jsonl", required=True)
    parser.add_argument("--iso_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    orig, iso = {}, {}
    for path, store in [(args.orig_jsonl, orig), (args.iso_jsonl, iso)]:
        print(f"Loading {path}...")
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    store[r["problem_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass

    common = sorted(set(orig) & set(iso))
    print(f"{len(common)} common problems.")

    results = []
    for pid in common:
        o, i = orig[pid], iso[pid]
        jsd, l2 = centroid_divergence(
            o.get("opcode_centroid", {}), i.get("opcode_centroid", {}))
        results.append({
            "problem_id": pid,
            "dataset": o.get("dataset"),
            "jsd_orig": o.get("jsd"),
            "jsd_iso": i.get("jsd"),
            "n_orig": o.get("n_solutions"),
            "n_iso": i.get("n_solutions"),
            "cross_jsd": jsd,
            "cross_l2": l2,
        })

    with open(args.out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Written {len(results)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
