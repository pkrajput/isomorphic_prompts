
import argparse
import json
import numpy as np
from scipy.spatial.distance import jensenshannon, euclidean

def compute_energy(centroid_orig, centroid_iso):
    """
    Computes JSD and L2 distance between two opcode distributions (dicts).
    """
    if not centroid_orig or not centroid_iso:
        return None, None
        
    ops = set(centroid_orig.keys()) | set(centroid_iso.keys())
    sorted_ops = sorted(list(ops))
    
    vec_orig = np.array([float(centroid_orig.get(op, 0)) for op in sorted_ops])
    vec_iso = np.array([float(centroid_iso.get(op, 0)) for op in sorted_ops])
    
    # Normalize inputs just in case? Usually centroids sum to 1.
    s_orig = vec_orig.sum()
    if s_orig > 0: vec_orig /= s_orig
    
    s_iso = vec_iso.sum()
    if s_iso > 0: vec_iso /= s_iso
    
    # JSD
    # distance.jensenshannon returns distance (sqrt of divergence).
    # We want Divergence usually? "Centroid JSD energy ... E_jsd = JSD(P, Q)"
    # Usually implies JSD metric (0..1).
    # Let's return the square of the metric if we want proper divergence, OR just the metric.
    # "JSD (P_orig, P_iso)" typically refers to the value.
    # Scipy returns distance.
    d = jensenshannon(vec_orig, vec_iso, base=2)
    e_jsd = d ** 2 if not np.isnan(d) else 0.0
    
    # L2
    e_l2 = euclidean(vec_orig, vec_iso)
    
    return e_jsd, e_l2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_jsonl", required=True)
    parser.add_argument("--iso_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--metric_prefix", required=True, help="sctd or dctd")
    
    args = parser.parse_args()
    
    print(f"Loading {args.orig_jsonl}...")
    orig_data = {}
    with open(args.orig_jsonl, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
                orig_data[r["problem_id"]] = r
            except: pass
            
    print(f"Loading {args.iso_jsonl}...")
    iso_data = {}
    with open(args.iso_jsonl, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
                iso_data[r["problem_id"]] = r
            except: pass
            
    common_ids = sorted(list(set(orig_data.keys()) & set(iso_data.keys())))
    print(f"Found {len(common_ids)} common problems.")
    
    results = []
    
    for pid in common_ids:
        o = orig_data[pid]
        i = iso_data[pid]
        
        c_orig = o.get("opcode_centroid", {})
        c_iso = i.get("opcode_centroid", {})
        
        e_jsd, e_l2 = compute_energy(c_orig, c_iso)
        
        res = {
            "problem_id": pid,
            "dataset": o.get("dataset"),
            # Copy original metrics
            f"{args.metric_prefix}_jsd_orig": o.get(f"{args.metric_prefix}_jsd"),
            f"{args.metric_prefix}_tau_orig": o.get(f"{args.metric_prefix}_tau"),
            f"n_{args.metric_prefix}_orig": o.get("n_solutions_used"),
            
            # Copy iso metrics
            f"{args.metric_prefix}_jsd_iso": i.get(f"{args.metric_prefix}_jsd"),
            f"{args.metric_prefix}_tau_iso": i.get(f"{args.metric_prefix}_tau"),
            f"n_{args.metric_prefix}_iso": i.get("n_solutions_used"),
            
            # Energy
            f"{args.metric_prefix}_energy_jsd": e_jsd,
            f"{args.metric_prefix}_energy_l2": e_l2
        }
        results.append(res)
        
    with open(args.out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Written merged results to {args.out_jsonl}")

if __name__ == "__main__":
    main()
