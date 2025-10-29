# inspect_checkpoint.py
import os
import torch
import collections

CHK = r"D:\Project\ECG\ECGFounder\checkpoint\12_lead_ECGFounder.pth" 

def summarize_value(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return f"Tensor, shape={tuple(v.size())}, dtype={v.dtype}"
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)}"
    if isinstance(v, dict):
        return f"dict len={len(v)}"
    return f"{type(v).__name__}"

def dump_keys(d, prefix=""):
    lines = []
    if isinstance(d, dict) or isinstance(d, collections.abc.Mapping):
        for i, (k, v) in enumerate(d.items(), 1):
            lines.append(f"{prefix}{i}. {k} -> {summarize_value(v)}")
    else:
        lines.append(f"{prefix}Object of type {type(d).__name__}")
    return lines

def main(path):
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return

    print(f"Loading checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        print("Failed to load checkpoint:", e)
        return

    print("\nTop-level type:", type(checkpoint).__name__)
    if isinstance(checkpoint, dict):
        print("Top-level keys and summary:")
        top_lines = dump_keys(checkpoint)
        for ln in top_lines:
            print(" ", ln)

        # If there's a state_dict inside, inspect it too
        for candidate in ("state_dict", "model_state_dict", "net", "model"):
            if candidate in checkpoint:
                print(f"\nFound sub-key '{candidate}' — inspecting its keys (first 200):")
                sd = checkpoint[candidate]
                # sd may be OrderedDict or dict
                if isinstance(sd, (dict, collections.abc.Mapping)):
                    keys = list(sd.keys())
                    print(f"  number of keys: {len(keys)}")
                    for i, k in enumerate(keys[:200], 1):
                        v = sd[k]
                        print(f"   {i:3}. {k} -> {summarize_value(v)}")
                else:
                    print("  sub-object is not a dict/mapping, type:", type(sd))
                break
        else:
            # no typical state_dict found — maybe checkpoint is itself the state dict
            print("\nNo typical 'state_dict' key found — maybe the top-level is the state_dict.")
            if all(isinstance(k, str) for k in checkpoint.keys()):
                keys = list(checkpoint.keys())
                print(f"  number of top-level keys: {len(keys)}")
                for i, k in enumerate(keys[:200], 1):
                    v = checkpoint[k]
                    print(f"   {i:3}. {k} -> {summarize_value(v)}")
    else:
        print("Checkpoint is not a dict; it's a", type(checkpoint))

    # Helpful: detect module. prefix (DataParallel)
    def has_module_prefix(mapping):
        if not isinstance(mapping, dict):
            return False
        for k in mapping.keys():
            if k.startswith("module."):
                return True
        return False

    # check state_dict candidate used above
    candidate_sd = None
    for candidate in ("state_dict", "model_state_dict", "net", "model"):
        if isinstance(checkpoint, dict) and candidate in checkpoint:
            candidate_sd = checkpoint[candidate]
            break
    if candidate_sd is None and isinstance(checkpoint, dict):
        candidate_sd = checkpoint  # assume top-level is state dict

    if candidate_sd is not None and isinstance(candidate_sd, dict):
        if has_module_prefix(candidate_sd):
            print("\nNOTE: keys appear to have 'module.' prefix (DataParallel).")
            # show first 10 keys before/after stripping
            keys = list(candidate_sd.keys())
            print("  sample keys (first 10):")
            for k in keys[:10]:
                print("   ", k)
            print("\n  sample keys after stripping 'module.':")
            for k in keys[:10]:
                nk = k[len("module."):] if k.startswith("module.") else k
                print("   ", nk)

    # Optionally write results to file for later review
    outtxt = "checkpoint_inspect.txt"
    with open(outtxt, "w", encoding="utf-8") as f:
        f.write(f"Inspected: {path}\n\n")
        f.write("Top-level type: %s\n\n" % type(checkpoint).__name__)
        if isinstance(checkpoint, dict):
            for ln in top_lines:
                f.write(ln + "\n")
            f.write("\n\n")
            if candidate_sd is not None and isinstance(candidate_sd, dict):
                keys = list(candidate_sd.keys())
                f.write(f"state-dict keys count: {len(keys)}\n")
                for i, k in enumerate(keys, 1):
                    f.write(f"{i:4}. {k} -> {summarize_value(candidate_sd[k])}\n")
        else:
            f.write("Checkpoint is not a dict.\n")
    print(f"\nWrote summary to ./{outtxt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", "-p", default=CHK, help="checkpoint file path")
    args = ap.parse_args()
    main(args.path)