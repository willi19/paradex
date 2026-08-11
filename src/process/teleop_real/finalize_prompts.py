"""
Stage 2: turn the VLM batch verdicts into per-session prompts + a viewer index.

Reads the batch manifests/outputs under scratchpad/batches (written by the vision
agents), and for each analyzed (valid) session writes:
  prompt.json  {session, valid, in_basket, moved_object, confidence, note, prompt}
  prompt.txt   the instruction sentence (only when the object reached the basket)

Rules:
  - test folder  -> the object is always the pepsi can; a success prompt reads
                    "move pepsi to the pink basket".
  - test_object  -> "move {moved_object} to the pink basket" with the VLM's name.
  - no success (nothing landed in the basket) -> no prompt.txt, prompt=None.

Then writes a combined index JSON per folder for the viewer:
  <root>/<folder>/_index.json
"""
import os
import glob
import json

from paradex.utils.path import local_shared_dir

BATCHES = "/tmp/claude-1000/-home-jisoo-data2-Dexterous-Grasp-paradex/35f231d7-2295-420d-a851-adcc276f54d6/scratchpad/batches"
ROOT = local_shared_dir

_PEPSI_ALIASES = {"can", "pepsi can", "pepsi", "blue can", "navy can", "dark can",
                  "soda can", "cola can", "coke can", "pepsi-can", "black can",
                  "dark blue can", "blue soda can"}


def norm_object(folder, obj):
    o = (obj or "").strip().lower()
    if folder.rstrip("/").endswith("test"):
        return "pepsi"                      # test folder is the pepsi task
    if o in _PEPSI_ALIASES:
        return "pepsi"
    return o


def apply_verdicts():
    """Write prompt.json/prompt.txt from every batch output. Returns folders seen."""
    folders = set()
    n_prompt = 0
    n_fail = 0
    for man in sorted(glob.glob(os.path.join(BATCHES, "*_[0-9][0-9].json"))):
        if man.endswith("_out.json"):
            continue
        m = json.load(open(man))
        folder, out = m["folder"], m["output"]
        folders.add(folder)
        if not os.path.exists(out):
            print(f"[prompts] MISSING agent output: {os.path.basename(out)}")
            continue
        try:
            res = json.load(open(out))
        except Exception as e:
            print(f"[prompts] bad JSON {os.path.basename(out)}: {e}")
            continue
        for r in res:
            sess = r["session"]
            sdir = os.path.join(ROOT, folder, sess)
            if not os.path.isdir(sdir):
                continue
            obj = norm_object(folder, r.get("moved_object"))
            in_basket = bool(r.get("in_basket"))
            success = in_basket and obj not in ("none", "")
            prompt = f"move {obj} to the pink basket" if success else None
            pj = {
                "session": sess, "valid": True, "in_basket": in_basket,
                "moved_object": obj, "confidence": r.get("confidence"),
                "note": r.get("note", ""), "prompt": prompt, "source": "vlm-firstlast",
            }
            json.dump(pj, open(os.path.join(sdir, "prompt.json"), "w"), indent=2)
            if success:
                with open(os.path.join(sdir, "prompt.txt"), "w") as f:
                    f.write(prompt + "\n")
                n_prompt += 1
            else:
                # remove any stale prompt.txt from a previous run
                stale = os.path.join(sdir, "prompt.txt")
                if os.path.exists(stale):
                    os.remove(stale)
                n_fail += 1
    print(f"[prompts] wrote {n_prompt} success prompts, {n_fail} no-success (valid but not placed)")
    return folders


def dir_bytes(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def build_index(folder):
    base = os.path.join(ROOT, folder)
    sessions = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    entries = []
    tot_bytes = 0
    for sdir in sessions:
        name = os.path.basename(sdir)
        check = _load(os.path.join(sdir, "check.json"))
        prompt = _load(os.path.join(sdir, "prompt.json"))
        sz = dir_bytes(sdir)
        tot_bytes += sz
        entries.append({
            "session": name,
            "valid": bool(check.get("valid")) if check else None,
            "reason": check.get("reason", []) if check else [],
            "med_frames": check.get("med_frames") if check else None,
            "frame_counts": check.get("frame_counts", {}) if check else {},
            "n_videos": check.get("n_videos", 0) if check else 0,
            "arm_len": check.get("arm_len") if check else None,
            "prompt": prompt.get("prompt") if prompt else None,
            "moved_object": prompt.get("moved_object") if prompt else None,
            "in_basket": prompt.get("in_basket") if prompt else None,
            "confidence": prompt.get("confidence") if prompt else None,
            "bytes": sz,
            "composite": (f"{name}/_frames/26053260_firstlast.jpg"
                          if os.path.exists(os.path.join(sdir, "_frames", "26053260_firstlast.jpg"))
                          else None),
        })
    idx = {
        "folder": folder,
        "total_sessions": len(entries),
        "valid": sum(1 for e in entries if e["valid"]),
        "with_prompt": sum(1 for e in entries if e["prompt"]),
        "total_bytes": tot_bytes,
        "entries": entries,
    }
    out = os.path.join(base, "_index.json")
    json.dump(idx, open(out, "w"), indent=1)
    print(f"[index] {folder}: sessions={idx['total_sessions']} valid={idx['valid']} "
          f"prompts={idx['with_prompt']} size={tot_bytes/1e9:.1f}GB -> {out}")
    return idx


def _load(p):
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception:
        return None


if __name__ == "__main__":
    folders = apply_verdicts()
    for folder in ["capture/hri_vive/test", "capture/hri_vive/test_object"]:
        build_index(folder)
