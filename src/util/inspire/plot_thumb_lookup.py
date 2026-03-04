import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


THUMB_KEYS = ("thumb_2", "thumb_3", "thumb_4")


def is_numeric_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.basename(path).isdigit()


def load_lookup_entry(json_path: str, folder_name: str) -> Dict[str, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid lookup json: {json_path}")

    if folder_name in payload:
        entry = payload[folder_name]
    elif len(payload) == 1:
        entry = next(iter(payload.values()))
    else:
        raise KeyError(
            f"Could not resolve entry for folder '{folder_name}' in {json_path}. "
            f"Available keys: {list(payload.keys())}"
        )

    if not isinstance(entry, dict):
        raise ValueError(f"Invalid entry for folder '{folder_name}' in {json_path}")

    return entry


def collect_thumb_series(base_path: str, json_name: str) -> Tuple[List[int], Dict[str, List[float]]]:
    folder_names = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if is_numeric_dir(full_path):
            folder_names.append(name)
    folder_names.sort(key=int)

    xs: List[int] = []
    series = {key: [] for key in THUMB_KEYS}

    for folder_name in folder_names:
        json_path = os.path.join(base_path, folder_name, json_name)
        if not os.path.exists(json_path):
            continue

        entry = load_lookup_entry(json_path, folder_name)
        if not all(key in entry for key in THUMB_KEYS):
            missing = [key for key in THUMB_KEYS if key not in entry]
            raise KeyError(f"Missing keys {missing} in {json_path}")

        xs.append(int(folder_name))
        for key in THUMB_KEYS:
            series[key].append(float(entry[key]))

    if not xs:
        raise RuntimeError(
            f"No valid '{json_name}' files found under numeric subfolders of {base_path}"
        )

    return xs, series


def plot_thumb_series(xs: List[int], series: Dict[str, List[float]], output_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, key in zip(axes, THUMB_KEYS):
        ax.plot(xs, series[key], marker="o", linewidth=1.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        ax.set_title(key)

    axes[-1].set_xlabel("Episode Folder")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--json-name", type=str, default="lookup_thumb.json")
    parser.add_argument("--out-path", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out_path
    if out_path is None:
        out_path = os.path.join(args.base_path, "thumb_lookup_plot.png")

    xs, series = collect_thumb_series(args.base_path, args.json_name)
    plot_thumb_series(xs, series, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
