"""Generate a print-ready ChArUco calibration board PDF at exact physical scale.

Correctness contract (a hand-drawn board is what burned us before): the board is
rendered by OpenCV's own ``cv2.aruco.CharucoBoard.generateImage()`` from a
``CharucoBoard`` built exactly the way ``paradex.image.aruco`` builds it for
detection (same dict, same marker IDs, same ``setLegacyPattern``). The rendered
raster is then fed back through that same board's ``CharucoDetector`` and we assert
every chessboard corner is recovered. If that self-check fails the PDF is **not**
written — so a PDF produced by this tool is, by construction, detectable.

Board geometry is read from a ``charuco_info.json``:
  * default: the live system config (``system/current/charuco_info.json``).
  * ``--config PATH``: a proposed config file, WITHOUT touching the live one. Use this
    to print new boards before you swap them in (the physical boards on the rig still
    match the live config until you replace them).

Page == board size (+ optional ``--margin-mm`` white quiet zone). The matching
``charuco_info.json`` entry (metric lengths) is printed for reference; no config
file is modified.

Usage:
    python src/calibration/generate_board.py                       # board 3, 50 mm sq
    python src/calibration/generate_board.py --board all --square-mm 50 --margin-mm 10
    python src/calibration/generate_board.py --config new.json --board all \
        --square-mm 50 --margin-mm 10 --out-dir outputs/new_boards
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
from cv2 import aruco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from paradex.image.aruco import aruco_dict, boardinfo_dict as SYSTEM_BOARDINFO


def build_board(info):
    """Build a CharucoBoard exactly like paradex.image.aruco does (relative units)."""
    board = aruco.CharucoBoard(
        (info["numX"], info["numY"]),
        info["checkerLength"], info["markerLength"],
        aruco_dict[info["dict_type"]],
        np.array(info["markerIDs"]),
    )
    board.setLegacyPattern(bool(info.get("setLegacyPattern")))
    return board


def n_corners(info):
    return (info["numX"] - 1) * (info["numY"] - 1)


def render_board(b_id, info, args, out_dir):
    board = build_board(info)
    numX, numY = info["numX"], info["numY"]
    ratio = info["markerLength"] / info["checkerLength"]
    marker_mm = args.square_mm * ratio

    px_per_mm = args.dpi / 25.4
    board_w_mm, board_h_mm = numX * args.square_mm, numY * args.square_mm
    w_px = int(round(board_w_mm * px_per_mm))
    h_px = int(round(board_h_mm * px_per_mm))
    margin_px = int(round(args.margin_mm * px_per_mm))
    img = board.generateImage((w_px + 2 * margin_px, h_px + 2 * margin_px),
                              marginSize=margin_px, borderBits=1)

    # --- self-check: this board's own detector must recover every corner ---
    detector = aruco.CharucoDetector(board)
    _, cids, _, _ = detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    got = 0 if cids is None else len(cids)
    exp = n_corners(info)
    if got != exp:
        raise SystemExit(
            f"SELF-CHECK FAILED for board {b_id!r}: recovered {got}/{exp} corners "
            f"(legacy={bool(info.get('setLegacyPattern'))}). Refusing to write.")
    print(f"  board {b_id!r}: self-check OK {got}/{exp} corners")

    page_w_mm = board_w_mm + 2 * args.margin_mm
    page_h_mm = board_h_mm + 2 * args.margin_mm
    out = os.path.join(out_dir,
                       f"charuco_{b_id}_{numX}x{numY}_{int(args.square_mm)}mm.pdf")

    fig = plt.figure(figsize=(page_w_mm / 25.4, page_h_mm / 25.4), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray", interpolation="nearest", aspect="auto", vmin=0, vmax=255)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print(f"    wrote {out}")
    print(f"    page {page_w_mm:.0f}x{page_h_mm:.0f} mm | square {args.square_mm:g} mm | "
          f"marker {marker_mm:.1f} mm (ratio {ratio:.3f}) | margin {args.margin_mm:g} mm")
    if page_w_mm > 297 or page_h_mm > 420:
        print(f"    note: exceeds A3 -> large-format printer/plotter.")
    return {b_id: {
        "numX": numX, "numY": numY,
        "checkerLength": round(args.square_mm / 1000.0, 6),
        "markerLength": round(marker_mm / 1000.0, 6),
        "dict_type": info["dict_type"],
        "numMarker": info["numMarker"],
        "markerIDs": info["markerIDs"],
        **({"setLegacyPattern": True} if info.get("setLegacyPattern") else {}),
    }}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="3", help="board id, or 'all' (default 3)")
    ap.add_argument("--config", default=None,
                    help="charuco_info.json to read (default: live system config)")
    ap.add_argument("--square-mm", type=float, default=50.0, help="checker square, mm")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin-mm", type=float, default=0.0, help="white quiet zone per side, mm")
    ap.add_argument("--out", default=None, help="single-board output PDF path")
    ap.add_argument("--out-dir", default=None, help="output dir (default: alongside this script)")
    args = ap.parse_args()

    boardinfo = json.load(open(args.config)) if args.config else SYSTEM_BOARDINFO
    src = args.config or "system/current/charuco_info.json"
    print(f"config: {src}")

    if args.board == "all":
        keys = list(boardinfo)
    else:
        if args.board not in boardinfo:
            raise SystemExit(f"board {args.board!r} not in config (have {list(boardinfo)})")
        keys = [args.board]

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"boards_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    entries = {}
    for b_id in keys:
        entries.update(render_board(b_id, boardinfo[b_id], args, out_dir))

    # single-file back-compat: if --out given and exactly one board, also copy there
    if args.out and len(keys) == 1:
        import shutil
        shutil.copy(os.path.join(out_dir, os.listdir(out_dir)[0]), args.out)

    print(f"\nall PDFs -> {out_dir}")
    print(f"  >>> PRINT AT 100% / ACTUAL SIZE (no 'fit to page'); one square = {args.square_mm:g} mm.")
    print("\nmatching charuco_info.json (metric lengths for this physical print):")
    print(json.dumps(entries, indent=4))


if __name__ == "__main__":
    main()
