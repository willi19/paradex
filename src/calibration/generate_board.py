"""Generate a print-ready ChArUco calibration board PDF at exact physical scale.

Correctness contract (a hand-drawn board is what burned us before): the board is
rendered by OpenCV's own ``cv2.aruco.CharucoBoard.generateImage()`` from the **same**
``CharucoBoard`` object ``paradex.image.aruco`` uses for detection (same dict, same
marker IDs, same ``setLegacyPattern``). The rendered raster is then fed back through
``detect_charuco()`` and we assert every chessboard corner is recovered. If that
self-check fails the PDF is **not** written — so a PDF produced by this tool is, by
construction, detectable by this repo's detector. No manual marker placement, ever.

Defaults reproduce board ``"3"`` (11x8, DICT_6X6_1000). Page == board size exactly
(no A4 letterboxing). The matching ``charuco_info.json`` entry (metric lengths) is
printed to stdout for reference; **no config file is modified**.

Usage:
    python src/calibration/generate_board.py                    # board 3, 50 mm squares
    python src/calibration/generate_board.py --square-mm 40 --dpi 600
    python src/calibration/generate_board.py --margin-mm 20     # add a white quiet zone
"""

import argparse
import os
import time
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paradex.image.aruco import (
    get_charuco_detector, _charuco_board_cache, boardinfo_dict, detect_charuco,
)


def n_corners_of_board(b_id):
    return _charuco_board_cache[b_id].getChessboardCorners().reshape(-1, 3).shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="3", help="board id from charuco_info.json (default 3)")
    ap.add_argument("--square-mm", type=float, default=50.0,
                    help="physical size of one checker square, mm (default 50)")
    ap.add_argument("--dpi", type=int, default=300, help="render resolution (default 300)")
    ap.add_argument("--margin-mm", type=float, default=0.0,
                    help="white quiet zone added on every side, mm (0 = page is exactly the board)")
    ap.add_argument("--out", default=None, help="output PDF path (default: alongside this script)")
    args = ap.parse_args()

    get_charuco_detector()  # builds/caches every board (incl. setLegacyPattern)
    b_id = args.board
    if b_id not in _charuco_board_cache:
        raise SystemExit(f"board {b_id!r} not in charuco_info.json (have {list(boardinfo_dict)})")

    info = boardinfo_dict[b_id]
    legacy = bool(info.get("setLegacyPattern"))      # source of truth: charuco_info.json
    board = _charuco_board_cache[b_id]               # exactly the object detect_charuco uses
    numX, numY = info["numX"], info["numY"]
    ratio = info["markerLength"] / info["checkerLength"]   # preserve the board's own proportions
    marker_mm = args.square_mm * ratio

    px_per_mm = args.dpi / 25.4
    board_w_mm, board_h_mm = numX * args.square_mm, numY * args.square_mm
    w_px = int(round(board_w_mm * px_per_mm))
    h_px = int(round(board_h_mm * px_per_mm))
    margin_px = int(round(args.margin_mm * px_per_mm))

    # OpenCV's own renderer, using the same board object detect_charuco uses.
    img = board.generateImage((w_px + 2 * margin_px, h_px + 2 * margin_px),
                              marginSize=margin_px, borderBits=1)

    # --- self-check: the repo's own detect_charuco must recover every corner ---
    n_expected = n_corners_of_board(b_id)
    det = detect_charuco(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    got = len(det.get(b_id, {}).get("checkerIDs", []))
    if got != n_expected:
        raise SystemExit(
            f"SELF-CHECK FAILED: detect_charuco recovered {got}/{n_expected} corners "
            f"for board {b_id!r} (legacy={legacy}). Refusing to write a board the detector can't read.")
    print(f"self-check OK: detect_charuco recovered {got}/{n_expected} corners on board {b_id!r} (legacy={legacy})")

    # --- exact-size PDF: page == board (+margin); one square measures exactly square-mm ---
    page_w_mm = board_w_mm + 2 * args.margin_mm
    page_h_mm = board_h_mm + 2 * args.margin_mm
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"charuco_board{b_id}_{numX}x{numY}_{int(args.square_mm)}mm_{ts}.pdf")

    fig = plt.figure(figsize=(page_w_mm / 25.4, page_h_mm / 25.4), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray", interpolation="nearest", aspect="auto", vmin=0, vmax=255)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print(f"\nwrote {out}")
    print(f"  page {page_w_mm:.1f} x {page_h_mm:.1f} mm | square {args.square_mm:g} mm | "
          f"marker {marker_mm:.2f} mm (ratio {ratio:.3f})")
    print(f"  >>> PRINT AT 100% / ACTUAL SIZE (no 'fit to page'), then measure one square = {args.square_mm:g} mm.")
    if page_w_mm > 297 or page_h_mm > 420:
        print(f"  note: {page_w_mm:.0f}x{page_h_mm:.0f} mm exceeds A3 — use a large-format printer/plotter or tile.")

    # --- matching charuco_info.json entry (metric), reference only; NOT written ---
    entry = {b_id: {
        "numX": numX, "numY": numY,
        "checkerLength": round(args.square_mm / 1000.0, 6),
        "markerLength": round(marker_mm / 1000.0, 6),
        "dict_type": info["dict_type"],
        "numMarker": info["numMarker"],
        "markerIDs": info["markerIDs"],
    }}
    if legacy:
        entry[b_id]["setLegacyPattern"] = True
    print("\ncharuco_info.json entry (metric lengths for this physical print; paste manually if you change size):")
    print(json.dumps(entry, indent=4))


if __name__ == "__main__":
    main()
