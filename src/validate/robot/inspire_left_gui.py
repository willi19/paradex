"""Interactive GUI for the Inspire LEFT hand.

Single window combining:
- 6 sliders to command each finger DOF (little, ring, middle, index,
  thumb_bend, thumb_rot) in real time.
- Force bar plot (6 channels).
- Tactile heatmap grid (17 regions: per-finger tip/nail/pad, thumb_middle, palm).

All three update concurrently with the hand. Close the window or Ctrl+C to exit.

Usage:
    python src/validate/robot/inspire_left_gui.py
"""
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

from paradex.utils.system import network_info
from paradex.io.robot_controller.inspire_controller_ip import (
    InspireControllerIP,
    TACTILE_LAYOUT,
)


DOF_LABELS = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
ANGLE_MIN, ANGLE_MAX = 0, 1000
HOME = np.full(6, 800, dtype=np.int32)
TACTILE_VMAX = 4095   # 12-bit sensor full scale
FORCE_VMAX = 2000
FORCE_FILTER_WINDOW = 5   # median over last N samples (spike-resistant)


def _ordered_tactile_names():
    # 5 rows × 4 cols = 20 slots; 17 regions → 3 spare. Order by finger.
    fingers = ("little", "ring", "middle", "index", "thumb")
    parts_by_finger = {
        "little": ("tip", "nail", "pad"),
        "ring":   ("tip", "nail", "pad"),
        "middle": ("tip", "nail", "pad"),
        "index":  ("tip", "nail", "pad"),
        "thumb":  ("tip", "nail", "middle", "pad"),
    }
    out = []
    for f in fingers:
        for p in parts_by_finger[f]:
            name = f"{f}_{p}"
            if name in TACTILE_LAYOUT:
                out.append(name)
    if "palm" in TACTILE_LAYOUT:
        out.append("palm")
    return out


def main():
    print(f"Connecting to Inspire LEFT at "
          f"{network_info['inspire']['ip']}:{network_info['inspire']['port']} ...")
    ic = InspireControllerIP(**network_info["inspire"], tactile=True)
    print("Connected.")

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(5, 8, figure=fig,
                  left=0.04, right=0.98, top=0.96, bottom=0.20,
                  wspace=0.35, hspace=0.45)

    # --- tactile heatmaps (left 5 cols × 4 rows = 20 cells, use 17) ---
    tactile_names = _ordered_tactile_names()
    heatmap_axes, heatmap_ims = {}, {}
    for idx, name in enumerate(tactile_names):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        rows, cols = TACTILE_LAYOUT[name][1], TACTILE_LAYOUT[name][2]
        shape = (cols, rows) if name == "palm" else (rows, cols)
        im = ax.imshow(np.zeros(shape), cmap="viridis",
                       vmin=0, vmax=TACTILE_VMAX, aspect="auto")
        ax.set_title(name, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        heatmap_axes[name] = ax
        heatmap_ims[name] = im

    # --- force bar plot (right side, top) ---
    # Bar height = median over last N samples (spike-resistant).
    # Text on top of each bar = instantaneous raw value.
    force_ax = fig.add_subplot(gs[0:3, 5:8])
    force_bars = force_ax.bar(DOF_LABELS, np.zeros(6), color="steelblue")
    force_ax.set_ylim(-FORCE_VMAX, FORCE_VMAX)
    force_ax.set_title(f"Force  (bar=median[{FORCE_FILTER_WINDOW}], text=raw)")
    force_ax.axhline(0, color="black", linewidth=0.5)
    force_ax.tick_params(axis="x", rotation=20)
    force_texts = [
        force_ax.text(i, 0, "0", ha="center", va="bottom", fontsize=8)
        for i in range(6)
    ]
    force_history = [deque(maxlen=FORCE_FILTER_WINDOW) for _ in range(6)]

    # --- status text (right side, bottom of tactile area) ---
    status_ax = fig.add_subplot(gs[3:5, 5:8])
    status_ax.axis("off")
    status_text = status_ax.text(0.02, 0.95, "", family="monospace",
                                 fontsize=10, va="top")

    # --- sliders (bottom of figure) ---
    slider_axes = []
    sliders = []
    for i, label in enumerate(DOF_LABELS):
        y = 0.14 - i * 0.022
        ax_s = fig.add_axes([0.10, y, 0.55, 0.018])
        s = Slider(ax_s, label, ANGLE_MIN, ANGLE_MAX,
                   valinit=int(HOME[i]), valstep=1, valfmt="%d")
        slider_axes.append(ax_s)
        sliders.append(s)

    target = HOME.copy()

    def push_target():
        ic.move(target.copy())

    def on_slider_change(_val):
        for i, s in enumerate(sliders):
            target[i] = int(s.val)
        push_target()

    for s in sliders:
        s.on_changed(on_slider_change)

    # --- buttons: HOME / OPEN / CLOSED ---
    def make_button(rect, label, preset):
        ax_b = fig.add_axes(rect)
        btn = Button(ax_b, label)
        def _on(_):
            for i, s in enumerate(sliders):
                s.set_val(int(preset[i]))   # triggers on_slider_change
        btn.on_clicked(_on)
        return btn

    btn_home   = make_button([0.70, 0.10, 0.07, 0.04], "HOME",
                             np.full(6, 800))
    btn_open   = make_button([0.78, 0.10, 0.07, 0.04], "OPEN",
                             np.full(6, 1000))
    btn_closed = make_button([0.86, 0.10, 0.07, 0.04], "CLOSED",
                             np.zeros(6))

    # --- force calibration button ---
    # Opens hand fully, triggers force-sensor zeroing, prints result.
    # Hand MUST not be in contact with anything during calibration.
    ax_cal = fig.add_axes([0.78, 0.04, 0.15, 0.04])
    btn_cal = Button(ax_cal, "CALIBRATE FORCE", color="lightcoral",
                     hovercolor="salmon")

    def _on_calibrate(_):
        print("[calibrate] opening hand and waiting 1.5s...")
        for i, s in enumerate(sliders):
            s.set_val(1000)
        time.sleep(1.5)
        print("[calibrate] triggering force zeroing — keep hand untouched.")
        try:
            result = ic.calibrate_force(timeout=15.0)
            print(f"[calibrate] done. forceClb register = {result}")
        except Exception as e:
            print(f"[calibrate] FAILED: {e}")

    btn_cal.on_clicked(_on_calibrate)

    # send initial target
    push_target()

    plt.ion()
    plt.show()

    print("GUI running. Close the window or Ctrl+C to exit.")
    try:
        last_print = 0.0
        while plt.fignum_exists(fig.number):
            loop_t0 = time.time()

            # force
            try:
                force = ic.get_force()
            except Exception as e:
                force = np.zeros(6)
                print(f"[warn] get_force failed: {e}")
            force_filtered = np.zeros(6, dtype=int)
            for i, val in enumerate(force):
                force_history[i].append(int(val))
                force_filtered[i] = int(np.median(force_history[i]))
                force_bars[i].set_height(force_filtered[i])
                # raw value text above bar (sign-aware placement)
                raw = int(val)
                force_texts[i].set_position(
                    (i, raw if raw >= 0 else raw - 50)
                )
                force_texts[i].set_text(str(raw))
                force_texts[i].set_va("bottom" if raw >= 0 else "top")

            # tactile
            try:
                tactile = ic.get_tactile()
                for name, im in heatmap_ims.items():
                    arr = tactile.get(name)
                    if arr is None:
                        continue
                    im.set_data(arr)
                    # auto-scale upper bound a bit for visibility
                    im.set_clim(0, max(TACTILE_VMAX // 4, int(arr.max())))
            except Exception as e:
                print(f"[warn] get_tactile failed: {e}")

            # status
            try:
                qpos = ic.get_qpos()
            except Exception:
                qpos = np.zeros(6, dtype=np.int32)
            lines = ["target  actual   force(med/raw)   label"]
            for i in range(6):
                lines.append(
                    f"{int(target[i]):>5d}  {int(qpos[i]):>5d}  "
                    f"{int(force_filtered[i]):>5d}/{int(force[i]):>5d}    "
                    f"{DOF_LABELS[i]}")
            status_text.set_text("\n".join(lines))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            dt = time.time() - loop_t0
            if time.time() - last_print > 2.0:
                print(f"viz loop: {dt*1000:.0f} ms ({1.0/max(dt,1e-3):.1f} Hz)")
                last_print = time.time()

            time.sleep(max(0, 0.05 - dt))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        ic.end()
        plt.close(fig)


if __name__ == "__main__":
    main()
