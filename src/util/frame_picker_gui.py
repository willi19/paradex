"""
Frame picker GUI.

Walks <root>/<object>/<seq>/videos/<cam_id>.avi, lets the user scrub each
video and save one frame number per (object, seq). Results persist to JSON
so the session can be resumed.

Usage:
    python src/util/frame_picker_gui.py \
        --root /home/temp_id/shared_data/capture/eccv2026/allegro_v5 \
        --cam-id 22684755 \
        --out /home/temp_id/shared_data/capture/eccv2026/allegro_v5/picked_frames.json

Keys:
    ←/→         step 1 frame
    Shift+←/→   step 10 frames
    Ctrl+←/→   step 100 frames
    Space       save current frame, advance to next clip
    s           save without advancing
    n / p       next / previous clip
    Delete      clear saved frame for current clip
    q           quit
"""

import argparse
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


def discover_clips(root: Path, cam_id: str):
    clips = []
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seq_dir in sorted(
            (p for p in obj_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            video = seq_dir / "videos" / f"{cam_id}.avi"
            grasp = seq_dir / "grasp_result.json"
            paired = seq_dir / "paired_human_episode.json"
            if not (video.exists() and grasp.exists() and paired.exists()):
                continue
            try:
                if not json.loads(grasp.read_text()).get("grasp_success"):
                    continue
                paired_info = json.loads(paired.read_text())
            except Exception:
                continue
            clips.append((obj_dir.name, seq_dir.name, video, paired_info))
    return clips


class FramePicker:
    def __init__(self, root_tk, clips, out_path: Path, display_width=480):
        self.root = root_tk
        self.clips = clips
        self.out_path = out_path
        self.display_width = display_width

        self.results = {}
        if out_path.exists():
            self.results = json.loads(out_path.read_text())

        self.idx = self._first_unsaved_index()
        self.frames = []
        self.total_frames = 0
        self.cur_frame = 0
        self._slider_user = True

        self._build_ui()
        self._load_clip(self.idx)
        self._bind_keys()

    @staticmethod
    def _saved_frame(entry):
        if entry is None:
            return None
        if isinstance(entry, int):
            return entry
        return entry.get("frame")

    def _first_unsaved_index(self):
        for i, (obj, seq, _v, _p) in enumerate(self.clips):
            if self._saved_frame(self.results.get(obj, {}).get(seq)) is None:
                return i
        return 0

    def _build_ui(self):
        self.root.title("Frame Picker")
        self.root.geometry("1100x800")

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)
        self.info_lbl = ttk.Label(top, text="", font=("TkDefaultFont", 11, "bold"))
        self.info_lbl.pack(side=tk.LEFT)
        self.progress_lbl = ttk.Label(top, text="")
        self.progress_lbl.pack(side=tk.RIGHT)

        self.canvas = tk.Label(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8)

        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        self.frame_lbl = ttk.Label(ctrl, text="", width=22)
        self.frame_lbl.pack(side=tk.LEFT)
        self.saved_lbl = ttk.Label(ctrl, text="", foreground="green")
        self.saved_lbl.pack(side=tk.LEFT, padx=12)

        self.slider = ttk.Scale(self.root, from_=0, to=1, orient=tk.HORIZONTAL,
                                command=self._on_slider)
        self.slider.pack(fill=tk.X, padx=8, pady=4)

        btns = ttk.Frame(self.root)
        btns.pack(fill=tk.X, padx=8, pady=6)

        def mk(parent, text, cmd, **pack):
            b = ttk.Button(parent, text=text, command=cmd, takefocus=False)
            b.pack(**pack)
            return b

        mk(btns, "◀ Prev clip (p)", self._prev_clip, side=tk.LEFT)
        mk(btns, "−10", lambda: self._step(-10), side=tk.LEFT, padx=4)
        mk(btns, "−1",  lambda: self._step(-1),  side=tk.LEFT)
        mk(btns, "+1",  lambda: self._step(1),   side=tk.LEFT, padx=4)
        mk(btns, "+10", lambda: self._step(10),  side=tk.LEFT)
        mk(btns, "Save (s)", self._save_current, side=tk.LEFT, padx=12)
        mk(btns, "Save & Next (Space)", self._save_and_next, side=tk.LEFT)
        mk(btns, "Clear (Del)", self._clear_current, side=tk.LEFT, padx=12)
        mk(btns, "Next clip ▶ (n)", self._next_clip, side=tk.RIGHT)
        self.slider.configure(takefocus=False)

    def _bind_keys(self):
        b = self.root.bind
        b("<Left>",        lambda e: self._step(-1))
        b("<Right>",       lambda e: self._step(1))
        b("<Shift-Left>",  lambda e: self._step(-10))
        b("<Shift-Right>", lambda e: self._step(10))
        b("<Control-Left>",  lambda e: self._step(-100))
        b("<Control-Right>", lambda e: self._step(100))
        b("<space>", lambda e: self._save_and_next())
        b("s",       lambda e: self._save_current())
        b("n",       lambda e: self._next_clip())
        b("p",       lambda e: self._prev_clip())
        b("<Delete>", lambda e: self._clear_current())
        b("q",       lambda e: self.root.destroy())

    def _load_clip(self, idx):
        if not (0 <= idx < len(self.clips)):
            messagebox.showinfo("Done", "No more clips.")
            return
        self.idx = idx
        obj, seq, path, _paired = self.clips[idx]
        self.info_lbl.config(text=f"{obj} / {seq}   loading...")
        self.root.update_idletasks()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open {path}")
            return
        self.frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > self.display_width:
                s = self.display_width / w
                frame = cv2.resize(frame, (self.display_width, int(h * s)))
            self.frames.append(frame)
        cap.release()
        self.total_frames = len(self.frames)
        if self.total_frames == 0:
            messagebox.showerror("Error", f"Empty video {path}")
            return

        self._slider_user = False
        self.slider.config(from_=0, to=self.total_frames - 1)
        self._slider_user = True

        saved = self._saved_frame(self.results.get(obj, {}).get(seq))
        start = saved if saved is not None else 0
        self.info_lbl.config(text=f"{obj} / {seq}   ({path.name})")
        self.progress_lbl.config(
            text=f"clip {idx + 1} / {len(self.clips)}   saved: {self._saved_count()} / {len(self.clips)}"
        )
        self._seek(start)

    def _saved_count(self):
        return sum(
            1 for o, s, _v, _p in self.clips
            if self._saved_frame(self.results.get(o, {}).get(s)) is not None
        )

    def _seek(self, frame_no):
        frame_no = int(frame_no) % self.total_frames
        self.cur_frame = frame_no
        obj, seq, _v, _p = self.clips[self.idx]
        saved = self._saved_frame(self.results.get(obj, {}).get(seq))
        self._show(self.frames[frame_no].copy(), frame_no, saved)
        self._slider_user = False
        self.slider.set(frame_no)
        self._slider_user = True
        self.frame_lbl.config(text=f"frame {frame_no} / {self.total_frames - 1}")
        self.saved_lbl.config(text=f"saved: {saved}" if saved is not None else "")

    def _show(self, frame_bgr, frame_no=None, saved=None):
        if frame_no is not None:
            txt = f"frame {frame_no} / {self.total_frames - 1}"
            if saved is not None:
                txt += f"   [saved: {saved}]"
            cv2.putText(frame_bgr, txt, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame_bgr, txt, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=img)
        self.canvas.image = img

    def _on_slider(self, val):
        if not self._slider_user:
            return
        self._seek(float(val))

    def _step(self, delta):
        self._seek(self.cur_frame + delta)

    def _save_current(self):
        obj, seq, _v, paired = self.clips[self.idx]
        entry = {
            "frame": self.cur_frame,
            "paired_human_episode": (paired or {}).get("paired human episode"),
        }
        self.results.setdefault(obj, {})[seq] = entry
        self._persist()
        self.saved_lbl.config(text=f"saved: {self.cur_frame}")
        self.progress_lbl.config(
            text=f"clip {self.idx + 1} / {len(self.clips)}   saved: {self._saved_count()} / {len(self.clips)}"
        )

    def _save_and_next(self):
        self._save_current()
        self._next_clip()

    def _clear_current(self):
        obj, seq, _v, _p = self.clips[self.idx]
        if obj in self.results and seq in self.results[obj]:
            del self.results[obj][seq]
            if not self.results[obj]:
                del self.results[obj]
            self._persist()
        self.saved_lbl.config(text="")

    def _next_clip(self):
        if self.idx + 1 < len(self.clips):
            self._load_clip(self.idx + 1)
        else:
            messagebox.showinfo("Done", "Reached last clip.")

    def _prev_clip(self):
        if self.idx > 0:
            self._load_clip(self.idx - 1)

    def _persist(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.results, indent=2, sort_keys=True))
        os.replace(tmp, self.out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--cam-id", default="22684755")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    clips = discover_clips(args.root, args.cam_id)
    if not clips:
        print(f"No clips found under {args.root} for cam {args.cam_id}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(clips)} clips.")

    root = tk.Tk()
    FramePicker(root, clips, args.out)
    root.mainloop()


if __name__ == "__main__":
    main()
