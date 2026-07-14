import json
import os
import time
from collections import deque
from threading import Event, Lock, Thread

import numpy as np


class HumanTactileRecorder:
    def __init__(
        self,
        port,
        baud_rate,
        timeout=0.05,
        reset_wait=2.0,
        plot_realtime=False,
        plot_refresh_interval=0.02,
        plot_max_samples=200,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.reset_wait = reset_wait
        self.plot_realtime = plot_realtime
        self.plot_refresh_interval = plot_refresh_interval
        self.plot_max_samples = plot_max_samples

        self.ser = None
        self.reader_thread = None
        self.plot_thread = None
        self.exit_event = Event()
        self.save_event = Event()
        self.plot_stop_event = Event()
        self.latest_lock = Lock()
        self.save_lock = Lock()

        self.save_path = None
        self.data = None
        self.invalid_line_count = 0
        self.latest_timestamp = None
        self.latest_sample = None
        self.latest_sample_id = 0

        self.plot_enabled = False
        self.plt = None
        self.fig = None
        self.ax = None
        self.lines = None
        self.plot_sample_ids = deque(maxlen=plot_max_samples)
        self.plot_history = [
            deque(maxlen=plot_max_samples),
            deque(maxlen=plot_max_samples),
        ]

    def connect(self):
        if self.reader_thread is not None:
            if self.reader_thread.is_alive():
                return
            self.reader_thread = None
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for --human_tactile.") from exc

        self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        if self.reset_wait > 0:
            time.sleep(self.reset_wait)

        self.reader_thread = Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        print(f"Human tactile serial connected: {self.port} @ {self.baud_rate}")

        if self.plot_realtime:
            self.start_plot()

    def start(self, save_path):
        self.connect()
        os.makedirs(save_path, exist_ok=True)
        with self.save_lock:
            self.save_path = save_path
            self.data = {
                "time": [],
                "tactile": [],
            }
            self.invalid_line_count = 0
            self.save_event.set()

    def stop(self):
        with self.save_lock:
            self.save_event.clear()
            if self.save_path is None or self.data is None:
                return 0

            save_path = self.save_path
            times = list(self.data["time"])
            tactile = list(self.data["tactile"])
            invalid_line_count = self.invalid_line_count

            self.save_path = None
            self.data = None
            self.invalid_line_count = 0

        time_arr = np.asarray(times, dtype=np.float64)
        tactile_arr = np.asarray(tactile, dtype=np.int32)
        if tactile_arr.size == 0:
            tactile_arr = tactile_arr.reshape(0, 2)

        np.save(os.path.join(save_path, "time.npy"), time_arr)
        np.save(os.path.join(save_path, "tactile.npy"), tactile_arr)

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0
            sample_rate_hz = float((len(time_arr) - 1) / duration) if duration > 0.0 else 0.0
            json.dump(
                {
                    "port": self.port,
                    "baud_rate": self.baud_rate,
                    "num_samples": int(len(time_arr)),
                    "duration": duration,
                    "sample_rate_hz": sample_rate_hz,
                    "invalid_line_count": int(invalid_line_count),
                },
                f,
                indent=2,
            )

        print(f"Saved human tactile samples: {len(time_arr)}")
        return len(time_arr)

    def close(self):
        if self.save_event.is_set():
            self.stop()

        self.exit_event.set()
        self.stop_plot()
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=max(1.0, self.timeout + 0.5))
            self.reader_thread = None
        self.ser = None

    def get_latest(self):
        with self.latest_lock:
            if self.latest_sample is None:
                return None
            return self.latest_sample_id, self.latest_timestamp, self.latest_sample.copy()

    def start_plot(self):
        if self.plot_thread is not None and self.plot_thread.is_alive():
            return

        if self.plt is None:
            try:
                import matplotlib.pyplot as plt

                self.plt = plt
                self.plt.ion()
                self.plot_enabled = True
            except Exception as exc:
                self.plot_enabled = False
                print(f"Failed to initialize human tactile realtime plot: {exc}")
                return

        self.plot_stop_event.clear()
        self.plot_thread = Thread(target=self._plot_loop, daemon=True)
        self.plot_thread.start()

    def stop_plot(self):
        self.plot_stop_event.set()
        if self.plot_thread is not None:
            self.plot_thread.join(timeout=1.0)
            self.plot_thread = None

        if self.fig is not None and self.plt is not None:
            self.plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.lines = None

    def _read_loop(self):
        while not self.exit_event.is_set():
            try:
                raw_line = self.ser.readline()
            except Exception as exc:
                if not self.exit_event.is_set():
                    print(f"Human tactile serial read failed: {exc}")
                break

            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                first, second = line.split(",")
                sample = [int(first), int(second)]
            except ValueError:
                with self.save_lock:
                    if self.save_event.is_set():
                        self.invalid_line_count += 1
                continue

            timestamp = time.time()
            sample_arr = np.asarray(sample, dtype=np.int32)
            with self.latest_lock:
                self.latest_timestamp = timestamp
                self.latest_sample = sample_arr
                self.latest_sample_id += 1

            if self.save_event.is_set():
                with self.save_lock:
                    if self.save_event.is_set() and self.data is not None:
                        self.data["time"].append(timestamp)
                        self.data["tactile"].append(sample)

    def _ensure_plot(self):
        if self.fig is not None:
            return

        self.fig, raw_axes = self.plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self.ax = list(raw_axes)
        self.fig.suptitle("Human Tactile Realtime")
        self.lines = [
            self.ax[0].plot([], [], linewidth=1.5, label="value1")[0],
            self.ax[1].plot([], [], linewidth=1.5, label="value2")[0],
        ]
        for ax, label in zip(self.ax, ("value1", "value2")):
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        self.ax[-1].set_xlabel("Samples")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_plot(self, sample_id, sample):
        self._ensure_plot()

        self.plot_sample_ids.append(sample_id)
        self.plot_history[0].append(int(sample[0]))
        self.plot_history[1].append(int(sample[1]))
        x_vals = list(self.plot_sample_ids)

        for ax, line, y_vals in zip(self.ax, self.lines, self.plot_history):
            line.set_data(x_vals, list(y_vals))
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _plot_loop(self):
        last_sample_id = 0

        try:
            while not self.plot_stop_event.is_set() and not self.exit_event.is_set():
                if not self.plot_enabled:
                    break

                latest = self.get_latest()
                if latest is None:
                    time.sleep(self.plot_refresh_interval)
                    continue

                sample_id, _, sample = latest
                if sample_id != last_sample_id:
                    self._update_plot(sample_id, sample)
                    last_sample_id = sample_id

                time.sleep(self.plot_refresh_interval)
        except Exception as exc:
            self.plot_enabled = False
            print(f"Disabling human tactile realtime plot after update failure: {exc}")
