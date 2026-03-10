
import time
from collections import deque
from threading import Event, Lock, Thread

# import chime
# chime.theme('pokemon')


class InspireF1RealtimeTactilePlotter:
    def __init__(self, hand_controller, max_samples=200, refresh_interval=0.05):
        self.hand_controller = hand_controller
        self.max_samples = max_samples
        self.refresh_interval = refresh_interval
        self.sites = []
        self.history = {}
        self.sample_ids = deque(maxlen=max_samples)
        self.sample_count = 0
        self.enabled = False
        self.fig = None
        self.axes = None
        self.lines = {}
        self.site_row_index = {}
        self.stop_event = Event()
        self.plot_lock = Lock()
        self.worker = None

        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            self.plt.ion()
            self.enabled = True
        except Exception as exc:
            self.plt = None
            print(f"Failed to initialize matplotlib realtime tactile plot: {exc}")

    def _extract_sites(self, tactile):
        suffixes = (
            "_normal_force",
            "_tangential_force",
            "_tangential_direction",
        )
        sites = set()
        for key in tactile:
            for suffix in suffixes:
                if key.endswith(suffix):
                    sites.add(key[: -len(suffix)])
                    break
        return sorted(sites)

    def _ensure_figure(self, sites):
        if not self.enabled or self.fig is not None:
            return

        self.sites = sites
        metrics = (
            ("normal_force", "Normal Force"),
            ("tangential_force", "Tangential Force"),
            ("tangential_direction", "Direction"),
        )
        n_rows = len(self.sites)
        figsize = (12, max(2.5 * n_rows, 4))
        self.fig, raw_axes = self.plt.subplots(n_rows, len(metrics), figsize=figsize, squeeze=False)
        self.axes = raw_axes
        self.fig.suptitle("Inspire F1 Tactile Realtime")

        for row_idx, site in enumerate(self.sites):
            self.site_row_index[site] = row_idx
            self.history[site] = {
                "normal_force": deque(maxlen=self.max_samples),
                "tangential_force": deque(maxlen=self.max_samples),
                "tangential_direction": deque(maxlen=self.max_samples),
            }
            self.lines[site] = {}
            for col_idx, (metric_key, metric_label) in enumerate(metrics):
                ax = self.axes[row_idx][col_idx]
                (line,) = ax.plot([], [], linewidth=1.5)
                ax.set_title(f"{site} {metric_label}")
                ax.grid(True, alpha=0.3)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Samples")
                self.lines[site][metric_key] = line

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_plot(self, tactile):
        if self.fig is None:
            sites = self._extract_sites(tactile)
            if not sites:
                return
            self._ensure_figure(sites)

        self.sample_count += 1
        self.sample_ids.append(self.sample_count)
        x_vals = list(self.sample_ids)

        for site in self.sites:
            row_idx = self.site_row_index[site]
            site_history = self.history[site]
            site_history["normal_force"].append(float(tactile.get(f"{site}_normal_force", 0.0)))
            site_history["tangential_force"].append(float(tactile.get(f"{site}_tangential_force", 0.0)))
            site_history["tangential_direction"].append(float(tactile.get(f"{site}_tangential_direction", -1.0)))

            for col_idx, metric_key in enumerate(("normal_force", "tangential_force", "tangential_direction")):
                ax = self.axes[row_idx][col_idx]
                line = self.lines[site][metric_key]
                y_vals = list(site_history[metric_key])
                line.set_data(x_vals, y_vals)
                ax.relim()
                ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                if not self.enabled:
                    break
                hand_data = self.hand_controller.get_data()
                tactile = hand_data.get("tactile")
                if tactile:
                    with self.plot_lock:
                        self._update_plot(tactile)
                time.sleep(self.refresh_interval)
        except Exception as exc:
            self.enabled = False
            print(f"Disabling realtime tactile plot after update failure: {exc}")

    def start(self):
        if not self.enabled or self.worker is not None:
            return
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            self.worker = None
        with self.plot_lock:
            if self.fig is not None and self.plt is not None:
                self.plt.close(self.fig)
                self.fig = None