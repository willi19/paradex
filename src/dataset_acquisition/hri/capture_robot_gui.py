"""GUI-driven version of ``capture_robot.py``.

All command-line options from ``capture_robot.py`` remain available except
``--name``. Object names are selected from directories below
``shared_data/mesh_new``.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets

SHARED_DIR = Path.home() / "shared_data"
MESH_ROOT = SHARED_DIR / "mesh_new"

def parse_args():
    parser = argparse.ArgumentParser(
        description="GUI for VIVE xArm capture sessions.",
    )
    parser.add_argument(
        "--device",
        choices=["xsens", "occulus", "vive"],
        default="vive",
    )
    parser.add_argument(
        "--hand-side",
        choices=["right", "left", "bimanual"],
        default="right",
    )
    vive_group = parser.add_mutually_exclusive_group()
    vive_group.add_argument(
        "--use-vive",
        dest="use_vive",
        action="store_true",
        help="Use the VIVE wrist tracker for arm/wrist pose fusion (default).",
    )
    vive_group.add_argument(
        "--no-vive",
        dest="use_vive",
        action="store_false",
        help="Use Manus glove data only; suitable for hand-only teleoperation.",
    )
    parser.set_defaults(use_vive=True)

    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--camera",
        dest="camera_mode",
        nargs="?",
        const="capture",
        choices=["capture", "preview"],
        default="capture",
        help=(
            "Enable remote camera recording. Use '--camera=preview' to also "
            "show the independent live capture-PC preview window."
        ),
    )
    camera_group.add_argument(
        "--no-camera",
        dest="camera_mode",
        action="store_const",
        const="off",
        help="Disable remote cameras, sync generator, and timestamp monitor.",
    )
    camera_group.add_argument(
        "--camera-preview",
        dest="camera_mode",
        action="store_const",
        const="preview",
        help="Alias for '--camera=preview'.",
    )
    parser.add_argument("--camera-preview-port", type=int, default=5484)
    parser.add_argument(
        "--camera-preview-refresh-interval",
        type=float,
        default=1.0 / 30.0,
    )
    parser.add_argument(
        "--camera-preview-request-timeout",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--no-timestamp",
        dest="timestamp",
        action="store_false",
        help="Disable the timestamp monitor connection.",
    )
    parser.set_defaults(timestamp=True)
    parser.add_argument(
        "--arm",
        default="xarm",
        help="Arm controller name. Use 'none' (or empty) to disable arm control.",
    )
    parser.add_argument(
        "--hand",
        default="inspire_f1",
        help=(
            "Hand controller/retargetor name. Use 'none' for arm-only teleop; "
            "'allegro_v5' for the direct-anchor retargeter; 'wuji' for "
            "optimization, 'wuji_direct' for direct mapping, or 'wuji_hybrid' "
            "for opt thumb + direct fingers."
        ),
    )
    parser.add_argument(
        "--capture_root",
        default="eccv2026/allegro_v5",
        help="Directory below shared_data/capture used for saved episodes.",
    )
    parser.add_argument("--tactile", action="store_true")
    parser.add_argument("--ip", action="store_true")
    parser.add_argument(
        "--allegro-command-rate-hz",
        type=float,
        default=30.0,
        help=(
            "Maximum live target-update rate for Allegro V5 hands. Defaults "
            "to 30 Hz."
        ),
    )
    parser.add_argument(
        "--allegro-teleop-log",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Record commanded Allegro V5 ticks. Without PATH, save a "
            "timestamped .npz below the initially selected object name."
        ),
    )
    parser.add_argument("--inspire-right-interface", default="enp8s0f1")
    parser.add_argument("--inspire-right-ip", default="192.168.11.211")
    parser.add_argument("--inspire-left-interface", default="enp8s0f2")
    parser.add_argument("--inspire-left-ip", default="192.168.11.210")
    parser.add_argument(
        "--visualize-tactile-realtime",
        action="store_true",
        help="Show live tactile feedback.",
    )
    parser.add_argument(
        "--allegro-visualization-rate-hz",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--allegro-tactile-display-max",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--xarm-servo-api",
        choices=["cartesian_aa", "angle_j"],
        default="cartesian_aa",
    )
    parser.add_argument(
        "--scale",
        "--hand-scale",
        dest="hand_scale",
        type=float,
        default=1.15,
    )

    args = parser.parse_args()
    args.arm = _normalize_optional_name(args.arm)
    args.hand = _normalize_optional_name(args.hand)
    if args.allegro_command_rate_hz <= 0.0:
        parser.error("--allegro-command-rate-hz must be positive.")
    if args.camera_preview_port <= 0:
        parser.error("--camera-preview-port must be positive.")
    if args.camera_preview_refresh_interval <= 0.0:
        parser.error("--camera-preview-refresh-interval must be positive.")
    if args.camera_preview_request_timeout <= 0.0:
        parser.error("--camera-preview-request-timeout must be positive.")
    if args.allegro_visualization_rate_hz <= 0.0:
        parser.error("--allegro-visualization-rate-hz must be positive.")
    if args.allegro_tactile_display_max <= 0.0:
        parser.error("--allegro-tactile-display-max must be positive.")
    if args.allegro_teleop_log is not None and (
        args.hand != "allegro_v5" or args.hand_side != "right"
    ):
        parser.error(
            "--allegro-teleop-log currently requires --hand allegro_v5 "
            "and --hand-side right."
        )
    return args


def _normalize_optional_name(name):
    if name is not None and name.strip().lower() in ("", "none", "null"):
        return None
    return name


def discover_mesh_names(mesh_root=MESH_ROOT):
    mesh_root = Path(mesh_root)
    if not mesh_root.is_dir():
        return []
    return sorted(
        path.name
        for path in mesh_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


class QueueStream:
    """Route stdout/stderr text to the GUI event queue."""

    def __init__(self, ui_queue, stream_name):
        self.ui_queue = ui_queue
        self.stream_name = stream_name

    def write(self, text):
        if text:
            self.ui_queue.put(("log", text, self.stream_name))
        return len(text)

    def flush(self):
        return None

    def isatty(self):
        return False


class StepCard(QtWidgets.QFrame):
    def __init__(self, number, title, subtitle):
        super().__init__()
        self.setObjectName("stepCard")
        self.setProperty("state", "pending")
        self.setMinimumHeight(74)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 12, 10)
        layout.setSpacing(12)

        self.number = QtWidgets.QLabel(str(number))
        self.number.setObjectName("stepNumber")
        self.number.setAlignment(QtCore.Qt.AlignCenter)
        self.number.setFixedSize(38, 38)
        layout.addWidget(self.number)

        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("stepTitle")
        subtitle_label = QtWidgets.QLabel(subtitle)
        subtitle_label.setObjectName("stepSubtitle")
        subtitle_label.setWordWrap(True)
        text.addWidget(title_label)
        text.addWidget(subtitle_label)
        layout.addLayout(text, 1)

    def set_state(self, state, number):
        self.setProperty("state", state)
        self.number.setText("✓" if state == "complete" else str(number))
        for widget in (self, self.number):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, label, initial="—"):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 11, 18, 11)
        layout.setSpacing(2)
        caption = QtWidgets.QLabel(label)
        caption.setObjectName("metricCaption")
        self.value = QtWidgets.QLabel(initial)
        self.value.setObjectName("metricValue")
        self.value.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        layout.addWidget(caption)
        layout.addWidget(self.value)


class CaptureRobotGui(QtWidgets.QMainWindow):
    POLL_INTERVAL_MS = 40

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.capture_root = args.capture_root
        self.ui_queue = queue.Queue()

        self.save_event = threading.Event()
        self.stop_event = threading.Event()
        self.exit_event = threading.Event()
        self.grasp_yes_event = threading.Event()
        self.grasp_no_event = threading.Event()
        self.paired_episode_event = threading.Event()
        self.name_selected_event = threading.Event()
        self.events = {
            "save": self.save_event,
            "stop": self.stop_event,
            "exit": self.exit_event,
        }

        self._name_lock = threading.Lock()
        self._selected_name = None
        self._paired_episode = None
        self.mesh_names = []
        self.filtered_names = []
        self.runtime_state = "initializing"
        self.worker = None
        self.camera_preview = None
        self.tactile_plotter = None
        self._next_preview_time = 0.0
        self._close_requested = False
        self._shutdown_complete = False
        self.success_count = 0
        self.fail_count = 0
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._build_ui()
        self._apply_style()
        self._refresh_mesh_names()
        self._set_runtime_state("select_name")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_ui_queue)
        self.timer.start(self.POLL_INTERVAL_MS)

    def _build_ui(self):
        self.setWindowTitle("Robot Capture")
        self.setMinimumSize(1280, 760)
        self.resize(1540, 920)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(390)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(26, 28, 26, 24)
        side.setSpacing(12)
        brand = QtWidgets.QLabel("PARADEX · CAPTURE")
        brand.setObjectName("brand")
        heading = QtWidgets.QLabel("Robot\nCapture Session")
        heading.setObjectName("sideHeading")
        side.addWidget(brand)
        side.addWidget(heading)
        side.addSpacing(6)

        self.step_cards = []
        steps = [
            ("Object", "캡처할 물체 선택"),
            ("Ready", "장치 연결과 대기"),
            ("Recording", "로봇 조작과 데이터 수집"),
            ("Result", "성공 여부와 paired episode 저장"),
        ]
        for index, (title, subtitle) in enumerate(steps, start=1):
            card = StepCard(index, title, subtitle)
            self.step_cards.append(card)
            side.addWidget(card)

        object_card = QtWidgets.QFrame()
        object_card.setObjectName("objectCard")
        object_layout = QtWidgets.QVBoxLayout(object_card)
        object_layout.setContentsMargins(14, 13, 14, 13)
        object_layout.setSpacing(8)
        object_title = QtWidgets.QLabel("OBJECT NAME")
        object_title.setObjectName("sectionTitle")
        object_layout.addWidget(object_title)
        self.search_entry = QtWidgets.QLineEdit()
        self.search_entry.setObjectName("searchEntry")
        self.search_entry.setPlaceholderText("물체 이름 검색…")
        self.search_entry.setClearButtonEnabled(True)
        self.search_entry.textChanged.connect(self._filter_mesh_names)
        object_layout.addWidget(self.search_entry)
        self.name_listbox = QtWidgets.QListWidget()
        self.name_listbox.setObjectName("nameList")
        self.name_listbox.setMinimumHeight(115)
        self.name_listbox.currentTextChanged.connect(self._on_name_selected)
        object_layout.addWidget(self.name_listbox, 1)
        object_actions = QtWidgets.QHBoxLayout()
        self.selected_name_label = QtWidgets.QLabel("선택된 물체: 없음")
        self.selected_name_label.setObjectName("selectedName")
        self.selected_name_label.setWordWrap(True)
        object_actions.addWidget(self.selected_name_label, 1)
        refresh = QtWidgets.QPushButton("새로고침")
        refresh.setObjectName("smallButton")
        refresh.clicked.connect(self._refresh_mesh_names)
        object_actions.addWidget(refresh)
        object_layout.addLayout(object_actions)
        side.addWidget(object_card, 1)

        capture_path = QtWidgets.QLabel(
            f"CAPTURE ROOT\nshared_data/capture/{self.capture_root}"
        )
        capture_path.setObjectName("mapping")
        capture_path.setWordWrap(True)
        capture_path.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        side.addWidget(capture_path)
        root.addWidget(sidebar)

        content = QtWidgets.QWidget()
        content.setObjectName("content")
        main = QtWidgets.QVBoxLayout(content)
        main.setContentsMargins(34, 26, 34, 28)
        main.setSpacing(18)

        top_row = QtWidgets.QHBoxLayout()
        self.stage_badge = QtWidgets.QLabel("STEP 1 OF 4")
        self.stage_badge.setObjectName("stageBadge")
        top_row.addWidget(self.stage_badge)
        top_row.addStretch(1)
        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setObjectName("detail")
        top_row.addWidget(self.detail_label)
        main.addLayout(top_row)

        self.current_title = QtWidgets.QLabel("장치를 준비하는 중입니다")
        self.current_title.setObjectName("currentTitle")
        self.current_title.setWordWrap(True)
        main.addWidget(self.current_title)
        self.current_subtitle = QtWidgets.QLabel("초기 물체를 선택하면 캡처 장치를 연결합니다.")
        self.current_subtitle.setObjectName("currentSubtitle")
        self.current_subtitle.setWordWrap(True)
        main.addWidget(self.current_subtitle)

        instruction_card = QtWidgets.QFrame()
        instruction_card.setObjectName("instructionCard")
        instruction_layout = QtWidgets.QVBoxLayout(instruction_card)
        instruction_layout.setContentsMargins(22, 16, 22, 16)
        instruction_layout.setSpacing(12)
        self.instructions = QtWidgets.QLabel("")
        self.instructions.setObjectName("instructions")
        self.instructions.setWordWrap(True)
        instruction_layout.addWidget(self.instructions)

        capture_buttons = QtWidgets.QHBoxLayout()
        capture_buttons.setSpacing(10)
        self.start_button = QtWidgets.QPushButton("C  Capture start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._on_start)
        capture_buttons.addWidget(self.start_button, 2)
        self.stop_button = QtWidgets.QPushButton("S  Stop & save")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._on_stop)
        capture_buttons.addWidget(self.stop_button, 2)
        self.exit_button = QtWidgets.QPushButton("Q  Exit")
        self.exit_button.setObjectName("exitButton")
        self.exit_button.clicked.connect(self._on_exit)
        capture_buttons.addWidget(self.exit_button, 1)
        instruction_layout.addLayout(capture_buttons)
        main.addWidget(instruction_card)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        main.addWidget(self.progress)

        metrics = QtWidgets.QHBoxLayout()
        metrics.setSpacing(12)
        self.object_metric = MetricCard("선택된 물체")
        self.episode_metric = MetricCard("현재 EPISODE", "대기")
        self.success_metric = MetricCard("성공", "0")
        self.failure_metric = MetricCard("실패", "0")
        for card in (
            self.object_metric,
            self.episode_metric,
            self.success_metric,
            self.failure_metric,
        ):
            metrics.addWidget(card)
        main.addLayout(metrics)

        result_card = QtWidgets.QFrame()
        result_card.setObjectName("resultCard")
        result_layout = QtWidgets.QHBoxLayout(result_card)
        result_layout.setContentsMargins(18, 14, 18, 14)
        result_layout.setSpacing(10)
        result_title = QtWidgets.QLabel("GRASP RESULT")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)
        self.success_button = QtWidgets.QPushButton("Y  Success")
        self.success_button.setObjectName("successButton")
        self.success_button.clicked.connect(self._on_grasp_success)
        result_layout.addWidget(self.success_button)
        self.failure_button = QtWidgets.QPushButton("N  Failure")
        self.failure_button.setObjectName("failureButton")
        self.failure_button.clicked.connect(self._on_grasp_failure)
        result_layout.addWidget(self.failure_button)
        result_layout.addSpacing(12)
        pair_label = QtWidgets.QLabel("Paired human episode")
        pair_label.setObjectName("fieldLabel")
        result_layout.addWidget(pair_label)
        self.paired_episode_entry = QtWidgets.QLineEdit()
        self.paired_episode_entry.setObjectName("pairEntry")
        self.paired_episode_entry.setPlaceholderText("episode number")
        self.paired_episode_entry.setValidator(QtGui.QIntValidator(0, 999999999))
        self.paired_episode_entry.returnPressed.connect(self._on_paired_submit)
        result_layout.addWidget(self.paired_episode_entry)
        self.paired_submit_button = QtWidgets.QPushButton("Save pair")
        self.paired_submit_button.setObjectName("smallButton")
        self.paired_submit_button.clicked.connect(self._on_paired_submit)
        result_layout.addWidget(self.paired_submit_button)
        main.addWidget(result_card)

        diagnostic_card = QtWidgets.QFrame()
        diagnostic_card.setObjectName("diagnosticCard")
        diagnostic_layout = QtWidgets.QVBoxLayout(diagnostic_card)
        diagnostic_layout.setContentsMargins(16, 10, 16, 10)
        diagnostic_layout.setSpacing(6)
        log_header = QtWidgets.QHBoxLayout()
        log_title = QtWidgets.QLabel("CAPTURE LOG")
        log_title.setObjectName("diagnosticTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_path = QtWidgets.QLabel("Terminal output is mirrored here")
        log_path.setObjectName("diagnosticPaths")
        log_header.addWidget(log_path)
        diagnostic_layout.addLayout(log_header)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setObjectName("liveLog")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        diagnostic_layout.addWidget(self.log_text, 1)
        main.addWidget(diagnostic_card, 1)
        root.addWidget(content, 1)

    def _apply_style(self):
        self.setStyleSheet(
            """
            * { font-family: "Noto Sans CJK KR", "Noto Sans KR", "Sans"; color: #e5e7eb; }
            QMainWindow, #content { background: #0b1120; }
            #sidebar { background: #111827; border-right: 1px solid #263247; }
            #brand { color: #60a5fa; font-size: 14px; font-weight: 800; }
            #sideHeading { color: white; font-size: 30px; font-weight: 800; }
            #stepCard { background: #172033; border: 1px solid #273449; border-radius: 12px; }
            #stepCard[state="current"] { background: #172554; border: 2px solid #3b82f6; }
            #stepCard[state="complete"] { background: #102a25; border: 1px solid #22c55e; }
            #stepCard[state="error"] { background: #351820; border: 2px solid #ef4444; }
            #stepNumber { background: #263247; border-radius: 19px; color: #cbd5e1; font-size: 16px; font-weight: 800; }
            #stepCard[state="current"] #stepNumber { background: #2563eb; color: white; }
            #stepCard[state="complete"] #stepNumber { background: #16a34a; color: white; }
            #stepCard[state="error"] #stepNumber { background: #dc2626; color: white; }
            #stepTitle { color: #f8fafc; font-size: 16px; font-weight: 750; }
            #stepSubtitle { color: #94a3b8; font-size: 12px; }
            #objectCard, #instructionCard, #resultCard, #metricCard, #diagnosticCard {
                background: #111827; border: 1px solid #273449; border-radius: 12px;
            }
            #objectCard { background: #0b1220; }
            #sectionTitle, #diagnosticTitle { color: #cbd5e1; font-size: 13px; font-weight: 800; }
            #searchEntry, #pairEntry { background: #080d18; border: 1px solid #334155; border-radius: 7px; padding: 8px; color: white; }
            #searchEntry:focus, #pairEntry:focus { border: 1px solid #3b82f6; }
            #nameList { background: #080d18; border: 1px solid #263247; border-radius: 7px; color: #cbd5e1; padding: 4px; }
            #nameList::item { padding: 6px; border-radius: 5px; }
            #nameList::item:selected { background: #1d4ed8; color: white; }
            #selectedName { color: #93c5fd; font-size: 13px; font-weight: 700; }
            #mapping { background: #0b1220; border-radius: 10px; padding: 14px; color: #94a3b8; font-size: 12px; }
            #stageBadge { background: #1d4ed8; border-radius: 12px; padding: 7px 13px; color: white; font-size: 14px; font-weight: 800; }
            #detail { color: #93c5fd; font-size: 15px; font-weight: 600; }
            #currentTitle { color: white; font-size: 32px; font-weight: 850; }
            #currentSubtitle { color: #a8b3c7; font-size: 18px; }
            #instructions { color: #dbeafe; font-size: 16px; }
            QPushButton { min-height: 38px; padding: 5px 14px; border-radius: 9px; font-weight: 750; }
            #primaryButton { background: #2563eb; border: none; color: white; font-size: 17px; }
            #primaryButton:hover { background: #3b82f6; }
            #stopButton { background: #b45309; border: none; color: white; font-size: 17px; }
            #stopButton:hover { background: #d97706; }
            #exitButton, #smallButton { background: #263247; border: 1px solid #3b4a62; color: #e5e7eb; }
            #exitButton:hover, #smallButton:hover { background: #334155; }
            #successButton { background: #15803d; border: none; color: white; }
            #successButton:hover { background: #16a34a; }
            #failureButton { background: #b91c1c; border: none; color: white; }
            #failureButton:hover { background: #dc2626; }
            QPushButton:disabled { background: #263247; border-color: #263247; color: #64748b; }
            QProgressBar { background: #1f2937; border: none; border-radius: 6px; }
            QProgressBar::chunk { background: #3b82f6; border-radius: 6px; }
            #metricCaption { color: #94a3b8; font-size: 13px; font-weight: 650; }
            #metricValue { color: white; font-size: 22px; font-weight: 850; }
            #fieldLabel { color: #94a3b8; font-size: 13px; }
            #diagnosticPaths { color: #64748b; font-size: 12px; }
            #liveLog { background: #080d18; border: 1px solid #263247; border-radius: 7px; color: #94a3b8; font-family: "DejaVu Sans Mono", "Monospace"; font-size: 11px; padding: 6px; }
            QScrollBar:vertical { background: #111827; width: 11px; margin: 0; }
            QScrollBar::handle:vertical { background: #334155; border-radius: 5px; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )

    def start_worker(self):
        sys.stdout = QueueStream(self.ui_queue, "stdout")
        sys.stderr = QueueStream(self.ui_queue, "stderr")
        self.worker = threading.Thread(
            target=self._capture_worker,
            name="capture-robot-worker",
            daemon=True,
        )
        self.worker.start()

    def restore_streams(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def _refresh_mesh_names(self):
        self.mesh_names = discover_mesh_names()
        self._filter_mesh_names()
        if not self.mesh_names:
            self._append_log(f"No object directories found in {MESH_ROOT}\n")

    def _filter_mesh_names(self, *_args):
        query = self.search_entry.text().strip().casefold()
        self.filtered_names = [
            name for name in self.mesh_names if query in name.casefold()
        ]
        selected = self._get_selected_name()
        self.name_listbox.blockSignals(True)
        self.name_listbox.clear()
        self.name_listbox.addItems(self.filtered_names)
        if selected in self.filtered_names:
            self.name_listbox.setCurrentRow(self.filtered_names.index(selected))
        self.name_listbox.blockSignals(False)

    def _on_name_selected(self, name=""):
        if not name:
            return
        with self._name_lock:
            self._selected_name = name
        self.name_selected_event.set()
        self.selected_name_label.setText(f"선택된 물체: {name}")
        self.object_metric.value.setText(name)
        if self.runtime_state == "ready":
            self._set_runtime_state("ready")
        if self.runtime_state == "recording":
            self._append_log(
                f"Next object selected: {name} "
                "(the active episode is unchanged)\n"
            )

    def _get_selected_name(self):
        with self._name_lock:
            return self._selected_name

    def _on_start(self):
        name = self._get_selected_name()
        if name is None:
            self._append_log("Select an object name before starting capture.\n")
            return
        self.save_event.set()
        self._set_runtime_state("start_requested", name=name)

    def _on_stop(self):
        self.stop_event.set()
        self._set_runtime_state("stop_requested")

    def _on_grasp_success(self):
        self.grasp_no_event.clear()
        self.grasp_yes_event.set()
        self._set_runtime_state("saving_result")

    def _on_grasp_failure(self):
        self.grasp_yes_event.clear()
        self.grasp_no_event.set()
        self._set_runtime_state("saving_result")

    def _on_paired_submit(self):
        try:
            episode = int(self.paired_episode_entry.text().strip())
        except ValueError:
            self._append_log("Paired human episode must be an integer.\n")
            self.paired_episode_entry.setFocus()
            return
        self._paired_episode = episode
        self.paired_episode_event.set()
        self._set_runtime_state("saving_result")

    def _on_exit(self):
        if self.exit_event.is_set():
            return
        self._close_requested = True
        self.exit_event.set()
        self.save_event.clear()
        self.stop_event.set()
        self.grasp_yes_event.set()
        self.grasp_no_event.set()
        self.paired_episode_event.set()
        self._set_runtime_state("exiting")
        self._append_log("Exit requested. Finishing device shutdown...\n")
        if self.worker is not None and not self.worker.is_alive():
            self._shutdown_complete = True
            QtCore.QTimer.singleShot(100, self.close)

    def closeEvent(self, event):
        if self._shutdown_complete:
            event.accept()
            return
        event.ignore()
        self._on_exit()

    def _set_runtime_state(self, state, **payload):
        self.runtime_state = state
        copy = {
            "initializing": ("캡처 장치를 연결하는 중입니다", "연결이 완료될 때까지 잠시 기다려 주세요.", "CaptureSession initializing", 22, 1),
            "select_name": ("첫 번째 물체를 선택하세요", "왼쪽 검색창에서 shared_data/mesh_new의 폴더를 고릅니다.", "Waiting for object selection", 4, 0),
            "ready": ("캡처를 시작할 수 있습니다", "VIVE teleoperation을 준비하고 C 버튼을 누르세요.", "Ready", 35, 1),
            "start_requested": ("녹화를 시작하는 중입니다", "카메라와 로봇 스트림이 준비됩니다.", "Start requested", 48, 2),
            "recording": ("로봇 데이터를 캡처하고 있습니다", "VIVE로 로봇을 조작하고, 끝내려면 S 버튼을 누르세요.", "Recording", 62, 2),
            "stop_requested": ("캡처 종료를 요청했습니다", "현재 스트림을 정상 종료하고 있습니다.", "Stop requested", 70, 2),
            "saving": ("캡처 데이터를 저장하고 있습니다", "장치 종료와 파일 저장이 끝날 때까지 잠시 기다리세요.", "Saving episode", 76, 2),
            "await_grasp": ("Grasp 성공 여부를 선택하세요", "Y Success 또는 N Failure 버튼으로 결과를 저장합니다.", "Waiting for Y / N", 84, 3),
            "saving_result": ("결과를 저장하고 있습니다", "Grasp 결과와 paired human episode을 연결합니다.", "Saving metadata", 90, 3),
            "await_paired": ("Paired human episode을 입력하세요", "연결할 human episode 번호를 입력한 뒤 Save pair를 누르세요.", "Waiting for episode number", 94, 3),
            "exiting": ("캡처 세션을 종료하고 있습니다", "로봇과 카메라 연결을 안전하게 닫고 있습니다.", "Shutting down", 98, 3),
            "error": ("캡처 중 오류가 발생했습니다", "아래 Capture log에서 traceback을 확인하세요.", "Capture failed", 0, 1),
            "done": ("캡처 세션이 종료됐습니다", "모든 장치 연결을 정상적으로 종료했습니다.", "Stopped", 100, 3),
        }
        title, subtitle, detail, progress, current_step = copy.get(
            state, (state, "", state, 0, 0)
        )
        self.current_title.setText(title)
        self.current_subtitle.setText(subtitle)
        self.detail_label.setText(detail)
        self.progress.setValue(progress)
        self.stage_badge.setText(f"STEP {current_step + 1} OF 4")
        instructions = {
            0: "✓ 물체 이름은 캡처 전에 반드시 선택합니다.\n✓ 캡처 중 변경한 이름은 다음 episode부터 적용됩니다.",
            1: "✓ 로봇·핸드·VIVE·카메라 연결 로그를 확인합니다.\n✓ C로 캡처를 시작하고 Q로 세션을 종료합니다.",
            2: "✓ VIVE 상대 변환으로 end effector를 조작합니다.\n✓ S를 누르면 현재 episode를 정지하고 저장합니다.",
            3: "✓ Y/N으로 grasp 결과를 저장합니다.\n✓ 이어서 paired human episode 번호를 저장합니다.",
        }
        self.instructions.setText(instructions[current_step])

        for index, card in enumerate(self.step_cards):
            if state == "error" and index == current_step:
                step_state = "error"
            elif index < current_step:
                step_state = "complete"
            elif index == current_step:
                step_state = "current"
            else:
                step_state = "pending"
            card.set_state(step_state, index + 1)

        name = payload.get("name")
        episode = payload.get("episode")
        if name is not None and episode is not None:
            self.object_metric.value.setText(name)
            self.episode_metric.value.setText(str(episode))
        elif state in ("ready", "done"):
            self.episode_metric.value.setText("대기")

        selected = self._get_selected_name()
        can_start = state == "ready" and selected is not None
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(state in ("recording", "start_requested"))
        grasp_enabled = state == "await_grasp"
        self.success_button.setEnabled(grasp_enabled)
        self.failure_button.setEnabled(grasp_enabled)
        paired_enabled = state == "await_paired"
        self.paired_episode_entry.setEnabled(paired_enabled)
        self.paired_submit_button.setEnabled(paired_enabled)
        self.exit_button.setEnabled(state != "done")

        if state == "await_paired":
            self.paired_episode_entry.clear()
            self.paired_episode_entry.setFocus()

    def _append_log(self, text):
        if not text:
            return
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _poll_ui_queue(self):
        should_destroy = False
        try:
            while True:
                message = self.ui_queue.get_nowait()
                kind = message[0]
                if kind == "log":
                    self._append_log(message[1])
                elif kind == "state":
                    self._set_runtime_state(message[1], **message[2])
                elif kind == "done":
                    should_destroy = bool(message[1])
                    if should_destroy:
                        self._set_runtime_state("done")
                    elif self.runtime_state != "error":
                        self._set_runtime_state("done")
                elif kind == "stats":
                    self.success_count = message[1]
                    self.fail_count = message[2]
                    self.success_metric.value.setText(str(self.success_count))
                    self.failure_metric.value.setText(str(self.fail_count))
        except queue.Empty:
            pass

        self._refresh_camera_preview()

        if should_destroy:
            self._shutdown_complete = True
            QtCore.QTimer.singleShot(250, self.close)

    def _refresh_camera_preview(self):
        preview = self.camera_preview
        if preview is None:
            return
        now = time.monotonic()
        if now < self._next_preview_time:
            return
        self._next_preview_time = now + self.args.camera_preview_refresh_interval
        try:
            preview.show()
        except Exception as exc:
            self.camera_preview = None
            self._append_log(f"Camera preview failed: {exc}\n")

    def _post_state(self, state, **payload):
        self.ui_queue.put(("state", state, payload))

    def _wait_for_grasp_result(self):
        self.grasp_yes_event.clear()
        self.grasp_no_event.clear()
        print("Grasp success? Press Y or N.")
        self._post_state("await_grasp")

        while not self.exit_event.is_set():
            if self.grasp_yes_event.wait(timeout=0.05):
                if self.exit_event.is_set():
                    return None
                return True
            if self.grasp_no_event.is_set():
                if self.exit_event.is_set():
                    return None
                return False
        return None

    def _wait_for_paired_episode(self, name):
        self._paired_episode = None
        self.paired_episode_event.clear()
        print(f"Enter the paired human episode for {name}.")
        self._post_state("await_paired")
        while not self.exit_event.is_set():
            if self.paired_episode_event.wait(timeout=0.05):
                return self._paired_episode
        return None

    def _capture_worker(self):
        while not self.exit_event.is_set():
            if self.name_selected_event.wait(timeout=0.05):
                break
        if self.exit_event.is_set():
            self.ui_queue.put(("done", True))
            return

        initial_name = self._get_selected_name()
        self._post_state("initializing")

        args = self.args
        cs = None
        pedal_state = None
        success_count = 0
        fail_count = 0

        try:
            from paradex.dataset_acqusition.capture import CaptureSession
            from paradex.utils.file_io import find_latest_index
            from paradex.utils.system import get_pc_list

            camera_pc_list = get_pc_list()
            camera_enabled = args.camera_mode != "off"
            camera_preview_enabled = args.camera_mode == "preview"
            allegro_v5_hands = {"allegro_v5"}
            allegro_realtime_visualization = (
                args.visualize_tactile_realtime
                and args.hand in allegro_v5_hands
                and args.hand_side == "right"
            )
            inspire_bimanual = (
                args.hand in ("inspire", "inspire_dftp")
                and args.hand_side == "bimanual"
            )
            hand_kwargs = None
            if inspire_bimanual:
                hand_kwargs = {
                    "right": {
                        "interface": args.inspire_right_interface,
                        "host": args.inspire_right_ip,
                    },
                    "left": {
                        "interface": args.inspire_left_interface,
                        "host": args.inspire_left_ip,
                    },
                }
                print(
                    "Bimanual Inspire Modbus TCP: "
                    f"right={args.inspire_right_ip} via "
                    f"{args.inspire_right_interface}, "
                    f"left={args.inspire_left_ip} via "
                    f"{args.inspire_left_interface}"
                )

            allegro_teleop_diagnostic_path = None
            if args.allegro_teleop_log is not None:
                if args.allegro_teleop_log == "auto":
                    diagnostic_name = datetime.now().strftime(
                        "allegro_teleop_%Y%m%d_%H%M%S.npz"
                    )
                    allegro_teleop_diagnostic_path = (
                        SHARED_DIR
                        / "capture"
                        / args.capture_root
                        / initial_name
                        / diagnostic_name
                    )
                else:
                    allegro_teleop_diagnostic_path = Path(
                        args.allegro_teleop_log
                    ).expanduser()

            cs = CaptureSession(
                camera=camera_enabled,
                realsense=False,
                arm=args.arm,
                hand=args.hand,
                teleop=args.device,
                hand_side=args.hand_side,
                events=self.events,
                tactile=args.tactile or allegro_realtime_visualization,
                ip=args.ip or inspire_bimanual,
                hand_kwargs=hand_kwargs,
                timestamp=args.timestamp,
                camera_pc_list=camera_pc_list,
                arm_kwargs=(
                    {"servo_api": args.xarm_servo_api}
                    if args.arm == "xarm"
                    else None
                ),
                hand_scale=args.hand_scale,
                hand_command_rate_hz=(
                    args.allegro_command_rate_hz
                    if args.hand == "allegro_v5"
                    else None
                ),
                allegro_teleop_diagnostic_path=allegro_teleop_diagnostic_path,
                use_vive=args.use_vive,
                require_left_control=args.use_vive,
            )

            if allegro_teleop_diagnostic_path is not None:
                print(
                    "Allegro teleop diagnostic will be saved to: "
                    f"{allegro_teleop_diagnostic_path}"
                )

            if args.hand_side == "bimanual":
                from paradex.io.streamdeck_pedal import MiddlePedalState

                pedal_state = MiddlePedalState()
            bimanual_state_provider = (
                pedal_state.get_state if pedal_state is not None else None
            )

            if args.visualize_tactile_realtime:
                if args.hand_side == "bimanual":
                    print(
                        "Realtime tactile visualization is not supported in "
                        "bimanual mode. Ignoring option."
                    )
                elif args.hand in allegro_v5_hands and args.hand_side != "right":
                    print(
                        "Realtime Allegro visualization currently supports "
                        "the right hand only. Ignoring option."
                    )
                elif args.hand in allegro_v5_hands:
                    from paradex.visualization.allegro_realtime import (
                        AllegroRealtimeViser,
                    )

                    self.tactile_plotter = AllegroRealtimeViser(
                        cs.hand,
                        update_rate_hz=args.allegro_visualization_rate_hz,
                        tactile_display_max=args.allegro_tactile_display_max,
                    )
                    self.tactile_plotter.start()
                elif args.hand != "inspire_f1":
                    print(
                        "Realtime tactile visualization supports inspire_f1 "
                        "and Allegro V5. Ignoring option."
                    )
                elif not args.tactile:
                    print(
                        "Realtime tactile visualization requires --tactile. "
                        "Ignoring option."
                    )
                else:
                    from paradex.io.robot_controller.inspire_f1_tactile_plotter import (
                        InspireF1RealtimeTactilePlotter,
                    )

                    self.tactile_plotter = InspireF1RealtimeTactilePlotter(
                        cs.hand
                    )
                    if self.tactile_plotter.enabled:
                        self.tactile_plotter.start()

            if camera_preview_enabled:
                from paradex.io.camera_system.capture_pc_preview import (
                    CapturePcPreviewGui,
                )

                self.camera_preview = CapturePcPreviewGui(
                    pc_list=camera_pc_list,
                    port=args.camera_preview_port,
                    refresh_interval=args.camera_preview_refresh_interval,
                    request_timeout=args.camera_preview_request_timeout,
                    side_panel_provider=getattr(
                        self.tactile_plotter,
                        "render_bgr",
                        None,
                    ),
                )
                self.camera_preview.start()

            print(f"Object directory: {MESH_ROOT}")
            print(f"Capture root: {self.capture_root}")
            print("GUI controls: C=start, S=stop, Q=exit, Y=success, N=failure")
            self._post_state("ready")

            while not self.exit_event.is_set():
                state = cs.teleop(
                    session_events=self.events,
                    state_policy="keyboard_control",
                    bimanual_state_provider=bimanual_state_provider,
                )
                if state == "exit":
                    break
                if state != "start":
                    continue

                name = self._get_selected_name()
                if name is None:
                    self.save_event.clear()
                    print("Select an object name before starting capture.")
                    self._post_state("ready")
                    continue

                episode_root = os.path.join(
                    str(SHARED_DIR),
                    "capture",
                    self.capture_root,
                    name,
                )
                episode = int(find_latest_index(episode_root)) + 1
                episode_rel_path = os.path.join(
                    "capture",
                    self.capture_root,
                    name,
                    str(episode),
                )
                episode_abs_path = os.path.join(str(SHARED_DIR), episode_rel_path)

                print("Prepare to record new session:", name, "episode:", episode)
                self.stop_event.clear()
                cs.start(episode_rel_path)
                print("Starting new recording session:", name)
                print("Capturing index:", episode)
                self._post_state("recording", name=name, episode=episode)

                state = cs.teleop(
                    session_events=self.events,
                    state_policy="keyboard_control",
                    bimanual_state_provider=bimanual_state_provider,
                )
                self._post_state("saving", name=name, episode=episode)
                print("Stopping recording session:", name)
                cs.stop()
                print("Stopped recording session:", name)

                timestamp_path = os.path.join(
                    episode_abs_path,
                    "raw",
                    "timestamps",
                    "timestamp.npy",
                )
                if os.path.exists(timestamp_path):
                    print(f"timestamp.npy length: {len(np.load(timestamp_path))}")
                else:
                    print(f"timestamp.npy not found at {timestamp_path}")

                self.save_event.clear()
                self.stop_event.clear()

                if state != "exit" and not self.exit_event.is_set():
                    grasp_success = self._wait_for_grasp_result()
                    if grasp_success is not None:
                        success_count += int(grasp_success)
                        fail_count += int(not grasp_success)
                        os.makedirs(episode_abs_path, exist_ok=True)
                        with open(
                            os.path.join(episode_abs_path, "grasp_result.json"),
                            "w",
                        ) as file:
                            json.dump(
                                {
                                    "episode": episode,
                                    "grasp_success": grasp_success,
                                },
                                file,
                                indent=2,
                            )

                        paired_episode = self._wait_for_paired_episode(name)
                        if paired_episode is not None:
                            with open(
                                os.path.join(
                                    episode_abs_path,
                                    "paired_human_episode.json",
                                ),
                                "w",
                            ) as file:
                                json.dump(
                                    {
                                        "human hand episode": episode,
                                        "paired human episode": paired_episode,
                                    },
                                    file,
                                    indent=2,
                                )
                            print(
                                f"Current Success count: {success_count} / "
                                f"Failure count: {fail_count}"
                            )
                        self.ui_queue.put(
                            ("stats", success_count, fail_count)
                        )

                self.grasp_yes_event.clear()
                self.grasp_no_event.clear()
                self.paired_episode_event.clear()
                print(f"============== episode {episode} done =========================")

                if state == "exit" or self.exit_event.is_set():
                    break
                self._post_state("ready")

        except Exception as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            self._post_state("error")
        finally:
            print("Exiting teleoperation recording.")
            camera_preview = self.camera_preview
            self.camera_preview = None
            if camera_preview is not None:
                try:
                    camera_preview.close()
                except Exception as exc:
                    print(f"Failed to close camera preview: {exc}", file=sys.stderr)
            tactile_plotter = self.tactile_plotter
            self.tactile_plotter = None
            if tactile_plotter is not None:
                try:
                    tactile_plotter.close()
                except Exception as exc:
                    print(f"Failed to close tactile plotter: {exc}", file=sys.stderr)
            if cs is not None:
                try:
                    if getattr(cs, "save_path", None) is not None:
                        cs.stop()
                except Exception as exc:
                    print(f"Failed to stop active capture: {exc}", file=sys.stderr)
                try:
                    cs.end()
                except Exception as exc:
                    print(f"Failed to close capture devices: {exc}", file=sys.stderr)
            if pedal_state is not None:
                try:
                    pedal_state.close()
                except Exception as exc:
                    print(f"Failed to close pedal input: {exc}", file=sys.stderr)
            self.ui_queue.put(("done", self.exit_event.is_set()))


def main():
    args = parse_args()
    qt_app = QtWidgets.QApplication([])
    qt_app.setApplicationName("Robot Capture")
    qt_app.setFont(QtGui.QFont("Noto Sans CJK KR", 12))
    window = CaptureRobotGui(args)
    window.showMaximized()
    window.start_worker()
    try:
        return qt_app.exec()
    finally:
        if not window.exit_event.is_set():
            window._on_exit()
        if window.worker is not None:
            window.worker.join(timeout=15.0)
        window.restore_streams()


if __name__ == "__main__":
    raise SystemExit(main())
