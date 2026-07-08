"""Reliable command channel (REQ/REP) between the main PC and capture PCs.

``CommandSender`` (main PC) fans a command out to every capture PC; each
``CommandReceiver`` (capture PC) fires a local :class:`threading.Event` for it and
replies. This is the control path — start/stop/save/exit — not the data path
(see :mod:`paradex.io.capture_pc.transport` for that).

Reliability: plain REQ/REP wedges if a peer dies mid-request (the REQ socket is
stuck until it receives a reply that never comes). ``CommandSender`` uses the
**lazy-pirate** pattern — on timeout it discards and rebuilds that PC's socket and
retries — so one unreachable/restarting PC can never hang the orchestrator.
"""

import threading
from typing import Dict, Optional, Any

import zmq

from paradex.utils.system import get_pc_list, get_pc_ip


class CommandSender:
    """Send commands to multiple PCs over REQ/REP, with per-PC retry.

    Args:
        pc_list: target PCs (default: ``get_pc_list()``).
        port:    REP port the receivers bind.
        timeout: per-attempt reply timeout in ms.
        retries: attempts per PC before giving up (socket is rebuilt between tries).
    """

    def __init__(self, pc_list: Optional[list] = None, port: int = 6890,
                 timeout: int = 60000, retries: int = 3):
        self.pc_list = list(pc_list or get_pc_list())
        self.port = port
        self.timeout = timeout
        self.retries = max(1, retries)

        self.context = zmq.Context.instance()
        self.ip = {pc: get_pc_ip(pc) for pc in self.pc_list}
        self.sockets: Dict[str, "zmq.Socket"] = {}
        for pc_name in self.pc_list:
            self.sockets[pc_name] = self._make_socket(pc_name)
            print(f"[{pc_name}] Connected to {self.ip[pc_name]}:{self.port}")

    def _make_socket(self, pc_name: str) -> "zmq.Socket":
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        socket.connect(f"tcp://{self.ip[pc_name]}:{self.port}")
        return socket

    def _reset_socket(self, pc_name: str) -> None:
        """Discard a wedged REQ socket and reconnect (lazy-pirate recovery)."""
        try:
            self.sockets[pc_name].close(linger=0)
        except Exception:
            pass
        self.sockets[pc_name] = self._make_socket(pc_name)

    def _send_to_pc(self, pc_name: str, cmd: str, wait: bool, cmd_info: dict) -> None:
        cmd_dict = {"command": cmd, "is_wait": wait, "info": cmd_info}
        for attempt in range(1, self.retries + 1):
            try:
                self.sockets[pc_name].send_json(cmd_dict)
                response = self.sockets[pc_name].recv_json()
                if response.get("state") == "error":
                    print(f"[{pc_name}] Error: {response.get('message')}")
                else:
                    print(f"[{pc_name}] {response.get('message')}")
                return
            except zmq.ZMQError as e:
                # timeout / peer gone: rebuild the socket and retry.
                self._reset_socket(pc_name)
                if attempt == self.retries:
                    print(f"[{pc_name}] '{cmd}' failed after {self.retries} tries: {e}")
                else:
                    print(f"[{pc_name}] '{cmd}' retry {attempt}/{self.retries} ({e})")

    def send_command(self, cmd: str, wait: bool = False, cmd_info: dict = None) -> None:
        """Send ``cmd`` to all PCs in parallel and wait for their replies."""
        cmd_info = cmd_info or {}
        threads = [
            threading.Thread(target=self._send_to_pc,
                             args=(pc_name, cmd, wait, cmd_info), daemon=True)
            for pc_name in self.sockets
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def end(self):
        self.send_command("exit")
        for socket in self.sockets.values():
            socket.close(linger=0)


class CommandReceiver:
    """Bind a REP port and fire a local Event for each known command.

    Args:
        event_dict: ``{command: threading.Event}`` — the event is ``set()`` when
                    its command arrives; ``info`` is stashed in ``event_info``.
        port:       REP port to bind.
    """

    def __init__(self, event_dict: Optional[Dict[str, threading.Event]] = None,
                 port: int = 6890):
        self.port = port
        self.event_dict = event_dict or {}
        self.event_info: Dict[str, Any] = {}

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        self.exit_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        print(f"[CommandReceiver] Listening on port {self.port}")
        self.start()

    def _recv_loop(self) -> None:
        while not self.exit_event.is_set():
            # poll so we never busy-spin and always reply exactly once per recv
            if not dict(self.poller.poll(timeout=100)):
                continue
            try:
                cmd_dict = self.socket.recv_json()
            except zmq.ZMQError:
                continue

            cmd = cmd_dict.get("command", "")
            is_wait = cmd_dict.get("is_wait", False)
            cmd_info = cmd_dict.get("info", {})

            if cmd in self.event_dict:
                try:
                    self.event_info[cmd] = cmd_info
                    self.event_dict[cmd].set()
                    if is_wait:
                        self.event_dict[cmd].wait()
                    response = {"state": "success",
                                "message": f"Command '{cmd}' executed"}
                except Exception as e:
                    response = {"state": "error", "message": f"Callback error: {e}"}
            else:
                response = {"state": "error", "message": f"Unknown command: {cmd}"}

            try:
                self.socket.send_json(response)
            except zmq.ZMQError as e:
                print(f"[CommandReceiver] reply failed: {e}")

    def start(self) -> None:
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        print("[CommandReceiver] Started")

    def end(self) -> None:
        self.exit_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.socket.close(linger=0)
        print("[CommandReceiver] Closed")
