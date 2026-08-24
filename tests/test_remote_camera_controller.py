import queue
import threading
import time

from paradex.io.camera_system import remote_camera_controller as controller_module


class FakeSocket:
    def __init__(self, slow_action=None, release=None):
        self.slow_action = slow_action
        self.release = release
        self.sent = []
        self.thread_ids = []
        self.current_command = None
        self.closed = False

    def setsockopt(self, _option, _value):
        self.thread_ids.append(threading.get_ident())

    def send_json(self, command):
        self.current_command = dict(command)
        self.sent.append(dict(command))
        self.thread_ids.append(threading.get_ident())

    def recv_json(self):
        self.thread_ids.append(threading.get_ident())
        if (
            self.current_command.get("action") == self.slow_action
            and self.release is not None
        ):
            assert self.release.wait(2.0)
        return {"status": "ok", "msg": "accepted"}

    def close(self):
        self.closed = True
        self.thread_ids.append(threading.get_ident())


def make_worker_controller(monkeypatch, sockets):
    controller = controller_module.remote_camera_controller.__new__(
        controller_module.remote_camera_controller
    )
    controller.name = "test_controller"
    controller.command_queues = {pc: queue.Queue() for pc in sockets}
    controller.command_sockets = {}
    controller.worker_threads = {}
    controller.worker_stop = threading.Event()
    controller._worker_ready = {pc: threading.Event() for pc in sockets}
    controller._worker_errors = {}
    controller._send_command_lock = threading.Lock()
    controller._heartbeat_lock = threading.Lock()
    controller._last_heartbeat_failures = {}
    controller.error_event = threading.Event()
    controller.HEARTBEAT_INTERVAL_SECONDS = 0.02
    controller.HEARTBEAT_RESPONSE_SECONDS = 0.2
    monkeypatch.setattr(
        controller,
        "_create_command_socket",
        lambda pc: sockets[pc],
    )

    for pc in sockets:
        thread = threading.Thread(
            target=controller._command_worker,
            args=(pc,),
            daemon=True,
        )
        controller.worker_threads[pc] = thread
        thread.start()
        assert controller._worker_ready[pc].wait(1.0)
    return controller


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_slow_pc_does_not_starve_fast_pc_heartbeat(monkeypatch):
    release_slow_pc = threading.Event()
    sockets = {
        "slow": FakeSocket("snapshot", release_slow_pc),
        "fast": FakeSocket(),
    }
    controller = make_worker_controller(monkeypatch, sockets)

    try:
        registered = controller.send_command(
            {"action": "register"},
            timeout_seconds=1.0,
        )
        assert all(response["status"] == "ok" for response in registered.values())

        result = {}
        caller = threading.Thread(
            target=lambda: result.update(
                controller.send_command(
                    {"action": "snapshot", "save_path": "step-0"},
                    timeout_seconds=1.0,
                )
            ),
            daemon=True,
        )
        caller.start()

        def fast_pc_heartbeat_followed_snapshot():
            actions = [command["action"] for command in sockets["fast"].sent]
            if "snapshot" not in actions:
                return False
            snapshot_index = actions.index("snapshot")
            return "heartbeat" in actions[snapshot_index + 1 :]

        assert wait_until(fast_pc_heartbeat_followed_snapshot)
        assert caller.is_alive(), "slow PC should still be blocking the caller"

        release_slow_pc.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        assert set(result) == {"slow", "fast"}
    finally:
        release_slow_pc.set()
        controller._stop_command_workers()


def test_each_command_socket_is_owned_by_one_worker_thread(monkeypatch):
    sockets = {"capture12": FakeSocket(), "capture13": FakeSocket()}
    controller = make_worker_controller(monkeypatch, sockets)

    try:
        controller.send_command({"action": "register"}, timeout_seconds=1.0)
        assert wait_until(
            lambda: all(
                any(command["action"] == "heartbeat" for command in socket.sent)
                for socket in sockets.values()
            )
        )
    finally:
        controller._stop_command_workers()

    calling_thread = threading.get_ident()
    owner_threads = []
    for socket in sockets.values():
        thread_ids = set(socket.thread_ids)
        assert len(thread_ids) == 1
        owner_threads.append(next(iter(thread_ids)))
    assert calling_thread not in owner_threads
    assert len(set(owner_threads)) == len(sockets)
