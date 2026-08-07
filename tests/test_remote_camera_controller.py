import threading

from paradex.io.camera_system import remote_camera_controller as controller_module


class FakeSocket:
    def __init__(self, response):
        self.response = response
        self.sent = []
        self.thread_ids = []

    def send_json(self, command):
        self.sent.append(command)
        self.thread_ids.append(threading.get_ident())

    def recv_json(self, flags=0):
        self.thread_ids.append(threading.get_ident())
        return self.response


class FakePoller:
    def __init__(self):
        self.sockets = set()

    def register(self, socket, _event):
        self.sockets.add(socket)

    def unregister(self, socket):
        self.sockets.remove(socket)

    def poll(self, _timeout_ms):
        return [(socket, controller_module.zmq.POLLIN) for socket in self.sockets]


def test_send_command_keeps_zmq_sockets_on_calling_thread(monkeypatch):
    monkeypatch.setattr(controller_module.zmq, "Poller", FakePoller)
    calling_thread = threading.get_ident()
    sockets = {
        "capture12": FakeSocket({"status": "ok", "msg": "registered"}),
        "capture13": FakeSocket({"status": "ok", "msg": "registered"}),
    }
    controller = controller_module.remote_camera_controller.__new__(
        controller_module.remote_camera_controller
    )
    controller.name = "test_controller"
    controller.command_sockets = sockets

    responses = controller.send_command({"action": "register"})

    assert responses == {
        "capture12": {"status": "ok", "msg": "registered"},
        "capture13": {"status": "ok", "msg": "registered"},
    }
    for socket in sockets.values():
        assert socket.sent == [
            {"action": "register", "controller_name": "test_controller"}
        ]
        assert socket.thread_ids == [calling_thread, calling_thread]
