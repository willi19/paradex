import socket
import struct

from paradex.io.robot_controller.deprecated import inspire_controller_ip


class FakeSocket:
    def __init__(self):
        self.timeout = None
        self.socket_option = None
        self.address = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def setsockopt(self, level, option, value):
        self.socket_option = (level, option, value)

    def connect(self, address):
        self.address = address

    def close(self):
        self.closed = True


class FakeModbusTcpClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.timeout = 3
        self.socket = None

    def close(self):
        if self.socket is not None:
            self.socket.close()
        self.socket = None


def test_open_modbus_binds_tcp_socket_to_requested_interface(monkeypatch):
    fake_socket = FakeSocket()
    monkeypatch.setattr(
        inspire_controller_ip,
        "ModbusTcpClient",
        FakeModbusTcpClient,
    )
    monkeypatch.setattr(
        inspire_controller_ip.socket,
        "if_nametoindex",
        lambda interface: 17 if interface == "enp8s0f2" else 0,
    )
    monkeypatch.setattr(
        inspire_controller_ip.socket,
        "socket",
        lambda *_args: fake_socket,
    )

    controller = inspire_controller_ip.InspireControllerIP.__new__(
        inspire_controller_ip.InspireControllerIP
    )
    controller.ip = "192.168.11.210"
    controller.port = 6000
    controller.interface = "enp8s0f2"

    controller.open_modbus()

    assert controller.inspire_node.socket is fake_socket
    assert fake_socket.timeout == 3
    assert fake_socket.socket_option == (
        socket.IPPROTO_IP,
        50,
        struct.pack("!I", 17),
    )
    assert fake_socket.address == ("192.168.11.210", 6000)
