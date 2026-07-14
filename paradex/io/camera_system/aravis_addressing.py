"""GigE Vision camera discovery and ForceIP recovery for capture PCs.

The capture agents keep a fixed ``X.Y.Z.1/24`` address on each camera-facing
NIC. GigE cameras can lose their persistent address after a power cycle and
fall back to a link-local address. Aravis can discover such cameras but cannot
control them until their address is moved back onto the NIC subnet, so this
module runs the GVCP ForceIP recovery before a recording agent accepts
commands.

All SDK imports are deliberately lazy.  This keeps the main Paradex package
importable on the main PC and on development machines without Aravis.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


log = logging.getLogger(__name__)

GVCP_PORT = 3956
FORCEIP_OPCODE = 0x0004
FORCEIP_PAYLOAD_LEN = 56
DISCOVERY_SETTLE_SECONDS = 3.0
FIRST_CAMERA_HOST = 100


class CameraAddressingError(RuntimeError):
    """The capture PC cannot place every configured camera on a NIC subnet."""


@dataclass(frozen=True)
class CameraRecord:
    serial: str
    device_id: str
    ip: str
    mac: str


@dataclass(frozen=True)
class NicSubnet:
    name: str
    host_ip: str
    network: ipaddress.IPv4Network


def gvcp_forceip_packet(
    mac: str,
    ip: str,
    mask: str = "255.255.255.0",
    gateway: str = "0.0.0.0",
    request_id: int = 1,
) -> bytes:
    """Build the 64-byte GigE Vision ``FORCEIP_CMD`` packet."""

    mac_bytes = bytes(int(part, 16) for part in mac.replace("-", ":").split(":"))
    if len(mac_bytes) != 6:
        raise ValueError("MAC address must contain exactly six octets: {!r}".format(mac))

    packet = bytearray(64)
    packet[0] = 0x42  # GigE Vision command packet
    packet[1] = 0x01  # acknowledge required
    struct.pack_into(">H", packet, 2, FORCEIP_OPCODE)
    struct.pack_into(">H", packet, 4, FORCEIP_PAYLOAD_LEN)
    struct.pack_into(">H", packet, 6, request_id)
    packet[10:16] = mac_bytes
    packet[28:32] = socket.inet_aton(ip)
    packet[44:48] = socket.inet_aton(mask)
    packet[60:64] = socket.inet_aton(gateway)
    return bytes(packet)


def _load_aravis():
    try:
        import gi

        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis
    except (ImportError, ValueError) as exc:
        raise CameraAddressingError(
            "Aravis is unavailable. Install gir1.2-aravis-0.8 and run the "
            "capture-agent Python environment with --system-site-packages."
        ) from exc
    return Aravis


def _load_iproute():
    try:
        from pyroute2 import IPRoute
    except ImportError as exc:
        raise CameraAddressingError(
            "pyroute2 is unavailable. Install the Paradex [aravis] extra on this capture PC."
        ) from exc
    return IPRoute


def _camera_subnet(
    name: str,
    host_ip: str,
    prefixlen: int,
    *,
    explicitly_selected: bool,
) -> Optional[NicSubnet]:
    """Build the camera's intended /24 subnet from a NIC address.

    The deployed Paradex rigs historically use ``11.0.<nic>.1``. Some hosts
    use a broad legacy netmask even though each physical camera link is a
    separate /24. Explicitly selected NICs and that legacy layout are thus
    interpreted as their containing /24. Other inferred addresses stay strict
    to avoid accepting Docker, VPN, or lab interfaces as camera NICs.
    """

    try:
        address = ipaddress.IPv4Address(host_ip)
    except ipaddress.AddressValueError:
        return None
    if address.is_loopback or address.is_link_local or int(address.packed[-1]) != 1:
        return None

    if prefixlen == 24:
        network = ipaddress.ip_network("{}/24".format(address), strict=False)
    elif explicitly_selected or (address.packed[0] == 11 and address.packed[1] == 0):
        network = ipaddress.ip_network("{}/24".format(address), strict=False)
        log.warning(
            "Camera NIC %s is configured as %s/%s; using %s for its dedicated "
            "camera link. Configure it as /24 when practical.",
            name,
            address,
            prefixlen,
            network,
        )
    else:
        return None

    return NicSubnet(name=name, host_ip=str(address), network=network)


def _is_deployed_11_camera_address(host_ip: str) -> bool:
    try:
        address = ipaddress.IPv4Address(host_ip)
    except ipaddress.AddressValueError:
        return False
    return address.packed[:2] == bytes((11, 0)) and address.packed[-1] == 1


def discover_camera_nics() -> List[NicSubnet]:
    """Return dedicated camera NICs without imposing an address range.

    A camera NIC is normally an IPv4 ``/24`` address ending in ``.1`` that is
    not on the host's default route. The existing Paradex ``11.0.<nic>.1``
    layout is also accepted when it has a broad legacy netmask; each physical
    link is treated as its own /24. Set ``PARADEX_CAMERA_NICS=nic0,nic1`` to
    select NICs explicitly and avoid Docker/VPN/lab-network inference.
    """

    IPRoute = _load_iproute()
    requested_names = {
        name.strip() for name in os.getenv("PARADEX_CAMERA_NICS", "").split(",") if name.strip()
    }
    nics: List[NicSubnet] = []
    with IPRoute() as ipr:
        link_names = {
            link["index"]: link.get_attr("IFLA_IFNAME") for link in ipr.get_links()
        }
        default_route_indices = {
            route.get_attr("RTA_OIF")
            for route in ipr.get_routes(family=socket.AF_INET, dst_len=0)
            if route.get_attr("RTA_OIF") is not None
        }
        for address in ipr.get_addr(family=socket.AF_INET):
            name = link_names.get(address["index"])
            if not name or (requested_names and name not in requested_names):
                continue
            host_ip = address.get_attr("IFA_ADDRESS")
            if not host_ip:
                continue
            explicitly_selected = name in requested_names
            # The deployed rigs deliberately use 11.0.X.1 for camera links.
            # Trust that convention even if NetworkManager has incorrectly
            # marked one of those links as default-route capable.
            is_deployed_camera_nic = _is_deployed_11_camera_address(host_ip)
            if (
                address["index"] in default_route_indices
                and not explicitly_selected
                and not is_deployed_camera_nic
            ):
                continue
            subnet = _camera_subnet(
                name,
                host_ip,
                address["prefixlen"],
                explicitly_selected=explicitly_selected,
            )
            if subnet is not None:
                nics.append(subnet)
    return sorted(nics, key=lambda nic: nic.name)


class CameraAddressing:
    """Reconcile configured cameras into the capture PC's camera subnets."""

    def __init__(self, nic_subnets: Optional[Iterable[NicSubnet]] = None) -> None:
        self._nic_subnets = list(nic_subnets) if nic_subnets is not None else None
        self._seen: Dict[str, CameraRecord] = {}
        self._mac_to_nic: Dict[str, NicSubnet] = {}

    @property
    def nic_subnets(self) -> List[NicSubnet]:
        if self._nic_subnets is None:
            self._nic_subnets = discover_camera_nics()
        return list(self._nic_subnets)

    @property
    def seen(self) -> Dict[str, CameraRecord]:
        return dict(self._seen)

    def discover(self) -> Dict[str, CameraRecord]:
        if not self.nic_subnets:
            raise CameraAddressingError(
                "No camera-facing non-default IPv4 /24 NIC ending in .1 was found. "
                "Configure PARADEX_CAMERA_NICS explicitly or configure the capture NICs before starting the agent."
            )

        Aravis = _load_aravis()
        Aravis.update_device_list()
        records: Dict[str, CameraRecord] = {}
        for index in range(Aravis.get_n_devices()):
            serial = Aravis.get_device_serial_nbr(index)
            if not serial:
                continue
            record = CameraRecord(
                serial=str(serial),
                device_id=str(Aravis.get_device_id(index)),
                ip=str(Aravis.get_device_address(index)),
                mac=str(Aravis.get_device_physical_id(index)),
            )
            records[record.serial] = record
        self._seen = records
        self._snapshot_neighbor_table()
        return self.seen

    def _snapshot_neighbor_table(self) -> None:
        """Map each discovered camera MAC to the NIC on which it replied."""

        self._mac_to_nic.clear()
        known = {
            record.mac.lower().replace("-", ":"): record.serial
            for record in self._seen.values()
        }
        if not known:
            return

        try:
            IPRoute = _load_iproute()
            with IPRoute() as ipr:
                link_names = {
                    link["index"]: link.get_attr("IFLA_IFNAME") for link in ipr.get_links()
                }
                subnets = {nic.name: nic for nic in self.nic_subnets}
                for neighbor in ipr.get_neighbours(family=socket.AF_INET):
                    mac = (neighbor.get_attr("NDA_LLADDR") or "").lower()
                    name = link_names.get(neighbor["ifindex"])
                    if mac in known and name in subnets:
                        self._mac_to_nic[mac] = subnets[name]
        except Exception as exc:  # Discovery still works; use deterministic fallback below.
            log.warning("Could not read camera NIC neighbor entries: %s", exc)

    def _is_aligned(self, record: CameraRecord) -> bool:
        try:
            address = ipaddress.IPv4Address(record.ip)
        except ValueError:
            return False
        return any(address in nic.network for nic in self.nic_subnets)

    def _plan_force_ips(self, serials: Optional[Iterable[str]] = None) -> Dict[str, tuple[NicSubnet, str]]:
        """Choose a physical NIC and unique ``.100+`` address per camera."""

        nics = self.nic_subnets
        if not nics:
            return {}
        selected = set(str(serial) for serial in serials) if serials is not None else set(self._seen)

        used_addresses: Set[ipaddress.IPv4Address] = set()
        used_nics: Set[str] = set()
        for record in self._seen.values():
            if record.serial not in selected:
                continue
            try:
                address = ipaddress.IPv4Address(record.ip)
            except ValueError:
                continue
            for nic in nics:
                if address in nic.network:
                    used_addresses.add(address)
                    used_nics.add(nic.name)
                    break

        plan: Dict[str, tuple[NicSubnet, str]] = {}
        fallback_index = 0
        for record in sorted(self._seen.values(), key=lambda item: item.serial):
            if record.serial not in selected:
                continue
            if self._is_aligned(record):
                continue

            nic = self._mac_to_nic.get(record.mac.lower().replace("-", ":"))
            if nic is None:
                free = [candidate for candidate in nics if candidate.name not in used_nics]
                nic = (free or nics)[fallback_index % len(free or nics)]
                fallback_index += 1
                log.warning(
                    "No ARP NIC mapping for camera %s; assigning it to %s",
                    record.serial,
                    nic.name,
                )
            used_nics.add(nic.name)

            host = FIRST_CAMERA_HOST
            while host < 255:
                candidate = ipaddress.IPv4Address(int(nic.network.network_address) + host)
                if candidate not in used_addresses:
                    used_addresses.add(candidate)
                    plan[record.serial] = (nic, str(candidate))
                    break
                host += 1
            else:
                raise CameraAddressingError("No free camera IP remains in {}".format(nic.network))
        return plan

    def force_ip(self, record: CameraRecord, nic: NicSubnet, target_ip: str) -> None:
        """Send a directed broadcast ForceIP command through one camera NIC."""

        packet = gvcp_forceip_packet(record.mac, target_ip)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # Binding the source IP is sufficient to select the NIC and avoids
            # SO_BINDTODEVICE/CAP_NET_RAW requirements in a normal systemd unit.
            sock.bind((nic.host_ip, 0))
            sock.sendto(packet, (str(nic.network.broadcast_address), GVCP_PORT))

    def _verify(self, record: CameraRecord) -> bool:
        Aravis = _load_aravis()
        try:
            camera = Aravis.Camera.new(record.device_id)
            if camera is None:
                return False
            device = camera.get_device()
            device.get_integer_feature_value("GevSCPSPacketSize")
            del device
            del camera
            return True
        except Exception as exc:
            log.warning("Camera %s at %s is not addressable: %s", record.serial, record.ip, exc)
            return False

    def reconcile(self, expected_serials: Iterable[str]) -> List[str]:
        """Discover, recover addresses, and verify all expected cameras.

        A missing configured camera is a hard startup error: responding READY
        with only a subset would let the main PC begin hardware triggering a
        partial rig.
        """

        expected = [str(serial) for serial in expected_serials]
        self.discover()
        missing = sorted(set(expected) - set(self._seen))
        if missing:
            raise CameraAddressingError("Configured cameras were not discovered: {}".format(missing))

        plan = self._plan_force_ips(expected)
        if plan:
            for serial, (nic, target_ip) in plan.items():
                record = self._seen[serial]
                log.info("ForceIP camera %s: %s -> %s via %s", serial, record.ip, target_ip, nic.name)
                self.force_ip(record, nic, target_ip)
            time.sleep(DISCOVERY_SETTLE_SECONDS)
            self.discover()

        missing = sorted(set(expected) - set(self._seen))
        if missing:
            raise CameraAddressingError("Configured cameras disappeared after ForceIP: {}".format(missing))

        failed = [serial for serial in expected if not self._verify(self._seen[serial])]
        if failed:
            raise CameraAddressingError("Configured cameras are not controllable: {}".format(failed))
        return expected
