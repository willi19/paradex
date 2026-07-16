"""GigE Vision camera discovery and ForceIP recovery for capture PCs.

The capture agents keep a fixed ``X.Y.Z.1/24`` address on each camera-facing
NIC. GigE cameras can lose their persistent address after a power cycle and
fall back to a link-local address. This module recovers camera addresses before
a recording agent accepts commands. Reachable cameras use Aravis' official
persistent-IP API and are reset so the address takes effect immediately. A raw
GVCP ForceIP packet remains as a fallback for cameras that cannot be opened.

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
DISCOVERY_OPCODE = 0x0002
DISCOVERY_ACK_OPCODE = 0x0003
DISCOVERY_REQUEST_ID = 0xFFFF
DISCOVERY_DATA_SIZE = 0xF8
DISCOVERY_MAC_OFFSET = 0x08
DISCOVERY_IP_OFFSET = 0x24
DISCOVERY_VENDOR_OFFSET = 0x48
DISCOVERY_VENDOR_SIZE = 32
DISCOVERY_MODEL_OFFSET = 0x68
DISCOVERY_MODEL_SIZE = 32
DISCOVERY_SERIAL_OFFSET = 0xD8
DISCOVERY_SERIAL_SIZE = 16
RAW_DISCOVERY_TIMEOUT_SECONDS = 0.5
FORCEIP_OPCODE = 0x0004
FORCEIP_PAYLOAD_LEN = 56
# FLIR DeviceReset performs a full camera reboot. Give all cameras enough time
# to rejoin discovery before treating the address recovery as failed.
DISCOVERY_SETTLE_SECONDS = 8.0
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


def gvcp_discovery_packet(request_id: int = DISCOVERY_REQUEST_ID) -> bytes:
    """Build an 8-byte GigE Vision ``DISCOVERY_CMD`` packet."""

    return struct.pack(">BBHHH", 0x42, 0x01, DISCOVERY_OPCODE, 0, request_id)


def _gvcp_text(data: bytes, offset: int, size: int) -> str:
    return data[offset : offset + size].split(b"\0", 1)[0].decode("ascii", "replace").strip()


def parse_gvcp_discovery_ack(packet: bytes) -> CameraRecord:
    """Parse a GVCP discovery acknowledgement without opening the camera."""

    if len(packet) < 8 + DISCOVERY_DATA_SIZE:
        raise ValueError("GVCP discovery acknowledgement is truncated")
    _status, opcode, payload_size, request_id = struct.unpack_from(">HHHH", packet, 0)
    if opcode != DISCOVERY_ACK_OPCODE or request_id != DISCOVERY_REQUEST_ID:
        raise ValueError("Packet is not a GVCP discovery acknowledgement")
    if payload_size < DISCOVERY_DATA_SIZE or len(packet) < 8 + payload_size:
        raise ValueError("GVCP discovery acknowledgement has an invalid payload size")

    data = packet[8 : 8 + payload_size]
    mac_bytes = data[DISCOVERY_MAC_OFFSET + 2 : DISCOVERY_MAC_OFFSET + 8]
    mac = ":".join("{:02x}".format(octet) for octet in mac_bytes)
    serial = _gvcp_text(data, DISCOVERY_SERIAL_OFFSET, DISCOVERY_SERIAL_SIZE) or mac
    vendor = _gvcp_text(data, DISCOVERY_VENDOR_OFFSET, DISCOVERY_VENDOR_SIZE)
    model = _gvcp_text(data, DISCOVERY_MODEL_OFFSET, DISCOVERY_MODEL_SIZE)
    current_ip = socket.inet_ntoa(data[DISCOVERY_IP_OFFSET : DISCOVERY_IP_OFFSET + 4])
    label = "-".join(part for part in (vendor, model, serial) if part)
    return CameraRecord(
        serial=serial,
        device_id="raw-gvcp:{}".format(label or mac),
        ip=current_ip,
        mac=mac,
    )


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

        raw_records, raw_mac_to_nic = self._raw_gvcp_discover()
        for serial, record in raw_records.items():
            # Preserve Aravis device IDs when its high-level discovery worked.
            records.setdefault(serial, record)
        self._seen = records
        self._mac_to_nic.update(raw_mac_to_nic)
        return self.seen

    def _raw_gvcp_discover(self) -> tuple[Dict[str, CameraRecord], Dict[str, NicSubnet]]:
        """Discover cameras from GVCP ACKs even when Aravis drops them.

        A wildcard socket receives limited-broadcast replies that are not
        delivered to Aravis 0.8.20 sockets bound to a specific ``11.0.X.1``
        address. ``IP_PKTINFO`` supplies the ingress interface, so ForceIP can
        be sent back through the exact physical camera link.
        """

        packet_info = getattr(socket, "IP_PKTINFO", None)
        if packet_info is None or not hasattr(socket.socket, "sendmsg"):
            log.warning("Raw GVCP discovery is unavailable: IP_PKTINFO/sendmsg is unsupported")
            return {}, {}

        nic_by_index: Dict[int, NicSubnet] = {}
        for nic in self.nic_subnets:
            try:
                nic_by_index[socket.if_nametoindex(nic.name)] = nic
            except OSError as exc:
                log.warning("Raw GVCP discovery skipped unavailable NIC %s: %s", nic.name, exc)
        records: Dict[str, CameraRecord] = {}
        mac_to_nic: Dict[str, NicSubnet] = {}
        discover_packet = gvcp_discovery_packet()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.IPPROTO_IP, packet_info, 1)
            sock.bind(("", 0))

            for ifindex, nic in nic_by_index.items():
                pktinfo = struct.pack(
                    "I4s4s",
                    ifindex,
                    socket.inet_aton(nic.host_ip),
                    b"\0" * 4,
                )
                try:
                    sock.sendmsg(
                        [discover_packet],
                        [(socket.IPPROTO_IP, packet_info, pktinfo)],
                        0,
                        ("255.255.255.255", GVCP_PORT),
                    )
                except OSError as exc:
                    log.warning("Raw GVCP discovery send failed on %s: %s", nic.name, exc)

            deadline = time.monotonic() + RAW_DISCOVERY_TIMEOUT_SECONDS
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(remaining)
                try:
                    payload, ancillary, _flags, _source = sock.recvmsg(65535, 256)
                except socket.timeout:
                    break

                ingress_index = None
                for level, kind, data in ancillary:
                    if level == socket.IPPROTO_IP and kind == packet_info and len(data) >= 4:
                        ingress_index = struct.unpack_from("I", data)[0]
                        break
                nic = nic_by_index.get(ingress_index)
                if nic is None:
                    continue
                try:
                    record = parse_gvcp_discovery_ack(payload)
                except ValueError:
                    continue
                records[record.serial] = record
                mac_to_nic[record.mac.lower().replace("-", ":")] = nic
                log.info(
                    "Raw GVCP discovery found camera %s at %s via %s (%s)",
                    record.serial,
                    record.ip,
                    nic.name,
                    record.mac,
                )

        return records, mac_to_nic

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

                # A recovery setup may carry one temporary link-local address
                # per camera NIC. Matching the camera's current address to
                # those networks gives an unambiguous physical NIC mapping,
                # even before the kernel neighbour table has been populated.
                address_networks = []
                for address in ipr.get_addr(family=socket.AF_INET):
                    name = link_names.get(address["index"])
                    host_ip = address.get_attr("IFA_ADDRESS")
                    if name not in subnets or not host_ip:
                        continue
                    try:
                        network = ipaddress.ip_network(
                            "{}/{}".format(host_ip, address["prefixlen"]),
                            strict=False,
                        )
                    except ValueError:
                        continue
                    address_networks.append((name, network))

                for record in self._seen.values():
                    try:
                        camera_ip = ipaddress.IPv4Address(record.ip)
                    except ValueError:
                        continue
                    for name, network in address_networks:
                        if camera_ip in network:
                            mac = record.mac.lower().replace("-", ":")
                            self._mac_to_nic[mac] = subnets[name]
                            break

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

    def _set_persistent_ip(self, record: CameraRecord, target_ip: str) -> None:
        """Configure and immediately apply a reachable camera's target IP."""

        Aravis = _load_aravis()
        camera = Aravis.Camera.new(record.device_id)
        if camera is None:
            raise CameraAddressingError("Could not open camera {}".format(record.serial))

        set_persistent = getattr(camera, "gv_set_persistent_ip_from_string", None)
        set_mode = getattr(camera, "gv_set_ip_configuration_mode", None)
        if set_persistent is None or set_mode is None:
            raise CameraAddressingError(
                "Installed Aravis lacks the persistent-IP API (requires Aravis >= 0.8.22)"
            )

        set_persistent(target_ip, "255.255.255.0", "0.0.0.0")
        set_mode(Aravis.GvIpConfigurationMode.PERSISTENT_IP)
        # FLIR applies the selected persistent address after DeviceReset. The
        # command acknowledgement is sent before the control connection drops.
        camera.execute_command("DeviceReset")

    def _send_raw_force_ip(self, record: CameraRecord, nic: NicSubnet, target_ip: str) -> None:
        """Send a raw GVCP ForceIP command when the camera cannot be opened."""

        packet = gvcp_forceip_packet(record.mac, target_ip)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind((nic.host_ip, 0))
            # A ForceIP target is, by definition, not necessarily on the
            # target subnet yet. Use the limited broadcast instead of that
            # subnet's directed broadcast.
            sock.sendto(packet, ("255.255.255.255", GVCP_PORT))

    def force_ip(self, record: CameraRecord, nic: NicSubnet, target_ip: str) -> None:
        """Move one camera to its target subnet before capture starts."""

        if record.device_id.startswith("raw-gvcp:"):
            self._send_raw_force_ip(record, nic, target_ip)
            return

        try:
            self._set_persistent_ip(record, target_ip)
            log.info(
                "Configured persistent IP for camera %s: %s; device reset requested",
                record.serial,
                target_ip,
            )
        except Exception as exc:
            log.warning(
                "Aravis persistent-IP setup failed for camera %s; falling back to raw ForceIP: %s",
                record.serial,
                exc,
            )
            self._send_raw_force_ip(record, nic, target_ip)

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

    def reconcile(
        self,
        expected_serials: Iterable[str],
        allow_partial: bool = False,
    ) -> List[str]:
        """Discover, recover addresses, and return the usable configured cameras.

        Strict callers retain the original all-or-nothing behavior. Camera
        agents may opt into ``allow_partial`` so one disconnected camera does
        not prevent the remaining local cameras from recording.
        """

        expected = [str(serial) for serial in expected_serials]
        self.discover()
        missing = sorted(set(expected) - set(self._seen))
        if missing:
            if not allow_partial:
                raise CameraAddressingError(
                    "Configured cameras were not discovered: {}".format(missing)
                )
            log.warning(
                "Configured cameras were not discovered and will be skipped: %s",
                missing,
            )

        usable = [serial for serial in expected if serial in self._seen]
        if not usable:
            raise CameraAddressingError(
                "No configured cameras were discovered; unavailable cameras: {}".format(missing)
            )

        plan = self._plan_force_ips(usable)
        if plan:
            for serial, (nic, target_ip) in plan.items():
                record = self._seen[serial]
                log.info("ForceIP camera %s: %s -> %s via %s", serial, record.ip, target_ip, nic.name)
                self.force_ip(record, nic, target_ip)
            time.sleep(DISCOVERY_SETTLE_SECONDS)
            self.discover()

        disappeared = sorted(set(usable) - set(self._seen))
        if disappeared:
            if not allow_partial:
                raise CameraAddressingError(
                    "Configured cameras disappeared after ForceIP: {}".format(disappeared)
                )
            log.warning(
                "Configured cameras disappeared after ForceIP and will be skipped: %s",
                disappeared,
            )
            usable = [serial for serial in usable if serial in self._seen]

        misaligned = [serial for serial in usable if not self._is_aligned(self._seen[serial])]
        if misaligned:
            details = {serial: self._seen[serial].ip for serial in misaligned}
            if not allow_partial:
                raise CameraAddressingError(
                    "Camera IP recovery failed; cameras remain outside capture NIC subnets: {}".format(
                        details
                    )
                )
            log.warning(
                "Camera IP recovery failed; cameras remain outside capture NIC subnets and will be skipped: %s",
                details,
            )
            usable = [serial for serial in usable if serial not in misaligned]

        failed = [serial for serial in usable if not self._verify(self._seen[serial])]
        if failed:
            if not allow_partial:
                raise CameraAddressingError(
                    "Configured cameras are not controllable: {}".format(failed)
                )
            log.warning(
                "Configured cameras are not controllable and will be skipped: %s",
                failed,
            )
            usable = [serial for serial in usable if serial not in failed]

        if not usable:
            raise CameraAddressingError("No configured cameras are usable")
        return usable
