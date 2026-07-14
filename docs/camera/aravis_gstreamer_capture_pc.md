# Aravis/GStreamer capture-PC migration

This guide replaces only the camera process on a capture PC.  The distributed
Paradex control topology does not change:

```text
Main PC: CaptureSession + UTG900E
   └─ ZMQ start/stop ──> capture PC: Aravis/GStreamer camera agent
                                  └─ GigE cameras, TriggerSource=Line0
```

The UTG900E stays connected to the main PC.  Do **not** install its udev rule,
PyVISA, PyUSB, or connect a trigger USB device on capture PCs.

## What the migration preserves

- `CaptureSession.start()` / `stop()` and the existing ports `5480`, `5481`,
  `5482` remain the control API.
- The old `raw/videos/{serial}.avi` layout is retained.  A capture PC writes
  to `~/captures1/<save_path>/videos/` and `~/captures2/<save_path>/videos/`
  exactly as the existing `CameraLoader` did.
- Agent boot (and `reload`) ForceIP-recovers and verifies every configured
  camera. Each agent responds `ready` to `start` only after every local camera
  was configured and its GStreamer pipeline was prepared. Only then does the
  main PC enable the UTG900E.
- On stop, the main PC disables the UTG first, waits for the existing drain
  interval, then the agent sends EOS and lets `avimux` finish the AVI index.

The intentionally changed part is the local implementation: PySpin frame
pulling and OpenCV `VideoWriter` are replaced with the native pipeline
`aravissrc -> bayer2rgb -> jpegenc -> avimux -> filesink`.

This migration deliberately covers the `CaptureSession` **video** path used by
the HRI capture scripts. The legacy remote `image`, `stream`, and `full`
utilities depend on PySpin-specific one-shot/shared-memory behavior; keep a
capture PC on `--backend pyspin` when using those utilities.

## 1. Install packages on every capture PC

Run this on a capture PC (Ubuntu 22.04 or later).  It installs the Aravis
runtime and the GStreamer elements used by the agent.

```bash
sudo apt update
sudo apt install -y \
  python3-gi python3-venv \
  gir1.2-aravis-0.8 aravis-tools \
  gir1.2-gstreamer-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  ffmpeg ethtool
```

Place this repository at the same absolute path on every capture PC (the
service template below uses `/opt/paradex`), then create the environment with
system packages visible to Python:

```bash
cd /opt/paradex
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[aravis]'
```

Check that the source element exists before trying the agent:

```bash
gst-inspect-1.0 aravissrc
arv-tool-0.8 discover
```

`arv-tool-0.8 discover` must list the serials assigned to this PC in
[`system/current/pc.json`](../../system/current/pc.json).  Gain and exposure
continue to come from [`system/current/camera.json`](../../system/current/camera.json).

## 2. Configure the camera NICs

Keep the management NIC separate.  Each physical camera NIC needs a permanent
`192.168.X.1/24` address (`X != 0`) and should normally carry one camera.  The
agent uses this convention to recover a camera that power-cycled to a
link-local address via GVCP ForceIP.

Example netplan stanza; substitute the real interface and subnet:

```yaml
network:
  version: 2
  ethernets:
    enp5s0:
      addresses: [192.168.11.1/24]
      mtu: 9216
```

Apply it only after validating the complete netplan configuration:

```bash
sudo netplan generate
sudo netplan apply
```

For jumbo frames, all of these must support the same MTU:

- capture-PC camera NIC: `mtu: 9216`
- camera: `GevSCPSPacketSize=9000` (the Paradex default)
- any switch port between them: jumbo frames enabled

If one link cannot do jumbo frames, set the NIC/camera path to normal MTU and
start the service with `PARADEX_GIGE_PACKET_SIZE=1400`.  Never leave a 9000
byte camera packet size on a 1500-byte path.

Do not run ParaOffice's `scripts/setup-flir.sh` wholesale on these existing
capture PCs: it regenerates `/etc/netplan/01-netcfg.yaml`, detects interfaces,
and installs a local UTG service, which is the wrong ownership model here.
It is useful as a reference for package and kernel settings only.

## 3. Tune the Linux receive path

Install a persistent receive-buffer setting:

```bash
sudo tee /etc/sysctl.d/99-gige-vision.conf >/dev/null <<'EOF'
net.core.rmem_max = 16777216
net.core.rmem_default = 16777216
EOF
sudo sysctl -p /etc/sysctl.d/99-gige-vision.conf
```

Inspect each camera NIC's supported RX ring size, then set its supported
maximum.  Use the number reported by your driver; it varies by NIC.

```bash
ethtool -g enp5s0
sudo ethtool -G enp5s0 rx <driver-supported-maximum>
```

The capture disks must also sustain the combined JPEG output of the cameras
assigned to the PC.  `queue_buffers=60` bounds the raw queue to roughly two
seconds at 30 fps, but it is a shock absorber—not a substitute for adequate
disk bandwidth.

## 4. Run the capture agent

First test interactively on each capture PC:

```bash
cd /opt/paradex
.venv/bin/python src/camera/server_daemon.py --backend aravis-gstreamer
```

The process must remain up and publish `READY` cameras without a UTG900E
attached locally.  The main PC can then run the existing capture application
unchanged.

For normal operation, install the included systemd unit.  The instance name
must match the Linux username/home directory used by that capture PC, because
the existing output paths are under that user's `~/captures1` and
`~/captures2` directories.

```bash
sudo install -m 644 systemd/paradex-camera-agent@.service \
  /etc/systemd/system/paradex-camera-agent@.service
sudo systemctl daemon-reload
sudo systemctl enable --now paradex-camera-agent@capture12
sudo systemctl status paradex-camera-agent@capture12
```

Use the actual account name instead of `capture12`.  Follow logs with:

```bash
journalctl -u paradex-camera-agent@capture12 -f
```

## 5. Verify one complete recording

1. Start every capture-PC agent and confirm the configured serials appear in
   `arv-tool-0.8 discover`.
2. Start one normal `CaptureSession` on the main PC.
3. Confirm each agent logs `All Aravis/GStreamer cameras READY` before the
   main PC starts the UTG.
4. Stop the session normally; do not kill the agent during AVI finalization.
5. On each capture PC, inspect an output file:

   ```bash
   ffprobe ~/captures1/<save_path>/videos/<serial>.avi
   ```

If startup fails before `ready`, the main PC now raises the capture-PC error
instead of enabling the trigger for a partial rig.  If an agent is still using
the legacy backend, start it with `--backend pyspin`; this is a safe rollback
while debugging deployment settings.

## Runtime configuration

The agent defaults are chosen to match the current Paradex PySpin setup.
Set these as systemd `Environment=` entries only when a rig differs:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PARADEX_GIGE_PACKET_SIZE` | `9000` | Use `1400` when jumbo frames are unavailable. |
| `PARADEX_CAMERA_WIDTH` | `2048` | Bayer frame width. |
| `PARADEX_CAMERA_HEIGHT` | `1536` | Bayer frame height. |
| `PARADEX_BAYER_FORMAT` | `rggb` | GStreamer Bayer caps format. |
| `PARADEX_JPEG_QUALITY` | `95` | MJPEG quality in the AVI file. |
| `PARADEX_GST_QUEUE_BUFFERS` | `60` | Bounded raw-frame queue depth. |

The camera's `Gain`, `ExposureTime`, `PixelFormat`, `GevSCPSPacketSize`,
frame-rate cap, and `Line0` trigger configuration are written and read back
by the agent before every recording start.
