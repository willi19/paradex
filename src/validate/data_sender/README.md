# Data Sender Validation

Tests the one-way telemetry path from capture PCs to the main PC: capture PCs run a
`DataPublisher`, the main PC runs a `DataCollector` and prints whatever each PC sends.
No command channel (unlike [`../command_sender/`](../command_sender)).

## Scripts
| File | Purpose |
|------|---------|
| `main.py` | **Main PC.** SSH-launches `client.py` on capture PCs, starts a `DataCollector`, prints collected data per PC once a second. |
| `client.py` | **Capture PC** (launched by `main.py`). A `DataPublisher(name="TestPublisher")` that sends an increasing `value` every 0.5 s forever. |

## Usage
Run only the main-PC script; it launches the client over SSH.

```bash
python src/validate/data_sender/main.py
```

No keyboard controls; stop with `Ctrl-C` (both scripts loop unconditionally).

## What it validates
A passing run prints `PC ID: <id>, Data: {'value': <increasing seconds>}` lines once
per second on the main PC, one entry per reachable capture PC. This confirms the
publisher starts on the remote PC, the collector discovers it, and dict payloads flow
over the network. If a PC never appears, its SSH launch or publisher failed.

## Related
- [`paradex/io/capture_pc/data_sender.py`](../../../paradex/io/capture_pc/data_sender.py) — `DataPublisher`, `DataCollector`
- [`paradex/io/capture_pc/ssh.py`](../../../paradex/io/capture_pc/ssh.py) — `run_script`
- [`../command_sender/`](../command_sender) — same data path plus a command channel and keyboard control
