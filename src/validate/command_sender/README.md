# Command Sender Validation

Exercises the full main-PC ↔ capture-PC control loop: the main PC sends
`start`/`stop`/`exit` commands to capture PCs and simultaneously collects a data
stream back from them. Confirms both `CommandSender`/`CommandReceiver` and the
`DataCollector`/`DataPublisher` paths work together.

## Scripts
| File | Purpose |
|------|---------|
| `stream_remote.py` | **Main PC.** SSH-launches `stream_client.py` on capture PCs, sends commands from the keyboard, and prints the data stream collected from each PC. |
| `stream_client.py` | **Capture PC** (launched by `stream_remote.py`). Receives commands; while `start`-ed (and not `stop`-ed) it publishes a running elapsed-time value. |

## Usage
Run only the main-PC script — it starts the capture-PC side over SSH.

```bash
python src/validate/command_sender/stream_remote.py
```

Keyboard controls (main PC, via `listen_keyboard`):
- `c` — send `start`
- `s` — send `stop`
- `q` — exit (tears down `DataCollector` and `CommandSender`)

## What it validates
A passing run: after pressing `c`, the main PC prints `start` and a stream of
`PC ID: <id>, Data: {'value': <increasing seconds>}` lines, one per capture PC.
Pressing `s` halts the values (client prints `Stopped`); `c` resumes from a reset.
`q` shuts both sides down cleanly. This proves command delivery, the receiver's
event toggling, and reverse data collection all work over the network.

## Related
- [`paradex/io/capture_pc/command_sender.py`](../../../paradex/io/capture_pc/command_sender.py) — `CommandSender`, `CommandReceiver`
- [`paradex/io/capture_pc/data_sender.py`](../../../paradex/io/capture_pc/data_sender.py) — `DataPublisher`, `DataCollector`
- [`paradex/io/capture_pc/ssh.py`](../../../paradex/io/capture_pc/ssh.py) — `run_script`
- [`paradex/utils/keyboard_listener.py`](../../../paradex/utils/keyboard_listener.py) — `listen_keyboard`
- [`../data_sender/`](../data_sender) — the data-only (no command) variant
