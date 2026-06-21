# CLAUDE.md — src/validate/data_sender

## Purpose
Minimal one-way telemetry smoke-test: `DataPublisher` (capture PC) → `DataCollector`
(main PC). Strips the command channel out of `command_sender/` to isolate the data path.

## Files
- `main.py` (MAIN PC):
  1. `run_script("python src/validate/data_sender/client.py")` — SSH launches client on capture PCs.
  2. `DataCollector().start()`.
  3. Infinite loop: `dc.get_data()` → `{pc_id: data}`, print each, `sleep(1)`.
  (Imports `CommandSender` but does not use it.)
- `client.py` (CAPTURE PC):
  - `DataPublisher(name="TestPublisher")`; loop forever `send_data({"value": time()-start_time})` every 0.5 s.

## paradex modules used
- `paradex.io.capture_pc.data_sender.{DataPublisher,DataCollector}`
- `paradex.io.capture_pc.ssh.run_script`
- (`paradex.io.capture_pc.command_sender.CommandSender` imported but unused in `main.py`)

## Data flow & IO
Capture → main only. Dict payload `{"value": float}` via `DataPublisher` →
`DataCollector.get_data()` keyed by PC id. No filesystem IO; no graceful shutdown (Ctrl-C).

## When working here
- Launch `main.py`; it spawns `client.py`. Don't run the client standalone.
- The unused `CommandSender` import in `main.py` is harmless leftover.

## Gotchas
- Both loops are unbounded `while True` — there is no exit/`end()` path (compare with
  `command_sender/` which has `q`/`exit`).
