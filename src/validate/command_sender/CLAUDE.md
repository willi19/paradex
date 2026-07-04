# CLAUDE.md — src/validate/command_sender

## Purpose
Round-trip test of command + data plumbing between main PC and capture PCs:
`CommandSender → CommandReceiver` (control) plus `DataPublisher → DataCollector`
(reverse telemetry) running at the same time.

## Files
- `stream_remote.py` (MAIN PC):
  1. `run_script("python src/validate/command_sender/stream_client.py")` — SSH launches client on capture PCs.
  2. `listen_keyboard({"c": start, "q": exit, "s": stop})`.
  3. `DataCollector().start()` and `CommandSender()`.
  4. Loop: on `c` → `cs.send_command("start")`; on `s` → `cs.send_command("stop")`;
     each tick prints `dc.get_data()` (`{pc_id: data}`). `q` → `dc.end()`, `cs.end()`.
  Uses `cur_state` guard so a command is only sent on transition.
- `stream_client.py` (CAPTURE PC):
  - `DataPublisher()` + `CommandReceiver({"start","exit","stop"})` (each mapped to a threading Event).
  - Loop: if `start` set and not `stop`, `dp.send_data({"value": time()-start_time})` every 0.1 s;
    on `stop` prints "Stopped" and clears both `start`/`stop` events.

## paradex modules used
- `paradex.io.capture_pc.command_sender.{CommandSender,CommandReceiver}`
- `paradex.io.capture_pc.data_sender.{DataPublisher,DataCollector}`
- `paradex.io.capture_pc.ssh.run_script`
- `paradex.utils.keyboard_listener.listen_keyboard`

## Data flow & IO
Main → capture: string commands (`start`/`stop`/`exit`) over `CommandSender`/`CommandReceiver`.
Capture → main: dict payloads via `DataPublisher` → `DataCollector.get_data()` keyed by PC id.
No filesystem IO. Network targets come from the capture-PC config in `system/current/`.

## When working here
- Always launch `stream_remote.py`; it spawns the client. Don't run `stream_client.py` by hand.
- `exit`/`q` is handled by `CommandReceiver`'s event but the client loop only checks `exit_event`;
  shutting down the main side via `q` calls `cs.end()`/`dc.end()`.

## Gotchas
- `data_sender/` is the same idea minus the command channel — keep the two consistent if editing.
- Client only sends data while `start` event is set and `stop` is clear; forgetting `c` looks like "no data".
