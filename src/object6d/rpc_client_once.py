import argparse
import json
from pathlib import Path
from typing import Any, Dict

import zmq


def load_request(command: str, request_json: str | None) -> Dict[str, Any]:
    if request_json is None:
        return {"command": command}

    req_path = Path(request_json)
    if not req_path.exists():
        raise FileNotFoundError(f"request json not found: {req_path}")

    with req_path.open("r", encoding="utf-8") as f:
        req = json.load(f)

    if not isinstance(req, dict):
        raise ValueError("request json must be an object/dict")

    req.setdefault("command", command)
    return req


def send_once(addr: str, req: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)

    try:
        sock.connect(addr)
        sock.send_json(req)
        return sock.recv_json()
    finally:
        sock.close()
        ctx.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one ZMQ RPC request and receive response")
    parser.add_argument("--addr", default="tcp://192.168.0.14:5570", help="RPC server address")
    parser.add_argument("--command", choices=["ping", "infer", "reset"], default="ping")
    parser.add_argument(
        "--request_json",
        default=None,
        help="Path to request JSON file. If omitted, sends {'command': <command>}.",
    )
    parser.add_argument("--timeout_ms", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    req = load_request(args.command, args.request_json)
    res = send_once(args.addr, req, args.timeout_ms)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
