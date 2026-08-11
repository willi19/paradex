import argparse
import pprint
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


def summarize(x):
    if isinstance(x, np.ndarray):
        arr = np.asarray(x)
        out = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
        if arr.size and np.issubdtype(arr.dtype, np.number):
            out.update({
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "first": np.round(arr.reshape(-1, arr.shape[-1])[0], 5).tolist()
                if arr.ndim >= 1 else float(arr),
            })
        return out
    if isinstance(x, dict):
        return {k: summarize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [summarize(v) for v in x]
    return repr(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--task", default="sweep the cutting board with the broom")
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.SNDTIMEO, int(args.timeout * 1000))
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.server)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {
        "video": {
            "cam0": img[None, None],
            "cam1": img[None, None],
            "cam2": img[None, None],
        },
        "state": {
            "arm": np.zeros((1, 1, 6), dtype=np.float32),
            "hand": np.zeros((1, 1, 6), dtype=np.float32),
        },
        "language": {
            "annotation.human.task_description": [[args.task]],
        },
    }
    req = {
        "endpoint": "get_action",
        "data": {"observation": obs, "options": None},
    }

    print(f"[probe] send get_action to {args.server}")
    t0 = time.time()
    sock.send(msgpack.packb(req, default=msgpack_numpy.encode, use_bin_type=True))
    reply = msgpack.unpackb(sock.recv(), raw=False)
    print(f"[probe] reply in {(time.time() - t0) * 1000:.1f} ms")
    pprint.pp(summarize(reply), width=120)

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
