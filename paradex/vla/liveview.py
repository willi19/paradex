"""Tiny MJPEG live-view HTTP server.

Give it a `get_frame()` callback returning a BGR uint8 image (or None to skip),
call .start(), and open the printed URL in a browser to watch frames in real
time. Runs its own grab + HTTP threads (non-blocking); works over SSH port
forwards, unlike cv2.imshow which needs an X display.
"""
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class MJPEGLiveView:
    def __init__(self, get_frame, port=8100, fps=15, quality=80, title="live"):
        self.get_frame = get_frame
        self.port = int(port)
        self.period = 1.0 / max(1.0, float(fps))
        self.quality = int(quality)
        self.title = title
        self._jpg = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._grab_t = threading.Thread(target=self._grab, daemon=True)
        self._srv = None

    def _grab(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                img = self.get_frame()
                if img is not None:
                    ok, buf = cv2.imencode(".jpg", img,
                                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if ok:
                        with self._lock:
                            self._jpg = buf.tobytes()
            except Exception:
                pass
            dt = self.period - (time.time() - t)
            if dt > 0:
                self._stop.wait(dt)

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        (f"<html><head><title>{outer.title}</title></head>"
                         f"<body style='margin:0;background:#111;text-align:center'>"
                         f"<img src='/stream' style='max-width:100%;height:auto'>"
                         f"</body></html>").encode())
                    return
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while not outer._stop.is_set():
                            with outer._lock:
                                jpg = outer._jpg
                            if jpg:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    (f"Content-Length: {len(jpg)}\r\n\r\n").encode())
                                self.wfile.write(jpg)
                                self.wfile.write(b"\r\n")
                            time.sleep(outer.period)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(404)
                self.end_headers()

        return H

    def start(self):
        self._grab_t.start()
        self._srv = ThreadingHTTPServer(("0.0.0.0", self.port), self._handler())
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"[liveview] {self.title}: http://localhost:{self.port}")
        return self

    def stop(self):
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.shutdown()
            except Exception:
                pass
