"""`dreamloop play`: a tiny local server. The browser sends keys, the world model sends frames."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

import numpy as np
from PIL import Image

from .rollout import WorldModel


class Session:
    def __init__(self, wm: WorldModel, temperature: float, seed: int = 0):
        self.wm, self.temperature = wm, temperature
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> np.ndarray:
        c = self.wm.context
        ep = int(self.rng.integers(len(self.wm.seed_frames)))
        start = int(self.rng.integers(0, self.wm.seed_frames.shape[1] - c))
        frames = self.wm.seed_frames[ep, start : start + c]
        self.ctx = self.wm.encode(frames)[None]
        self.steps = 0
        return frames[-1]

    def step(self, action: int) -> np.ndarray:
        action = int(np.clip(action, 0, self.wm.lam.num_actions - 1))
        nxt = self.wm.next_tokens(self.ctx, np.array([action]), temperature=self.temperature)
        self.ctx = np.concatenate([self.ctx[:, 1:], nxt[:, None]], axis=1)
        self.steps += 1
        return self.wm.decode(nxt)[0]


def _png(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def serve(run_dir: str, port: int = 8080, temperature: float = 0.0, open_browser: bool = True) -> None:
    wm = WorldModel(run_dir)
    session = Session(wm, temperature)
    page = resources.files("dreamloop").joinpath("web/index.html").read_bytes()
    idle = wm.bindings.get("noop", 0)
    info = {
        "game": wm.game,
        "num_actions": wm.lam.num_actions,
        "bindings": wm.bindings,
        "idle": idle,
        "fps": wm.cfg.get("fps", 15),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, kind: str = "application/json", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), status=status)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(info)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with session.lock:
                started = time.perf_counter()
                if self.path == "/api/reset":
                    frame = session.reset()
                elif self.path == "/api/step":
                    frame = session.step(int(body.get("action", idle)))
                else:
                    self._json({"error": "not found"}, 404)
                    return
                ms = (time.perf_counter() - started) * 1000
                self._json({"frame": _png(frame), "step": session.steps, "ms": round(ms, 1)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Dreaming {wm.game} at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
