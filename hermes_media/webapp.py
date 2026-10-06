"""本地网页界面（节点画布），不依赖 ComfyUI，只用标准库起服务。

    hermes-media ui        # 打开 http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import queue
import re
import signal
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .backends import ImageGenerator, TTSGenerator
from .config import Config
from .downloads import SOURCES, DownloadManager
from .models import ModelRegistry
from .nodes import Job, build_nodes, emotion_alphas, execute
from .presets import AUDIO_EXTS, Presets
from .workers import WorkerManager

WEB_DIR = Path(__file__).resolve().parent / "web"
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp"} | AUDIO_EXTS


class App:
    def __init__(self, cfg: Config, workers: WorkerManager | None = None,
                 images: ImageGenerator | None = None, tts: TTSGenerator | None = None):
        self.cfg = cfg
        self.workers = workers or WorkerManager(cfg)
        self.images = images or ImageGenerator(cfg, self.workers)
        self.tts = tts or TTSGenerator(cfg, self.workers)
        self.downloads = DownloadManager(cfg)
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._runner, daemon=True).start()

    def _runner(self):
        """和 ComfyUI 一样排队执行：一次只跑一个任务，显卡不会被抢。"""
        loop = asyncio.new_event_loop()
        while True:
            job, graph = self.queue.get()
            try:
                loop.run_until_complete(execute(graph, self.cfg, self.images, self.tts, job))
            except Exception as e:  # noqa: BLE001  任何意外都不能让排队线程退出
                job.status, job.error = "error", str(e)

    def submit(self, graph: dict) -> Job:
        job = Job()
        self.jobs[job.id] = job
        self.queue.put((job, graph))
        return job

    def node_info(self) -> dict:
        p = Presets.load(self.cfg)
        return {
            "nodes": [d.to_json() for d in build_nodes(self.cfg).values()],
            "emotion_alphas": emotion_alphas(self.cfg),
            "styles": [vars(s) for s in p.styles],
            "voices": [{"id": v.id, "name": v.name, "desc": v.desc, "available": v.available,
                        "source": v.source, "url": f"/api/voices/audio?name={quote(v.name)}" if v.available else None}
                       for v in p.voices(include_missing=True)],
        }

    def models(self) -> dict:
        reg = ModelRegistry(self.cfg)
        try:
            current = reg.get(None).id
        except ValueError:
            current = "default"
        return {"entries": [e.to_json() for e in reg.entries()], "tasks": self.downloads.status(),
                "sources": SOURCES, "models_dir": str(reg.models_dir), "default_model": current,
                "default_source": self.cfg.image.get("download_source") or "huggingface"}

    def status(self) -> dict:
        st = {"image_backend": self.cfg.image["backend"], "tts_backend": self.cfg.tts["backend"],
              "workers": self.workers.status(), "queue": self.queue.qsize()}
        img = st["workers"].get("image") or {}
        if img.get("progress"):
            st["progress"] = img["progress"]
        return st

    def history(self, limit: int = 60) -> list[dict]:
        out_dir = self.cfg.output_dir
        files = [f for f in out_dir.iterdir() if f.is_file() and f.suffix.lower() in MEDIA_EXTS]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return [{"url": "/files/" + f.name, "name": f.name,
                 "kind": "audio" if f.suffix.lower() in AUDIO_EXTS else "image"} for f in files[:limit]]

    def save_voice(self, name: str, ext: str, data: bytes) -> str:
        name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
        ext = ext.lower() if ext.lower() in AUDIO_EXTS else ".wav"
        if not name or not data:
            raise ValueError("音色名和音频都不能为空")
        if len(data) > 30 * 1024 * 1024:
            raise ValueError("音频太大（>30MB），参考音频 3~10 秒就够了")
        d = Presets.load(self.cfg).voices_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}{ext}").write_bytes(data)
        return name


def _safe_file(base: Path, rel: str) -> Path | None:
    p = (base / unquote(rel)).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _file(self, path: Path | None):
            if path is None:
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def do_GET(self):
            u = urlparse(self.path)
            path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self._file(WEB_DIR / "index.html")
                if path.startswith("/static/"):
                    return self._file(_safe_file(WEB_DIR, path[len("/static/"):]))
                if path.startswith("/files/"):
                    return self._file(_safe_file(app.cfg.output_dir, path[len("/files/"):]))
                if path == "/api/nodes":
                    return self._json(app.node_info())
                if path == "/api/models":
                    return self._json(app.models())
                if path == "/api/status":
                    return self._json(app.status())
                if path == "/api/history":
                    return self._json(app.history())
                if path == "/api/voices/audio":
                    name = parse_qs(u.query).get("name", [""])[0]
                    return self._file(Presets.load(app.cfg).voice(name).file)
                m = re.fullmatch(r"/api/jobs/(\w+)", path)
                if m:
                    job = app.jobs.get(m.group(1))
                    return self._json(job.to_json()) if job else self._json({"error": "没有这个任务"}, 404)
                self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                if u.path == "/api/run":
                    body = json.loads(self._body() or b"{}")
                    job = app.submit(body.get("graph") or {})
                    return self._json({"job_id": job.id})
                if u.path == "/api/models/download":
                    body = json.loads(self._body() or b"{}")
                    return self._json(app.downloads.start(body.get("id", ""), body.get("source")).to_json())
                if u.path == "/api/voices/upload":
                    q = parse_qs(u.query)
                    name = app.save_voice(q.get("name", [""])[0], q.get("ext", [".wav"])[0], self._body())
                    return self._json({"ok": True, "name": name})
                self._json({"error": "not found"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)

    return Handler


def serve(cfg: Config, host: str | None = None, port: int | None = None, open_browser: bool | None = None,
          app: App | None = None) -> None:
    web = cfg.data["web"]
    host, port = host or web["host"], int(port or web["port"])
    app = app or App(cfg)
    if web.get("preload", True):
        app.workers.preload()
    server = ThreadingHTTPServer((host, port), make_handler(app))

    def on_term(*_):
        app.workers.stop_all()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"hermes-media 界面已启动: {url}  （Ctrl+C 退出）", flush=True)
    if open_browser if open_browser is not None else web.get("open_browser", True):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.workers.stop_all()
        time.sleep(0.2)
