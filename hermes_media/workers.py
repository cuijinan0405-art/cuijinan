"""模型常驻进程管理。

出图（diffusers）和配音（IndexTTS2）各跑在一个独立进程里，用各自的 Python 环境，
模型只加载一次。用到时如果还没启动就自动拉起；端口上已经有在跑的（比如网页界面
已经启动了），Hermes 那边就直接复用，不会重复加载模型。
"""

from __future__ import annotations

import atexit
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .config import PROJECT_ROOT, Config
from .models import ModelRegistry

WORKERS_DIR = PROJECT_ROOT / "workers"


class WorkerError(RuntimeError):
    pass


def _split(cmd: str) -> list[str]:
    """把配置里的 python 命令拆成参数；Windows 路径里的反斜杠和带空格的引号路径都要能用。"""
    parts = shlex.split(cmd, posix=os.name != "nt")
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


class Worker:
    def __init__(self, name: str, label: str, port: int, argv: list[str], cwd: Path | None,
                 startup_timeout: float, log_dir: Path):
        self.name, self.label, self.port = name, label, port
        self.argv, self.cwd = argv, cwd
        self.startup_timeout = startup_timeout
        self.log_path = log_dir / f"{name}.log"
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health(self) -> dict | None:
        try:
            r = httpx.get(self.url + "/health", timeout=2)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def status(self) -> dict:
        h = self.health()
        if h:
            return {"state": h.get("state", "ready"), **h}
        if self.proc and self.proc.poll() is None:
            return {"state": "starting"}
        if self.proc and self.proc.returncode is not None:
            return {"state": "error", "error": f"进程已退出（code {self.proc.returncode}），日志: {self.log_path}"}
        return {"state": "stopped"}

    def start(self) -> None:
        with self._lock:
            if self.health() or (self.proc and self.proc.poll() is None):
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "ab")
            log.write(f"\n==== {time.ctime()} 启动: {' '.join(self.argv)} (cwd={self.cwd})\n".encode())
            log.flush()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            try:
                self.proc = subprocess.Popen(self.argv, cwd=self.cwd or None, stdout=log,
                                             stderr=subprocess.STDOUT, env=env)
            except FileNotFoundError as e:
                raise WorkerError(f"{self.label}启动失败：找不到命令 {self.argv[0]!r}，请检查配置里的 python") from e

    def ensure_ready(self) -> str:
        """确保模型已加载完毕，返回服务地址。"""
        h = self.health()
        if h and h.get("state", "ready") == "ready":
            return self.url
        self.start()
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            h = self.health()
            if h:
                if h.get("state", "ready") == "ready":
                    return self.url
                if h.get("state") == "error":
                    raise WorkerError(f"{self.label}加载模型失败: {h.get('error')}（日志: {self.log_path}）")
            elif self.proc and self.proc.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise WorkerError(f"{self.label}进程退出了（code {self.proc.returncode}）。日志末尾:\n{tail}")
            time.sleep(1)
        raise WorkerError(f"{self.label}启动超时（{self.startup_timeout}s），日志: {self.log_path}")

    def shutdown(self, timeout: float = 30) -> None:
        """让端口上的进程退出（不管是不是本进程启动的），用于切换模型。"""
        try:
            httpx.post(self.url + "/shutdown", timeout=3)
        except httpx.HTTPError:
            pass
        self.stop()
        deadline = time.time() + timeout
        while self.health() and time.time() < deadline:
            time.sleep(0.3)
        self.proc = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class WorkerManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.workers: dict[str, Worker] = {}
        log_dir = cfg.output_dir / "logs"

        img = cfg.image["local"]
        self._image_base_argv = _split(img["python"]) + [
            str(WORKERS_DIR / "image_worker.py"), "--port", str(img["port"]), "--device", img["device"],
            "--dtype", img["dtype"], "--parent-pid", str(os.getpid())] + (["--offload"] if img.get("offload") else [])
        self._switch_lock = threading.Lock()
        self.workers["image"] = Worker("image", "千问 Image 2.1 ", int(img["port"]), self._image_argv(None),
                                       cfg.resolve_path(img["cwd"]) if img.get("cwd") else None,
                                       float(img["startup_timeout"]), log_dir)

        tts = cfg.tts["local"]
        itts = Path(tts["index_tts_dir"]).expanduser() if tts.get("index_tts_dir") else None
        argv = _split(tts["python"]) + [
            str(WORKERS_DIR / "indextts2_server.py"), "--model-dir", tts["model_dir"],
            "--version", str(tts["version"]), "--port", str(tts["port"]), "--parent-pid", str(os.getpid())]
        self.workers["tts"] = Worker("tts", "IndexTTS2 ", int(tts["port"]), argv, itts,
                                     float(tts["startup_timeout"]), log_dir)
        atexit.register(self.stop_all)

    def __getitem__(self, name: str) -> Worker:
        return self.workers[name]

    def _image_argv(self, model: str | None) -> list[str]:
        try:
            entry, spec = ModelRegistry(self.cfg).load_spec(model)
        except ValueError:
            if model:
                raise
            return self._image_base_argv + ["--model", str(self.cfg.image["local"]["model"]), "--model-id", "default"]
        return self._image_base_argv + spec.args() + ["--model-id", entry.id]

    def image_ready(self, model: str | None = None) -> str:
        """确保出图进程加载的是指定模型（不同就重启换模型），返回地址。"""
        with self._switch_lock:
            w = self.workers["image"]
            argv = self._image_argv(model)
            want = argv[argv.index("--model-id") + 1]
            h = w.health()
            if h and (h.get("model_id") not in (want, None) or h.get("state") == "error"):
                print(f"[hermes-media] 重新加载出图模型: {h.get('model_id')} → {want}", flush=True)
                w.shutdown()
            w.argv = argv
            return w.ensure_ready()

    def status(self) -> dict:
        out = {}
        if self.cfg.image["backend"] == "local":
            out["image"] = self.workers["image"].status()
        if self.cfg.tts["backend"] == "local":
            st = self.workers["tts"].status()
            if not self.cfg.tts["local"].get("index_tts_dir") and st["state"] == "stopped":
                st = {"state": "unconfigured", "error": "还没在 config.yaml 里填 tts.local.index_tts_dir"}
            out["tts"] = st
        return out

    def preload(self) -> None:
        """后台提前启动需要的模型进程，第一次用的时候就不用等。"""
        for key, section in (("image", self.cfg.image), ("tts", self.cfg.tts)):
            if section["backend"] != "local":
                continue
            if key == "tts" and not self.cfg.tts["local"].get("index_tts_dir"):
                continue
            threading.Thread(target=self._safe_ready, args=(key,), daemon=True).start()

    def _safe_ready(self, key: str) -> None:
        try:
            self.image_ready() if key == "image" else self.workers[key].ensure_ready()
        except Exception as e:  # noqa: BLE001
            print(f"[hermes-media] {self.workers[key].label}预加载失败: {e}", flush=True)

    def stop_all(self) -> None:
        for w in self.workers.values():
            w.stop()
