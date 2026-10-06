"""模型下载：HuggingFace / HF 国内镜像 / ModelScope，后台线程下载，可查进度。"""

from __future__ import annotations

import fnmatch
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .models import ModelEntry, ModelRegistry, mark_done

HF_MIRROR = "https://hf-mirror.com"
SOURCES = {"huggingface": "HuggingFace", "hf-mirror": "HF 国内镜像", "modelscope": "ModelScope 魔搭"}


@dataclass
class Task:
    model_id: str
    source: str
    state: str = "running"  # running | done | error
    total: int = 0
    downloaded: int = 0
    error: str | None = None
    started: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return dict(vars(self))


def _dir_size(d: Path) -> int:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _match(name: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name.split("/")[-1], p) for p in patterns)


class DownloadManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.registry = ModelRegistry(cfg)
        self.tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def start(self, model_id: str, source: str | None = None) -> Task:
        entry = next((e for e in self.registry.catalog() if e.id == model_id), None)
        if entry is None:
            raise ValueError(f"模型目录里没有 {model_id}")
        source = source or self.default_source(entry)
        repo_source = "huggingface" if source == "hf-mirror" else source
        if repo_source not in entry.sources:
            raise ValueError(f"「{entry.name}」没有 {SOURCES.get(source, source)} 下载源")
        with self._lock:
            t = self.tasks.get(model_id)
            if t and t.state == "running":
                return t
            t = self.tasks[model_id] = Task(model_id, source)
        threading.Thread(target=self._run, args=(entry, source, t), daemon=True).start()
        return t

    def default_source(self, entry: ModelEntry) -> str:
        pref = self.cfg.image.get("download_source") or "huggingface"
        if pref == "hf-mirror" and "huggingface" in entry.sources:
            return pref
        return pref if pref in entry.sources else next(iter(entry.sources))

    def status(self) -> dict:
        for t in self.tasks.values():
            if t.state == "running":
                t.downloaded = _dir_size(self.registry.models_dir / t.model_id)
        return {k: t.to_json() for k, t in self.tasks.items()}

    # -- 下载实现 ---------------------------------------------------------------

    def _run(self, entry: ModelEntry, source: str, t: Task) -> None:
        dest = entry.path
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if source == "modelscope":
                files = self._modelscope(entry, dest, t)
            else:
                files = self._huggingface(entry, dest, t, HF_MIRROR if source == "hf-mirror" else None)
            mark_done(dest, {"id": entry.id, "source": source, "repo": entry.sources.get(
                "huggingface" if source == "hf-mirror" else source), "files": files, "time": time.ctime()})
            t.downloaded = _dir_size(dest)
            t.state = "done"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            t.state, t.error = "error", _friendly(e, source)

    def _huggingface(self, entry: ModelEntry, dest: Path, t: Task, endpoint: str | None) -> list[str]:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError("缺少 huggingface_hub，请运行: pip install huggingface_hub") from e
        repo = entry.sources["huggingface"]
        patterns = entry.files or None
        plan = snapshot_download(repo, local_dir=dest, allow_patterns=patterns, endpoint=endpoint, dry_run=True)
        files = [p.filename for p in plan]
        if not files:
            raise RuntimeError(f"仓库 {repo} 里没有匹配 {entry.files} 的文件")
        t.total = sum(p.file_size or 0 for p in plan)
        snapshot_download(repo, local_dir=dest, allow_patterns=patterns, endpoint=endpoint, max_workers=8)
        return files

    def _modelscope(self, entry: ModelEntry, dest: Path, t: Task) -> list[str]:
        try:
            from modelscope import snapshot_download
            from modelscope.hub.api import HubApi
        except ImportError as e:
            raise RuntimeError("用魔搭下载需要先安装: pip install modelscope") from e
        repo = entry.sources["modelscope"]
        try:
            listing = HubApi().get_model_files(repo, recursive=True)
            wanted = [f for f in listing if f.get("Type") != "tree" and _match(f["Path"], entry.files)]
            t.total = sum(int(f.get("Size") or 0) for f in wanted)
        except Exception:  # noqa: BLE001  拿不到大小就不显示百分比
            wanted = []
        snapshot_download(repo, local_dir=str(dest), allow_patterns=entry.files or None)
        return [f["Path"] for f in wanted]


def _friendly(e: Exception, source: str) -> str:
    msg = f"{e.__class__.__name__}: {e}"
    low = msg.lower()
    if "401" in low or "403" in low or "gated" in low:
        return msg + "\n这个仓库需要登录或先在网页上同意协议：设置环境变量 HF_TOKEN，或换 ModelScope 下载。"
    if any(k in low for k in ("connect", "timeout", "resolve", "ssl", "proxy")):
        hint = "可以换「HF 国内镜像」或「ModelScope 魔搭」再试" if source == "huggingface" else "请检查网络后重试"
        return msg + f"\n网络连不上：{hint}。已下载的部分会保留，重试会接着下。"
    return msg
