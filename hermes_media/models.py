"""出图模型：目录（presets/models.yaml）+ 已下载 + 配置里自定义的，统一成可选列表。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config

DONE_MARK = ".hm_complete"


@dataclass
class LoadSpec:
    """交给 image_worker 的加载参数。"""
    model: str
    transformer: str | None = None
    text_encoder: str | None = None
    lora: str | None = None
    lora_scale: float = 1.0

    def args(self) -> list[str]:
        out = ["--model", self.model]
        for flag, v in (("--transformer", self.transformer), ("--text-encoder", self.text_encoder),
                        ("--lora", self.lora)):
            if v:
                out += [flag, v]
        if self.lora:
            out += ["--lora-scale", str(self.lora_scale)]
        return out


@dataclass
class ModelEntry:
    id: str
    name: str
    kind: str = "pipeline"  # pipeline | transformer | text_encoder | custom
    desc: str = ""
    base: str | None = None
    files: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    custom: dict | None = None  # 配置里手写的模型
    installed: bool = False
    path: Path | None = None

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "desc": self.desc, "base": self.base,
                "sources": list(self.sources), "installed": self.installed,
                "downloadable": bool(self.sources), "path": str(self.path) if self.path else None}


class ModelRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def models_dir(self) -> Path:
        return self.cfg.resolve_path(self.cfg.image.get("models_dir") or "models")

    def catalog(self) -> list[ModelEntry]:
        path = self.cfg.presets_dir / "models.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else []
        out = []
        for m in data or []:
            e = ModelEntry(**{k: v for k, v in m.items() if k in ("id", "name", "kind", "desc", "base", "files", "sources")})
            e.path = self.models_dir / e.id
            e.installed = (e.path / DONE_MARK).is_file()
            out.append(e)
        return out

    def entries(self) -> list[ModelEntry]:
        """所有条目（含未下载的），顺序：配置默认 → 自定义 → 目录。"""
        m = str(self.cfg.image["local"]["model"])
        exists = _is_repo_id(m) or self.cfg.resolve_path(m).exists()
        out = [ModelEntry("default", "配置文件里的模型", "custom", m if exists else f"{m}（目录不存在）",
                          custom={"model": m}, installed=exists)]
        for i, m in enumerate(self.cfg.image.get("models") or []):
            out.append(ModelEntry(m.get("id") or f"custom{i + 1}", m.get("name") or f"自定义模型 {i + 1}", "custom",
                                  m.get("desc", m.get("model", "")), custom=m, installed=True))
        return out + self.catalog()

    def available(self) -> list[ModelEntry]:
        return [e for e in self.entries() if e.installed]

    def get(self, key: str | None) -> ModelEntry:
        entries = self.entries()
        if not key and not self.cfg.image.get("default_model"):
            # 没指定：用配置里的模型；它不存在就用第一个已下载的
            avail = [e for e in entries if e.installed]
            if not avail:
                raise ValueError("还没有可用的出图模型：在 config.yaml 填 image.local.model，或在界面「📦 模型」里下载")
            return avail[0]
        key = (key or self.cfg.image.get("default_model")).strip()
        for e in entries:
            if key in (e.id, e.name):
                if not e.installed:
                    raise ValueError(f"模型「{e.name}」还没下载，请在界面「模型管理」里下载")
                return e
        names = "、".join(e.name for e in self.available())
        raise ValueError(f"没有模型 {key!r}。可用: {names}")

    def load_spec(self, key: str | None) -> tuple[ModelEntry, LoadSpec]:
        e = self.get(key)
        if e.custom is not None:
            c = e.custom
            spec = LoadSpec(str(c["model"]), c.get("transformer"), c.get("text_encoder"), c.get("lora"),
                            float(c.get("lora_scale", 1.0)))
            for attr in ("model", "transformer", "text_encoder", "lora"):
                v = getattr(spec, attr)
                if v and not _is_repo_id(v):
                    setattr(spec, attr, str(self.cfg.resolve_path(v)))
            return e, spec
        if e.kind == "pipeline":
            return e, LoadSpec(str(e.path))
        base = next((b for b in self.entries() if b.id == e.base), None) if e.base else None
        if base is None or not base.installed or base.kind not in ("pipeline", "custom"):
            base_name = base.name if base else e.base
            raise ValueError(f"「{e.name}」只包含扩散模型，需要先在「模型管理」下载基础模型「{base_name}」")
        _, base_spec = self.load_spec(base.id)
        if e.kind == "transformer":
            base_spec.transformer = str(_main_file(e.path))
        elif e.kind == "text_encoder":
            base_spec.text_encoder = str(e.path)
        return e, base_spec


def _is_repo_id(v: str) -> bool:
    """"Qwen/Qwen-Image-2.1" 这种 HuggingFace 名称（不是本地路径）。"""
    return v.count("/") == 1 and not Path(v).expanduser().exists() and ":" not in v and not v.startswith((".", "~"))


def _main_file(d: Path) -> Path:
    files = sorted(list(d.rglob("*.gguf")) + list(d.rglob("*.safetensors")), key=lambda p: -p.stat().st_size)
    files = [f for f in files if ".cache" not in f.parts]
    if not files:
        raise ValueError(f"{d} 里没有 .gguf / .safetensors 文件")
    return files[0]


def mark_done(dest: Path, info: dict) -> None:
    (dest / DONE_MARK).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
