"""配置加载：YAML 文件 + 内置默认值。

查找顺序：
1. 环境变量 HERMES_MEDIA_CONFIG 指定的文件
2. 当前目录下的 config.yaml
3. ~/.hermes-media/config.yaml
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# 千问 Qwen-Image 官方推荐的分辨率（2.1 沿用同一套）
QWEN_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

DEFAULTS: dict[str, Any] = {
    "output_dir": "~/hermes-media-output",
    "image": {
        "backend": "comfyui",  # comfyui | openai
        "comfyui": {
            "url": "http://127.0.0.1:8188",
            "workflow": "workflows/qwen_image_2_1_api.json",
            "timeout": 600,
        },
        "openai": {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "",
            "model": "qwen-image-2.1",
            "timeout": 600,
        },
        "default_ratio": "1:1",
        "negative_prompt": "",
        "sizes": {k: list(v) for k, v in QWEN_IMAGE_SIZES.items()},
    },
    "tts": {
        "backend": "indextts2",  # indextts2 | openai | comfyui
        "indextts2": {
            "url": "http://127.0.0.1:7861",
            "timeout": 600,
        },
        "openai": {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "",
            "model": "tts-1",
            "timeout": 600,
        },
        "comfyui": {
            "url": "http://127.0.0.1:8188",
            "workflow": "workflows/indextts2_api.json",
            "timeout": 600,
        },
        "voices_dir": "voices",
        "default_voice": "",
        "emo_alpha": 0.6,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_config_file(explicit: str | os.PathLike | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("HERMES_MEDIA_CONFIG"):
        candidates.append(Path(os.environ["HERMES_MEDIA_CONFIG"]))
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path.home() / ".hermes-media" / "config.yaml")
    for c in candidates:
        c = c.expanduser()
        if c.is_file():
            return c.resolve()
    return None


class Config:
    def __init__(self, data: dict, base_dir: Path):
        self.data = data
        self.base_dir = base_dir

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        file = find_config_file(path)
        if path and file is None:
            raise FileNotFoundError(f"找不到配置文件: {path}")
        user: dict = {}
        base_dir = Path.cwd()
        if file is not None:
            user = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            base_dir = file.parent
        return cls(_deep_merge(DEFAULTS, user), base_dir)

    def resolve_path(self, p: str) -> Path:
        """相对路径以配置文件所在目录为基准。"""
        path = Path(os.path.expandvars(str(p))).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    @property
    def output_dir(self) -> Path:
        d = self.resolve_path(self.data["output_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def image(self) -> dict:
        return self.data["image"]

    @property
    def tts(self) -> dict:
        return self.data["tts"]

    def image_size(self, ratio: str | None) -> tuple[int, int]:
        sizes = self.image["sizes"]
        ratio = (ratio or self.image["default_ratio"]).replace("：", ":").strip()
        if ratio in sizes:
            w, h = sizes[ratio]
            return int(w), int(h)
        # 允许直接写 "1024x768"
        if "x" in ratio.lower():
            w, h = ratio.lower().split("x", 1)
            return int(w), int(h)
        raise ValueError(f"不支持的比例 {ratio!r}，可选: {', '.join(sizes)} 或 宽x高")
