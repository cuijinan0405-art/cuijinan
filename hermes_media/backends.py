"""出图 / 配音后端。每个后端只做一件事：把参数变成本地文件路径。"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .comfyui import ComfyUIClient, apply_params, resolve_specs
from .config import Config

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass
class Result:
    files: list[Path]
    info: dict[str, Any] = field(default_factory=dict)


def _stamp(text: str, kind: str) -> str:
    slug = re.sub(r"[^\w一-鿿]+", "_", text).strip("_")[:24] or kind
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}_{slug}"


def _load_workflow(cfg: Config, section: dict) -> dict:
    path = cfg.resolve_path(section["workflow"])
    if not path.is_file():
        raise FileNotFoundError(f"找不到 ComfyUI 工作流: {path}（请在 ComfyUI 里 导出 → API 格式 后放到这里）")
    wf = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in wf and "links" in wf:
        raise ValueError(f"{path} 是 UI 格式，请在 ComfyUI 里用「导出(API)」重新导出")
    return wf


# ----------------------------------------------------------------------------- 出图


class ImageGenerator:
    def __init__(self, cfg: Config, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.transport = transport

    async def generate(self, prompt: str, ratio: str | None = None, negative_prompt: str | None = None,
                       seed: int | None = None, count: int = 1) -> Result:
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        count = max(1, min(int(count or 1), 8))
        width, height = self.cfg.image_size(ratio)
        neg = negative_prompt if negative_prompt is not None else self.cfg.image["negative_prompt"]
        backend = self.cfg.image["backend"]
        if backend == "comfyui":
            return await self._comfyui(prompt, neg, width, height, seed, count)
        if backend == "openai":
            return await self._openai(prompt, neg, width, height, seed, count)
        raise ValueError(f"未知的出图后端: {backend}")

    async def _comfyui(self, prompt, neg, width, height, seed, count) -> Result:
        sec = self.cfg.image["comfyui"]
        wf = _load_workflow(self.cfg, sec)
        specs = resolve_specs(wf)
        client = ComfyUIClient(sec["url"], sec.get("timeout", 600), self.transport)
        prefix = _stamp(prompt, "img")
        files, seeds = [], []
        for i in range(count):
            params = {"prompt": prompt, "negative_prompt": neg, "width": width, "height": height,
                      "seed": None if seed is None else seed + i}
            run_wf = apply_params(wf, params, specs)
            seeds.append(params.get("seed"))
            files += await client.run(run_wf, self.cfg.output_dir, f"{prefix}_{i + 1}" if count > 1 else prefix)
        return Result(files, {"backend": "comfyui", "size": f"{width}x{height}", "seeds": seeds})

    async def _openai(self, prompt, neg, width, height, seed, count) -> Result:
        sec = self.cfg.image["openai"]
        body: dict[str, Any] = {"model": sec["model"], "prompt": prompt, "n": count,
                                "size": f"{width}x{height}", "response_format": "b64_json"}
        if neg:
            body["negative_prompt"] = neg
        if seed is not None:
            body["seed"] = seed
        headers = {"Authorization": f"Bearer {sec['api_key']}"} if sec.get("api_key") else {}
        async with httpx.AsyncClient(timeout=sec.get("timeout", 600), transport=self.transport) as c:
            r = await c.post(sec["base_url"].rstrip("/") + "/images/generations", json=body, headers=headers)
            if r.status_code != 200:
                raise RuntimeError(f"出图接口返回 {r.status_code}: {r.text[:500]}")
            prefix = _stamp(prompt, "img")
            files = []
            for i, item in enumerate(r.json().get("data", [])):
                if item.get("b64_json"):
                    data = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    data = (await c.get(item["url"])).content
                else:
                    continue
                dest = self.cfg.output_dir / f"{prefix}_{i + 1}.png"
                dest.write_bytes(data)
                files.append(dest)
        if not files:
            raise RuntimeError("出图接口没有返回图片")
        return Result(files, {"backend": "openai", "size": f"{width}x{height}", "seed": seed})


# ----------------------------------------------------------------------------- 配音


class TTSGenerator:
    def __init__(self, cfg: Config, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.transport = transport

    # -- 音色管理 --------------------------------------------------------------

    @property
    def voices_dir(self) -> Path:
        return self.cfg.resolve_path(self.cfg.tts["voices_dir"])

    def list_voices(self) -> list[Path]:
        d = self.voices_dir
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)

    def resolve_voice(self, voice: str | None) -> Path:
        voices = self.list_voices()
        if voice:
            p = Path(voice).expanduser()
            if p.is_file():
                return p.resolve()
            key = voice.strip().lower()
            for v in voices:  # 精确匹配文件名（不含扩展名）
                if v.stem.lower() == key:
                    return v
            for v in voices:  # 模糊匹配，比如 "男" 匹配 "男声_沉稳"
                if key in v.stem.lower():
                    return v
            names = "、".join(v.stem for v in voices) or "（空）"
            raise ValueError(f"找不到音色 {voice!r}。可用音色: {names}")
        default = self.cfg.tts.get("default_voice")
        if default:
            return self.resolve_voice(default)
        if voices:
            return voices[0]
        raise ValueError(f"还没有任何参考音色，请把 3~10 秒的人声 wav 放进 {self.voices_dir}")

    # -- 合成 ------------------------------------------------------------------

    async def speak(self, text: str, voice: str | None = None, emotion: str | None = None,
                    emo_alpha: float | None = None) -> Result:
        if not text.strip():
            raise ValueError("配音文本不能为空")
        backend = self.cfg.tts["backend"]
        alpha = float(emo_alpha if emo_alpha is not None else self.cfg.tts["emo_alpha"])
        if backend == "indextts2":
            return await self._indextts2(text, voice, emotion, alpha)
        if backend == "openai":
            return await self._openai(text, voice)
        if backend == "comfyui":
            return await self._comfyui(text, voice, emotion, alpha)
        raise ValueError(f"未知的配音后端: {backend}")

    async def _indextts2(self, text, voice, emotion, alpha) -> Result:
        sec = self.cfg.tts["indextts2"]
        ref = self.resolve_voice(voice)
        body = {
            "text": text,
            "voice_name": ref.name,
            "voice_b64": base64.b64encode(ref.read_bytes()).decode(),
            "emo_text": emotion or None,
            "emo_alpha": alpha,
        }
        async with httpx.AsyncClient(timeout=sec.get("timeout", 600), transport=self.transport) as c:
            r = await c.post(sec["url"].rstrip("/") + "/tts", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"IndexTTS2 服务返回 {r.status_code}: {r.text[:500]}")
        dest = self.cfg.output_dir / f"{_stamp(text, 'tts')}.wav"
        dest.write_bytes(r.content)
        return Result([dest], {"backend": "indextts2", "voice": ref.stem, "emotion": emotion})

    async def _openai(self, text, voice) -> Result:
        sec = self.cfg.tts["openai"]
        body = {"model": sec["model"], "input": text, "voice": voice or self.cfg.tts.get("default_voice") or "alloy",
                "response_format": "wav"}
        headers = {"Authorization": f"Bearer {sec['api_key']}"} if sec.get("api_key") else {}
        async with httpx.AsyncClient(timeout=sec.get("timeout", 600), transport=self.transport) as c:
            r = await c.post(sec["base_url"].rstrip("/") + "/audio/speech", json=body, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"配音接口返回 {r.status_code}: {r.text[:500]}")
        dest = self.cfg.output_dir / f"{_stamp(text, 'tts')}.wav"
        dest.write_bytes(r.content)
        return Result([dest], {"backend": "openai", "voice": body["voice"]})

    async def _comfyui(self, text, voice, emotion, alpha) -> Result:
        sec = self.cfg.tts["comfyui"]
        wf = _load_workflow(self.cfg, sec)
        specs = resolve_specs(wf)
        client = ComfyUIClient(sec["url"], sec.get("timeout", 600), self.transport)
        params: dict[str, Any] = {"text": text, "emotion": emotion, "emo_alpha": alpha}
        ref = None
        if any(s.param == "voice" for s in specs):
            ref = self.resolve_voice(voice)
            params["voice"] = await client.upload(ref)
        files = await client.run(apply_params(wf, params, specs), self.cfg.output_dir, _stamp(text, "tts"))
        return Result(files, {"backend": "comfyui", "voice": ref.stem if ref else None, "emotion": emotion})
