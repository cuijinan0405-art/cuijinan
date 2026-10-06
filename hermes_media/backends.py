"""出图 / 配音后端。每个后端只做一件事：把参数变成本地文件路径。

- local  ：直接跑模型（workers/ 下的常驻进程），不需要 ComfyUI
- openai ：任何 OpenAI 兼容接口
- comfyui：已有 ComfyUI 时可选
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .comfyui import ComfyUIClient, apply_params, resolve_specs
from .config import Config
from .presets import Presets
from .workers import WorkerManager


@dataclass
class Result:
    files: list[Path]
    info: dict[str, Any] = field(default_factory=dict)


def _stamp(text: str, kind: str) -> str:
    slug = re.sub(r"[^\w一-鿿]+", "_", text).strip("_")[:20] or kind
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}_{slug}"


def _load_workflow(cfg: Config, section: dict) -> dict:
    path = cfg.resolve_path(section["workflow"])
    if not path.is_file():
        raise FileNotFoundError(f"找不到 ComfyUI 工作流: {path}（请在 ComfyUI 里 导出 → API 格式 后放到这里）")
    wf = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in wf and "links" in wf:
        raise ValueError(f"{path} 是 UI 格式，请在 ComfyUI 里用「导出(API)」重新导出")
    return wf


def _error_text(r: httpx.Response) -> str:
    try:
        return r.json().get("error") or r.text[:500]
    except ValueError:
        return r.text[:500]


# ----------------------------------------------------------------------------- 出图


class ImageGenerator:
    def __init__(self, cfg: Config, workers: WorkerManager | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.workers = workers
        self.transport = transport

    async def generate(self, prompt: str, ratio: str | None = None, negative_prompt: str | None = None,
                       seed: int | None = None, count: int = 1, style: str | None = None,
                       steps: int | None = None, cfg_scale: float | None = None) -> Result:
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        st = Presets.load(self.cfg).style(style)
        if st:
            prompt = st.apply(prompt)
            ratio = ratio or st.ratio
            if negative_prompt is None and st.negative is not None:
                negative_prompt = st.negative
        count = max(1, min(int(count or 1), 8))
        width, height = self.cfg.image_size(ratio)
        neg = negative_prompt if negative_prompt is not None else self.cfg.image["negative_prompt"]
        seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else int(seed)
        steps = int(steps or self.cfg.image["steps"])
        cfg_scale = float(cfg_scale or self.cfg.image["cfg"])
        backend = self.cfg.image["backend"]
        if backend == "local":
            r = await self._local(prompt, neg, width, height, seed, count, steps, cfg_scale)
        elif backend == "comfyui":
            r = await self._comfyui(prompt, neg, width, height, seed, count)
        elif backend == "openai":
            r = await self._openai(prompt, neg, width, height, seed, count)
        else:
            raise ValueError(f"未知的出图后端: {backend}")
        r.info.update({"style": st.name if st else None, "prompt": prompt})
        return r

    async def _local(self, prompt, neg, width, height, seed, count, steps, cfg_scale) -> Result:
        sec = self.cfg.image["local"]
        url = await asyncio.to_thread(self.workers["image"].ensure_ready)
        prefix = _stamp(prompt, "img")
        files, seeds = [], []
        async with httpx.AsyncClient(timeout=sec.get("timeout", 1200), transport=self.transport) as c:
            for i in range(count):
                body = {"prompt": prompt, "negative_prompt": neg, "width": width, "height": height,
                        "steps": steps, "cfg": cfg_scale, "seed": seed + i}
                r = await c.post(url + "/generate", json=body)
                if r.status_code != 200:
                    raise RuntimeError(f"出图失败: {_error_text(r)}")
                dest = self.cfg.output_dir / (f"{prefix}_{i + 1}.png" if count > 1 else f"{prefix}.png")
                dest.write_bytes(r.content)
                files.append(dest)
                seeds.append(seed + i)
        return Result(files, {"backend": "local", "size": f"{width}x{height}", "seeds": seeds})

    async def _comfyui(self, prompt, neg, width, height, seed, count) -> Result:
        sec = self.cfg.image["comfyui"]
        wf = _load_workflow(self.cfg, sec)
        specs = resolve_specs(wf)
        client = ComfyUIClient(sec["url"], sec.get("timeout", 600), self.transport)
        prefix = _stamp(prompt, "img")
        files, seeds = [], []
        for i in range(count):
            params = {"prompt": prompt, "negative_prompt": neg, "width": width, "height": height, "seed": seed + i}
            run_wf = apply_params(wf, params, specs)
            seeds.append(params.get("seed"))
            files += await client.run(run_wf, self.cfg.output_dir, f"{prefix}_{i + 1}" if count > 1 else prefix)
        return Result(files, {"backend": "comfyui", "size": f"{width}x{height}", "seeds": seeds})

    async def _openai(self, prompt, neg, width, height, seed, count) -> Result:
        sec = self.cfg.image["openai"]
        body: dict[str, Any] = {"model": sec["model"], "prompt": prompt, "n": count, "seed": seed,
                                "size": f"{width}x{height}", "response_format": "b64_json"}
        if neg:
            body["negative_prompt"] = neg
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
        return Result(files, {"backend": "openai", "size": f"{width}x{height}", "seeds": [seed]})


# ----------------------------------------------------------------------------- 配音


class TTSGenerator:
    def __init__(self, cfg: Config, workers: WorkerManager | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.workers = workers
        self.transport = transport

    @property
    def presets(self) -> Presets:
        return Presets.load(self.cfg)

    async def speak(self, text: str, voice: str | None = None, emotion: str | None = None,
                    emo_alpha: float | None = None) -> Result:
        if not text.strip():
            raise ValueError("配音文本不能为空")
        presets = self.presets
        backend = self.cfg.tts["backend"]
        if backend == "openai":
            return await self._openai(text, voice)
        v = presets.voice(voice)
        emo = presets.emotion(emotion if emotion is not None else v.emotion)
        emo_text = emo.text if emo else ""
        alpha = float(emo_alpha if emo_alpha is not None and emo_alpha >= 0 else (emo.alpha if emo else 1.0))
        info = {"backend": backend, "voice": v.name, "emotion": emo.name if emo and emo.text else None}
        if backend == "local":
            files = await self._local(text, v.file, emo_text, alpha)
        elif backend == "comfyui":
            files = await self._comfyui(text, v.file, emo_text, alpha)
        else:
            raise ValueError(f"未知的配音后端: {backend}")
        return Result(files, info)

    async def _local(self, text, ref: Path, emo_text, alpha) -> list[Path]:
        sec = self.cfg.tts["local"]
        url = await asyncio.to_thread(self.workers["tts"].ensure_ready)
        body = {"text": text, "voice_name": ref.name, "voice_b64": base64.b64encode(ref.read_bytes()).decode(),
                "emo_text": emo_text or None, "emo_alpha": alpha}
        async with httpx.AsyncClient(timeout=sec.get("timeout", 900), transport=self.transport) as c:
            r = await c.post(url + "/tts", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"配音失败: {_error_text(r)}")
        dest = self.cfg.output_dir / f"{_stamp(text, 'tts')}.wav"
        dest.write_bytes(r.content)
        return [dest]

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

    async def _comfyui(self, text, ref: Path, emo_text, alpha) -> list[Path]:
        sec = self.cfg.tts["comfyui"]
        wf = _load_workflow(self.cfg, sec)
        specs = resolve_specs(wf)
        client = ComfyUIClient(sec["url"], sec.get("timeout", 600), self.transport)
        params: dict[str, Any] = {"text": text, "emotion": emo_text or None, "emo_alpha": alpha}
        if any(s.param == "voice" for s in specs):
            params["voice"] = await client.upload(ref)
        return await client.run(apply_params(wf, params, specs), self.cfg.output_dir, _stamp(text, "tts"))
