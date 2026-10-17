"""分镜脚本 → 短视频。

一个视频就是一个 YAML 文件：顶部是全片默认值，scenes 里每项是一个镜头。

    title: 江南古镇一日游
    ratio: "9:16"
    style: 国风水墨
    voice: 旁白·沉稳男声
    scenes:
      - text: 清晨的乌镇，还笼罩在一层薄雾里。
        image: 晨雾中的江南水乡，石桥，乌篷船
      - text: 下次，我们去西塘。
        image_file: 素材/西塘.jpg

每个镜头的图和每句话的配音都按内容算哈希缓存在 <输出目录>/video/<标题>/cache 里，
改了一句旁白再跑，只重新配那一句，其余直接复用。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .backends import ImageGenerator, TTSGenerator
from .config import Config
from .presets import Presets
from .video import (MOTIONS, Line, RenderOptions, Shot, audio_duration, build_command, build_timeline,
                    default_font, find_ffmpeg, run_ffmpeg, split_sentences, video_size, write_ass, write_srt)

# 镜头里能写的字段；没写的用脚本顶部的默认值
SCENE_KEYS = ("text", "image", "image_file", "style", "voice", "emotion", "emo_alpha", "motion", "transition",
              "hold", "seed", "model")
SCRIPT_KEYS = ("title", "ratio", "scenes", "subtitles") + tuple(k for k in SCENE_KEYS if k not in
                                                                ("text", "image", "image_file", "hold", "seed"))


@dataclass
class Scene:
    text: str = ""
    image: str = ""
    image_file: Path | None = None
    style: str | None = None
    voice: str | None = None
    emotion: str | None = None
    emo_alpha: float | None = None
    motion: str = "zoom_in"
    transition: str = "fade"
    hold: float = 0.0
    seed: int | None = None
    model: str | None = None


@dataclass
class Storyboard:
    title: str
    ratio: str
    scenes: list[Scene]
    subtitles: bool = True
    base_dir: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, path: str | Path) -> "Storyboard":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data, path.parent, default_title=path.stem)

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | None = None, default_title: str = "video") -> "Storyboard":
        if not isinstance(data, dict):
            raise ValueError("分镜脚本格式不对：顶层应该是 title / scenes 这样的键值")
        unknown = set(data) - set(SCRIPT_KEYS)
        if unknown:
            raise ValueError(f"分镜脚本里有不认识的字段: {'、'.join(sorted(unknown))}")
        base_dir = base_dir or Path.cwd()
        raw = data.get("scenes") or []
        if not raw:
            raise ValueError("分镜脚本里没有镜头（scenes 为空）")
        defaults = {k: data[k] for k in SCENE_KEYS if k in data}
        scenes = []
        for i, item in enumerate(raw, 1):
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                raise ValueError(f"第 {i} 个镜头格式不对")
            bad = set(item) - set(SCENE_KEYS)
            if bad:
                raise ValueError(f"第 {i} 个镜头有不认识的字段: {'、'.join(sorted(bad))}")
            merged = {**defaults, **item}
            text = str(merged.get("text") or "").strip()
            image = str(merged.get("image") or "").strip()
            image_file = None
            if merged.get("image_file"):
                image_file = Path(str(merged["image_file"])).expanduser()
                if not image_file.is_absolute():
                    image_file = base_dir / image_file
                if not image_file.is_file():
                    raise ValueError(f"第 {i} 个镜头的图片不存在: {image_file}")
            if not (image or text or image_file):
                raise ValueError(f"第 {i} 个镜头既没有旁白也没有画面")
            motion = str(merged.get("motion") or "zoom_in")
            if motion not in MOTIONS:
                raise ValueError(f"第 {i} 个镜头的运镜 {motion!r} 不支持，可选: {'、'.join(MOTIONS)}")
            scenes.append(Scene(
                text=text, image=image, image_file=image_file,
                style=merged.get("style"), voice=merged.get("voice"), emotion=merged.get("emotion"),
                emo_alpha=None if merged.get("emo_alpha") is None else float(merged["emo_alpha"]),
                motion=motion, transition=str(merged.get("transition") or "fade"),
                hold=float(merged.get("hold") or 0),
                seed=None if merged.get("seed") is None else int(merged["seed"]),
                model=merged.get("model"),
            ))
        return cls(str(data.get("title") or default_title), str(data.get("ratio") or "9:16"), scenes,
                   bool(data.get("subtitles", True)), base_dir)


@dataclass
class VideoResult:
    video: Path
    srt: Path | None
    duration: float
    info: dict[str, Any] = field(default_factory=dict)


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "_", text).strip("_")[:40] or "video"


class VideoComposer:
    def __init__(self, cfg: Config, images: ImageGenerator, tts: TTSGenerator,
                 progress: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.images = images
        self.tts = tts
        self.progress = progress or (lambda msg: None)

    @property
    def opts(self) -> dict:
        return self.cfg.data["video"]

    def work_dir(self, sb: Storyboard) -> Path:
        d = self.cfg.output_dir / "video" / _slug(sb.title)
        (d / "cache").mkdir(parents=True, exist_ok=True)
        return d

    async def compose(self, sb: Storyboard, out: Path | None = None) -> VideoResult:
        o = self.opts
        width, height = video_size(sb.ratio)
        check_presets(self.cfg, sb)
        ffmpeg = find_ffmpeg(o.get("ffmpeg") or "")
        work = self.work_dir(sb)
        cache = work / "cache"
        n = len(sb.scenes)

        # 1. 先配音：快，而且能先算出总时长
        lines_per_scene: list[list[Line]] = []
        stats = {"tts": 0, "tts_cached": 0, "image": 0, "image_cached": 0}
        for i, sc in enumerate(sb.scenes, 1):
            lines = []
            for sentence in split_sentences(sc.text):
                self.progress(f"配音 {i}/{n}：{sentence}")
                wav = await self._speak(sentence, sc, cache, stats)
                lines.append(Line(sentence, wav, audio_duration(wav, ffmpeg)))
            lines_per_scene.append(lines)

        # 2. 再出图（两个模型不同时跑，避免一起占显存）
        shots: list[Shot] = []
        for i, (sc, lines) in enumerate(zip(sb.scenes, lines_per_scene), 1):
            if sc.image_file:
                img = sc.image_file
            else:
                self.progress(f"出图 {i}/{n}：{sc.image or sc.text}")
                img = await self._image(sc, sb, i, cache, stats)
            shots.append(Shot(img, lines, sc.motion, sc.transition, sc.hold))

        # 3. 时间轴 + 字幕 + ffmpeg
        tl = build_timeline(shots, lead=float(o["lead"]), tail=float(o["tail"]), gap=float(o["sentence_gap"]),
                            max_chars=int(o["subtitle"]["max_chars"] or (16 if height > width else 24)))
        name = _slug(sb.title)
        out = Path(out).expanduser().resolve() if out else work / f"{name}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub = o["subtitle"]
        ass = srt = None
        if sb.subtitles and sub.get("burn", True) and tl.cues:
            ass = write_ass(tl.cues, work / "subtitles.ass", width, height, sub.get("font") or default_font(),
                            int(sub.get("size") or 0), int(sub.get("margin_v") or 0))
        if sb.subtitles and sub.get("srt", True) and tl.cues:
            srt = write_srt(tl.cues, out.with_suffix(".srt"))
        fonts_dir = self.cfg.resolve_path(sub["fonts_dir"]) if sub.get("fonts_dir") else None
        ropt = RenderOptions(
            width=width, height=height, fps=int(o["fps"]), transition_duration=float(o["transition_duration"]),
            zoom=float(o["zoom"]), codec=o["codec"], crf=int(o["crf"]), preset=o["preset"],
            audio_bitrate=str(o["audio_bitrate"]), loudness=float(o["loudness"]),
            subtitles=ass.name if ass else None, fonts_dir=fonts_dir,
        )
        args = build_command(ffmpeg, tl, out, ropt, seed=sb.title)
        (work / "ffmpeg_cmd.json").write_text(json.dumps(args, ensure_ascii=False, indent=1), encoding="utf-8")
        self.progress(f"合成视频：{n} 个镜头，{tl.total:.1f} 秒")
        await asyncio.to_thread(run_ffmpeg, args, work)
        info = {"scenes": n, "size": f"{width}x{height}", **stats}
        return VideoResult(out, srt, tl.total, info)

    async def _speak(self, sentence: str, sc: Scene, cache: Path, stats: dict) -> Path:
        voice = self.tts.presets.voice(sc.voice) if self.cfg.tts["backend"] != "openai" else None
        key = _hash({"text": sentence, "voice": sc.voice, "voice_file": _file_hash(voice.file) if voice else None,
                     "emotion": sc.emotion, "alpha": sc.emo_alpha, "backend": self.cfg.tts["backend"]})
        dest = cache / f"tts_{key}.wav"
        if dest.is_file():
            stats["tts_cached"] += 1
            return dest
        r = await self.tts.speak(sentence, sc.voice, sc.emotion, sc.emo_alpha)
        shutil.move(str(r.files[0]), dest)
        stats["tts"] += 1
        return dest

    async def _image(self, sc: Scene, sb: Storyboard, index: int, cache: Path, stats: dict) -> Path:
        prompt = sc.image or sc.text
        # 没指定种子时按内容算一个固定种子，这样重跑画面不变；想换一张就在镜头里写 seed
        seed = sc.seed if sc.seed is not None else zlib.crc32(f"{sb.title}|{index}|{prompt}".encode()) & 0x7FFFFFFF
        key = _hash({"prompt": prompt, "style": sc.style, "ratio": sb.ratio, "seed": seed, "model": sc.model,
                     "steps": self.cfg.image["steps"], "cfg": self.cfg.image["cfg"],
                     "backend": self.cfg.image["backend"]})
        dest = cache / f"img_{key}.png"
        if dest.is_file():
            stats["image_cached"] += 1
            return dest
        r = await self.images.generate(prompt, sb.ratio, None, seed, 1, style=sc.style, model=sc.model)
        shutil.move(str(r.files[0]), dest)
        stats["image"] += 1
        return dest


_FILE_HASHES: dict[tuple[str, float], str] = {}


def _file_hash(path: Path) -> str:
    st = path.stat()
    k = (str(path), st.st_mtime)
    if k not in _FILE_HASHES:
        _FILE_HASHES[k] = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return _FILE_HASHES[k]


def check_presets(cfg: Config, sb: Storyboard) -> None:
    """开跑前先检查风格、音色都存在，免得配完一半音才报错。"""
    p = Presets.load(cfg)
    for i, sc in enumerate(sb.scenes, 1):
        try:
            p.style(sc.style)
            if sc.text and cfg.tts["backend"] != "openai":
                p.voice(sc.voice)
        except ValueError as e:
            raise ValueError(f"第 {i} 个镜头：{e}") from e
