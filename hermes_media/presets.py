"""风格 / 音色 / 情绪预设。每次调用都重新读文件，改了 YAML 不用重启。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass
class Style:
    id: str
    name: str
    prompt: str = "{prompt}"
    category: str = ""
    ratio: str = ""
    negative: str | None = None

    def apply(self, subject: str) -> str:
        subject = subject.strip()
        if "{prompt}" in self.prompt:
            return self.prompt.replace("{prompt}", subject)
        return f"{subject}，{self.prompt}" if subject else self.prompt


@dataclass
class Emotion:
    id: str
    name: str
    text: str = ""
    alpha: float = 0.6


@dataclass
class Voice:
    id: str
    name: str
    file: Path
    emotion: str = ""
    desc: str = ""
    source: str = "preset"  # preset | folder | example

    @property
    def available(self) -> bool:
        return self.file.is_file()


def _read_yaml(path: Path):
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _match(key: str, items, *attrs):
    """按 id / 名称精确匹配，再做包含匹配（「女」→「温柔女声」）。"""
    k = key.strip().lower()
    for it in items:
        if any(str(getattr(it, a)).lower() == k for a in attrs):
            return it
    for it in items:
        if any(k in str(getattr(it, a)).lower() for a in attrs):
            return it
    return None


@dataclass
class Presets:
    cfg: Config
    styles: list[Style] = field(default_factory=list)
    emotions: list[Emotion] = field(default_factory=list)
    _voice_data: dict = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: Config) -> "Presets":
        d = cfg.presets_dir
        styles = [Style(**{k: v for k, v in s.items() if k in Style.__dataclass_fields__})
                  for s in (_read_yaml(d / "styles.yaml") or [])]
        emotions = [Emotion(**{k: v for k, v in e.items() if k in Emotion.__dataclass_fields__})
                    for e in (_read_yaml(d / "emotions.yaml") or [])]
        if not any(e.id == "none" for e in emotions):
            emotions.insert(0, Emotion("none", "跟随音色", "", 1.0))
        if not styles:
            styles = [Style("none", "不加风格（原样）")]
        return cls(cfg, styles, emotions, _read_yaml(d / "voices.yaml") or {})

    # -- 风格 -------------------------------------------------------------------

    def style(self, key: str | None) -> Style | None:
        if not key:
            return None
        found = _match(key, self.styles, "id", "name")
        if not found:
            raise ValueError(f"没有风格 {key!r}。可选: {'、'.join(s.name for s in self.styles)}")
        return found

    # -- 情绪 -------------------------------------------------------------------

    def emotion(self, key: str | None) -> Emotion | None:
        """预设名 / id 返回预设；其它文字当作自定义情绪描述。"""
        if not key or not key.strip():
            return None
        found = _match(key, self.emotions, "id", "name")
        if found:
            return found
        return Emotion("custom", key.strip(), key.strip(), float(self.cfg.tts["emo_alpha"]))

    # -- 音色 -------------------------------------------------------------------

    @property
    def voices_dir(self) -> Path:
        return self.cfg.resolve_path(self.cfg.tts["voices_dir"])

    def voices(self, include_missing: bool = False) -> list[Voice]:
        out: list[Voice] = []
        for v in self._voice_data.get("voices") or []:
            out.append(Voice(id=v["id"], name=v.get("name", v["id"]), file=self.cfg.resolve_path(v["file"]),
                             emotion=v.get("emotion", ""), desc=v.get("desc", "")))
        known = {v.file.resolve() for v in out}
        d = self.voices_dir
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS and f.resolve() not in known:
                    out.append(Voice(id=f.stem, name=f.stem, file=f, source="folder"))
        itts = self.cfg.tts.get("local", {}).get("index_tts_dir")
        if itts and self._voice_data.get("include_index_tts_examples", True):
            ex = Path(itts).expanduser() / "examples"
            for f in sorted(ex.glob("voice_*.wav")) if ex.is_dir() else []:
                n = f.stem.split("_")[-1]
                out.append(Voice(id=f"example_{n}", name=f"IndexTTS 示例 {n}", file=f, source="example"))
        return out if include_missing else [v for v in out if v.available]

    def voice(self, key: str | None) -> Voice:
        voices = self.voices()
        if key:
            p = Path(key).expanduser()
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                return Voice(p.stem, p.stem, p.resolve(), source="file")
            found = _match(key, voices, "id", "name")
            if found:
                return found
            names = "、".join(v.name for v in voices) or "（空）"
            raise ValueError(f"找不到音色 {key!r}。可用音色: {names}")
        default = self.cfg.tts.get("default_voice")
        if default:
            return self.voice(default)
        if voices:
            return voices[0]
        raise ValueError(f"还没有可用音色：把 3~10 秒的人声 wav 放进 {self.voices_dir}，"
                         f"或在配置里填 tts.local.index_tts_dir 使用 IndexTTS 自带示例音色")
