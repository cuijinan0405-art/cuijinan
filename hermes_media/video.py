"""视频合成：算时间轴、写字幕、拼 ffmpeg 命令。

这里只做纯计算和调用 ffmpeg，不碰模型；出图 / 配音在 storyboard.py 里。
做法：Python 先把每个镜头、每句话的起止时间算好，再翻译成一条 ffmpeg 命令一次渲染完，
不生成中间视频片段。
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

# 成片分辨率（出图用同一个比例，千问会出对应的推荐尺寸，再放大到这里）
VIDEO_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "3:4": (1080, 1440),
    "4:3": (1440, 1080),
}

MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right", "none")
# transition: random 时从这里挑（xfade 支持的转场还有很多，脚本里可以直接写名字）
RANDOM_TRANSITIONS = ("fade", "dissolve", "slideleft", "slideup", "wipeleft", "circleopen", "smoothleft")

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…\n])")
_CLAUSE_END = re.compile(r"(?<=[，,、：:])")
_TRAILING_PUNCT = "，,、。；;：: \n"


def video_size(ratio: str | None) -> tuple[int, int]:
    ratio = (ratio or "9:16").replace("：", ":").strip()
    if ratio in VIDEO_SIZES:
        return VIDEO_SIZES[ratio]
    raise ValueError(f"视频不支持的比例 {ratio!r}，可选: {', '.join(VIDEO_SIZES)}")


def default_font() -> str:
    if sys.platform == "win32":
        return "Microsoft YaHei"
    if sys.platform == "darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


# ----------------------------------------------------------------------------- 文本


def split_sentences(text: str) -> list[str]:
    """按句号、问号、叹号、分号、换行拆句；每句单独配音，字幕时间就等于这句的音频长度。"""
    parts = [p.strip() for p in _SENTENCE_END.split(text or "")]
    return [p for p in parts if p.strip(_TRAILING_PUNCT)]


def wrap_text(text: str, max_chars: int) -> list[str]:
    """字幕折行：优先在逗号处断开，一行不超过 max_chars 个字，行尾标点去掉。"""
    lines: list[str] = []
    cur = ""
    for clause in _CLAUSE_END.split(text.strip()):
        while len(clause) > max_chars:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(clause[:max_chars])
            clause = clause[max_chars:]
        if len(cur) + len(clause.rstrip(_TRAILING_PUNCT)) > max_chars and cur:
            lines.append(cur)
            cur = ""
        cur += clause
    if cur:
        lines.append(cur)
    lines = [ln.strip().rstrip(_TRAILING_PUNCT) for ln in lines]
    return [ln for ln in lines if ln]


# ----------------------------------------------------------------------------- 时间轴


@dataclass
class Line:
    """一句旁白：一个 wav + 它在字幕里的文字。"""
    text: str
    audio: Path | None
    duration: float


@dataclass
class Shot:
    image: Path
    lines: list[Line] = field(default_factory=list)
    motion: str = "zoom_in"
    transition: str = "fade"  # 进入这个镜头时的转场（第一个镜头忽略）
    hold: float = 0.0  # 额外停留秒数


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


@dataclass
class Timeline:
    shots: list[Shot]
    starts: list[float]  # 每个镜头开始时间
    durations: list[float]  # 每个镜头时长（不含转场多出来的部分）
    audio: list[tuple[Path, float]]  # (wav, 开始时间)
    cues: list[Cue]
    total: float


def build_timeline(shots: list[Shot], lead: float = 0.3, tail: float = 0.5, gap: float = 0.15,
                   max_chars: int = 16, empty_hold: float = 2.0) -> Timeline:
    """镜头时长 = 开头留白 + 各句配音 + 句间停顿 + 结尾留白 + hold。全片时长 = 所有镜头之和。"""
    if not shots:
        raise ValueError("分镜脚本里没有镜头")
    starts, durations, audio, cues = [], [], [], []
    t = 0.0
    for shot in shots:
        starts.append(t)
        cursor = t + lead
        for i, line in enumerate(shot.lines):
            if i:
                cursor += gap
            if line.audio is not None:
                audio.append((line.audio, cursor))
            cues += _cues_for(line, cursor, max_chars)
            cursor += line.duration
        dur = (cursor - t) + tail + max(shot.hold, 0)
        if not shot.lines:
            dur = max(shot.hold, 0) or empty_hold
        durations.append(dur)
        t += dur
    return Timeline(shots, starts, durations, audio, cues, t)


def _cues_for(line: Line, start: float, max_chars: int) -> list[Cue]:
    """一句太长时拆成几条字幕（每条最多两行），按字数分配时间。"""
    rows = wrap_text(line.text, max_chars)
    if not rows:
        return []
    groups = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    total_chars = sum(len(r) for r in rows)
    out, t = [], start
    for g in groups:
        d = line.duration * sum(len(r) for r in g) / total_chars
        out.append(Cue(t, t + d, g))
        t += d
    return out


# ----------------------------------------------------------------------------- 字幕文件


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _ass_escape(s: str) -> str:
    return s.replace("\\", "＼").replace("{", "｛").replace("}", "｝")


def write_ass(cues: list[Cue], path: Path, width: int, height: int, font: str, size: int = 0,
              margin_v: int = 0) -> Path:
    vertical = height > width
    size = size or round(min(width, height) * 0.06)
    # 竖屏底部留出抖音 / 视频号按钮和文案的位置
    margin_v = margin_v or round(height * (0.19 if vertical else 0.07))
    outline = max(2, round(size / 16))
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,"
        f"{outline},1,2,{round(width * 0.07)},{round(width * 0.07)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = "".join(
        f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,"
        + r"\N".join(_ass_escape(x) for x in c.lines) + "\n"
        for c in cues
    )
    path.write_text(head + events, encoding="utf-8")
    return path


def write_srt(cues: list[Cue], path: Path) -> Path:
    blocks = [f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n" + "\n".join(c.lines)
              for i, c in enumerate(cues, 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- ffmpeg


@dataclass
class RenderOptions:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    transition_duration: float = 0.5
    zoom: float = 0.12
    codec: str = "libx264"
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"
    loudness: float = -14.0
    subtitles: str | None = None  # 相对 ffmpeg 工作目录的 .ass 文件名
    fonts_dir: Path | None = None


def _filter_path(p: Path) -> str:
    """滤镜参数里的路径：统一用 /，冒号要转义（Windows 的 D:/...）。"""
    return "'" + p.as_posix().replace("'", r"'\''").replace(":", r"\:") + "'"


def _motion_filter(motion: str, frames: int, w: int, h: int, fps: int, zoom: float) -> str:
    n = max(frames - 1, 1)
    z_max = f"{1 + zoom:.4f}"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if motion == "zoom_out":
        z, x, y = f"{z_max}-{zoom:.4f}*on/{n}", cx, cy
    elif motion == "pan_left":
        z, x, y = z_max, f"(iw-iw/zoom)*(1-on/{n})", cy
    elif motion == "pan_right":
        z, x, y = z_max, f"(iw-iw/zoom)*on/{n}", cy
    elif motion == "none":
        z, x, y = "1", "0", "0"
    else:  # zoom_in
        z, x, y = f"1+{zoom:.4f}*on/{n}", cx, cy
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"


def pick_transition(name: str, index: int, seed: str = "") -> str:
    if name == "random":
        return random.Random(f"{seed}:{index}").choice(RANDOM_TRANSITIONS)
    return name or "fade"


def build_command(ffmpeg: str, tl: Timeline, out: Path, opt: RenderOptions, seed: str = "") -> list[str]:
    """把时间轴翻译成一条 ffmpeg 命令。"""
    W, H, fps = opt.width, opt.height, opt.fps
    T = opt.transition_duration if len(tl.shots) > 1 else 0.0
    use_xfade = T > 0 and any(s.transition != "none" for s in tl.shots[1:])
    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    fc: list[str] = []
    # 图片：先放大到 2 倍再 zoompan，避免低分辨率上运镜发抖。注意不能加 -loop 1，
    # 否则 zoompan 会对每一帧输入都生成一整段。
    for i, shot in enumerate(tl.shots):
        args += ["-i", str(shot.image)]
        extra = T if use_xfade and i < len(tl.shots) - 1 else 0.0
        frames = max(round((tl.durations[i] + extra) * fps), 1)
        fc.append(
            f"[{i}:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
            f"{_motion_filter(shot.motion, frames, W, H, fps, opt.zoom)},setsar=1,format=yuv420p[v{i}]"
        )
    # 转场：第 i 个转场落在第 i 个镜头开始的位置，前一个镜头的画面多渲染 T 秒垫在下面
    prev = "v0"
    if len(tl.shots) > 1 and use_xfade:
        for i in range(1, len(tl.shots)):
            tr = pick_transition(tl.shots[i].transition, i, seed)
            if tr == "none":
                tr, dur = "fade", 0.04
            else:
                dur = T
            fc.append(f"[{prev}][v{i}]xfade=transition={tr}:duration={dur}:offset={tl.starts[i]:.3f}[x{i}]")
            prev = f"x{i}"
    elif len(tl.shots) > 1:
        fc.append("".join(f"[v{i}]" for i in range(len(tl.shots))) + f"concat=n={len(tl.shots)}:v=1:a=0[x]")
        prev = "x"
    if opt.subtitles:
        sub = f"subtitles=filename={_filter_path(Path(opt.subtitles))}"
        if opt.fonts_dir:
            sub += f":fontsdir={_filter_path(opt.fonts_dir)}"
        fc.append(f"[{prev}]{sub}[vout]")
    else:
        fc.append(f"[{prev}]null[vout]")
    # 音频：每句放到时间轴上的位置再混在一起，补齐到全片长度，统一响度
    k = len(tl.shots)
    if tl.audio:
        for j, (wav, start) in enumerate(tl.audio):
            args += ["-i", str(wav)]
            ms = max(int(round(start * 1000)), 0)
            fc.append(f"[{k + j}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}:all=1[a{j}]")
        mix = "".join(f"[a{j}]" for j in range(len(tl.audio)))
        fc.append(f"{mix}amix=inputs={len(tl.audio)}:normalize=0:dropout_transition=0,apad,"
                  f"atrim=0:{tl.total:.3f},loudnorm=I={opt.loudness}:TP=-1.5:LRA=11,aresample=48000[aout]")
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        fc.append(f"[{k}:a]atrim=0:{tl.total:.3f}[aout]")
    args += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
             "-t", f"{tl.total:.3f}", "-r", str(fps),
             "-c:v", opt.codec, "-pix_fmt", "yuv420p"]
    if opt.codec == "libx264":
        args += ["-crf", str(opt.crf), "-preset", opt.preset]
    elif "nvenc" in opt.codec:
        args += ["-cq", str(opt.crf), "-preset", "p5"]
    args += ["-c:a", "aac", "-b:a", opt.audio_bitrate, "-movflags", "+faststart", str(out)]
    return args


def find_ffmpeg(configured: str = "") -> str:
    """ffmpeg 查找顺序：配置里写的路径 → 系统 PATH → pip 包 imageio-ffmpeg 自带的。"""
    if configured:
        path = shutil.which(configured) or (configured if Path(configured).is_file() else None)
        if not path:
            raise FileNotFoundError(f"找不到配置里的 ffmpeg: {configured}")
        return path
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            "找不到 ffmpeg：请在 config.yaml 的 video.ffmpeg 填 ffmpeg 路径，或 pip install imageio-ffmpeg") from e


def audio_duration(path: Path, ffmpeg: str | None = None) -> float:
    """音频时长（秒）。wav 直接读文件头；其它格式用 ffmpeg 读。"""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        pass
    if not ffmpeg:
        raise ValueError(f"读不出音频时长: {path}")
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise ValueError(f"读不出音频时长: {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def run_ffmpeg(args: list[str], cwd: Path) -> None:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg 合成失败（code {r.returncode}）:\n{tail}")
