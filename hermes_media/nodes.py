"""节点定义 + 图执行器（ComfyUI 同款思路：后端声明节点，前端画布按声明生成节点）。

前端提交的图格式（与 ComfyUI 的 API 格式类似）：
    {"<节点id>": {"type": "QwenImage", "widgets": {...}, "inputs": {"prompt": ["<上游id>", <输出序号>]}}}
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .backends import ImageGenerator, TTSGenerator
from .config import Config
from .models import ModelRegistry
from .presets import Presets

FOLLOW = "跟随风格"


@dataclass
class Port:
    name: str
    label: str
    type: str
    optional: bool = False


@dataclass
class Widget:
    name: str
    label: str
    kind: str  # text | textarea | combo | int | float
    default: Any = None
    options: list | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None


@dataclass
class NodeDef:
    type: str
    title: str
    category: str
    color: str
    run: Callable[..., Awaitable[tuple]]
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    widgets: list[Widget] = field(default_factory=list)
    output_node: bool = False
    description: str = ""

    def to_json(self) -> dict:
        return {
            "type": self.type, "title": self.title, "category": self.category, "color": self.color,
            "description": self.description, "output_node": self.output_node,
            "inputs": [vars(p) for p in self.inputs], "outputs": [vars(p) for p in self.outputs],
            "widgets": [{k: v for k, v in vars(w).items() if v is not None} for w in self.widgets],
        }


@dataclass
class Context:
    cfg: Config
    images: ImageGenerator
    tts: TTSGenerator
    presets: Presets
    job: "Job"

    def url(self, path: Path) -> str:
        return "/files/" + path.relative_to(self.cfg.output_dir).as_posix()


# ----------------------------------------------------------------------------- 节点实现
# 每个 run 返回 (输出元组, 界面展示数据或 None)


async def _text(ctx, text=""):
    return (text,), None


async def _style(ctx, style="", subject="", **_):
    st = ctx.presets.style(style) or ctx.presets.style("none")
    if not (subject or "").strip():
        raise ValueError("风格预设节点没有拿到主题文字，请连上「文本」节点")
    neg = st.negative if st.negative is not None else ctx.cfg.image["negative_prompt"]
    prompt = st.apply(subject)
    return (prompt, neg, st.ratio or ctx.cfg.image["default_ratio"]), {"text": prompt}


async def _qwen_image(ctx, prompt="", negative=None, ratio_in=None, model=None, ratio=FOLLOW, steps=50, cfg=4.0,
                      seed=-1, count=1, **_):
    use_ratio = ratio_in if ratio == FOLLOW else ratio
    r = await ctx.images.generate(prompt, use_ratio or None, negative, int(seed), int(count),
                                  steps=int(steps), cfg_scale=float(cfg), model=model or None)
    return (r.files,), {"seeds": r.info.get("seeds"), "size": r.info.get("size"), "model": model}


async def _preview_image(ctx, images=None, **_):
    return (), {"images": [ctx.url(p) for p in images or []]}


async def _voice(ctx, voice="", **_):
    v = ctx.presets.voice(voice)
    return (v.name,), {"text": v.desc} if v.desc else None


async def _emotion(ctx, emotion="", custom="", strength=0.6, **_):
    text = (custom or "").strip() or emotion
    return ({"emotion": text, "alpha": float(strength)},), None


async def _indextts2(ctx, text="", voice=None, emotion=None, **_):
    emo = emotion or {}
    r = await ctx.tts.speak(text, voice, emo.get("emotion"), emo.get("alpha"))
    return (r.files,), {"text": f"音色：{r.info.get('voice')}　情绪：{r.info.get('emotion') or '跟随音色'}"}


async def _preview_audio(ctx, audio=None, **_):
    return (), {"audio": [ctx.url(p) for p in audio or []]}


def build_nodes(cfg: Config) -> dict[str, NodeDef]:
    presets = Presets.load(cfg)
    ratios = list(cfg.image["sizes"])
    styles = [s.name for s in presets.styles]
    voices = [v.name for v in presets.voices()] or ["（还没有音色）"]
    emotions = [e.name for e in presets.emotions]
    reg = ModelRegistry(cfg)
    models = [m.name for m in reg.available()]
    try:
        default_model = reg.get(None).name
    except ValueError:
        default_model = models[0]
    defs = [
        NodeDef("TextInput", "文本", "输入", "#355", _text,
                outputs=[Port("text", "文本", "STRING")],
                widgets=[Widget("text", "文本", "textarea", "")],
                description="输入主题或要念的文字"),
        NodeDef("StylePreset", "风格预设", "出图", "#523", _style,
                inputs=[Port("subject", "主题", "STRING")],
                outputs=[Port("prompt", "提示词", "STRING"), Port("negative", "反向词", "STRING"),
                         Port("ratio", "画幅", "RATIO")],
                widgets=[Widget("style", "风格", "combo", styles[0], styles)],
                description="把主题套进预设风格，自动给出提示词、反向词和推荐画幅"),
        NodeDef("QwenImage", "千问 Image 2.1", "出图", "#335", _qwen_image,
                inputs=[Port("prompt", "提示词", "STRING"), Port("negative", "反向词", "STRING", True),
                        Port("ratio_in", "画幅", "RATIO", True)],
                outputs=[Port("images", "图片", "IMAGE")],
                widgets=[Widget("model", "模型", "combo", default_model, models),
                         Widget("ratio", "画幅", "combo", FOLLOW, [FOLLOW] + ratios),
                         Widget("steps", "步数", "int", int(cfg.image["steps"]), min=1, max=100, step=1),
                         Widget("cfg", "CFG", "float", float(cfg.image["cfg"]), min=1, max=10, step=0.5),
                         Widget("seed", "种子(-1随机)", "int", -1, min=-1, max=2**31 - 1, step=1),
                         Widget("count", "数量", "int", 1, min=1, max=8, step=1)]),
        NodeDef("PreviewImage", "预览图片", "出图", "#353", _preview_image, output_node=True,
                inputs=[Port("images", "图片", "IMAGE")]),
        NodeDef("VoicePreset", "音色预设", "配音", "#532", _voice,
                outputs=[Port("voice", "音色", "VOICE")],
                widgets=[Widget("voice", "音色", "combo", voices[0], voices)],
                description="选择参考音色；点「上传音色」可以添加自己的声音"),
        NodeDef("EmotionPreset", "情绪", "配音", "#552", _emotion,
                outputs=[Port("emotion", "情绪", "EMOTION")],
                widgets=[Widget("emotion", "情绪", "combo", emotions[0], emotions),
                         Widget("custom", "自定义(可空)", "text", ""),
                         Widget("strength", "强度", "float", 1.0, min=0, max=1, step=0.05)]),
        NodeDef("IndexTTS2", "IndexTTS2 配音", "配音", "#335", _indextts2,
                inputs=[Port("text", "文本", "STRING"), Port("voice", "音色", "VOICE", True),
                        Port("emotion", "情绪", "EMOTION", True)],
                outputs=[Port("audio", "音频", "AUDIO")]),
        NodeDef("PreviewAudio", "试听", "配音", "#353", _preview_audio, output_node=True,
                inputs=[Port("audio", "音频", "AUDIO")]),
    ]
    return {d.type: d for d in defs}


def emotion_alphas(cfg: Config) -> dict[str, float]:
    return {e.name: e.alpha for e in Presets.load(cfg).emotions}


# ----------------------------------------------------------------------------- 执行器


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued | running | done | error
    current: str | None = None
    done: list[str] = field(default_factory=list)
    results: dict[str, dict] = field(default_factory=dict)
    error: str | None = None
    error_node: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class GraphError(ValueError):
    pass


async def execute(graph: dict, cfg: Config, images: ImageGenerator, tts: TTSGenerator, job: Job) -> Job:
    nodes = build_nodes(cfg)
    ctx = Context(cfg, images, tts, Presets.load(cfg), job)
    cache: dict[str, tuple] = {}
    visiting: set[str] = set()

    async def evaluate(nid: str) -> tuple:
        if nid in cache:
            return cache[nid]
        if nid in visiting:
            raise GraphError("连线成环了")
        if nid not in graph:
            raise GraphError(f"连线指向了不存在的节点 {nid}")
        visiting.add(nid)
        spec = graph[nid]
        d = nodes[spec["type"]]
        kwargs = dict(spec.get("widgets") or {})
        links = spec.get("inputs") or {}
        for port in d.inputs:
            link = links.get(port.name)
            if link is None:
                if not port.optional:
                    raise GraphError(f"「{d.title}」的输入「{port.label}」没有连线", nid)
                continue
            src_id, slot = str(link[0]), int(link[1])
            kwargs[port.name] = (await evaluate(src_id))[slot]
        job.current = nid
        try:
            out, ui = await d.run(ctx, **kwargs)
        except GraphError:
            raise
        except Exception:
            job.error_node = nid
            raise
        if ui:
            job.results[nid] = ui
        job.done.append(nid)
        visiting.discard(nid)
        cache[nid] = out
        return out

    job.status = "running"
    try:
        for nid, n in graph.items():
            if n.get("type") not in nodes:
                raise GraphError(f"未知节点类型: {n.get('type')}")
        outputs = [nid for nid, n in graph.items() if nodes[n["type"]].output_node]
        if not outputs:
            raise GraphError("画布上没有输出节点（预览图片 / 试听），没东西可运行")
        for nid in outputs:
            await evaluate(nid)
        job.status = "done"
    except GraphError as e:
        job.status, job.error = "error", str(e.args[0])
        if len(e.args) > 1:
            job.error_node = e.args[1]
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.status, job.error = "error", str(e)
    finally:
        job.current = None
        job.finished = time.time()
    return job
