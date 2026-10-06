"""MCP 服务：把出图 / 配音包装成 Hermes Agent 能直接调用的工具。

在 Hermes 里注册后（服务名 media），工具名为：
  mcp_media_generate_image / mcp_media_text_to_speech / mcp_media_list_voices / mcp_media_list_styles
"""

from __future__ import annotations

from .backends import ImageGenerator, Result, TTSGenerator
from .config import Config
from .models import ModelRegistry
from .presets import Presets
from .workers import WorkerManager

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """本地出图 / 配音工具（千问 Qwen-Image 2.1 + IndexTTS2）。
- 用户说「画/出图/生成图片/来张图」→ 调 generate_image。prompt 写画面主体，style 选一个预设风格（见 list_styles）。
- 用户说「配音/念/朗读/读出来/转语音」→ 调 text_to_speech。voice 选预设音色，用户提到情绪就填 emotion。
- 用户问有哪些风格 / 音色 → 调 list_styles / list_voices。
- 工具返回的 MEDIA:/路径 行要原样放进最终回复，这样聊天平台会直接把文件发给用户。"""


def _format(result: Result, kind: str) -> str:
    lines = [f"{kind}完成，共 {len(result.files)} 个文件："]
    lines += [f"MEDIA:{p}" for p in result.files]
    extras = ", ".join(f"{k}={v}" for k, v in result.info.items() if v not in (None, "", []))
    if extras:
        lines.append(f"（{extras}）")
    return "\n".join(lines)


def build_server(cfg: Config | None = None):
    cfg = cfg or Config.load()
    workers = WorkerManager(cfg)  # 网页界面已经启动了模型的话会直接复用
    images = ImageGenerator(cfg, workers)
    tts = TTSGenerator(cfg, workers)
    mcp = _Server("hermes-media", instructions=INSTRUCTIONS)

    @mcp.tool()
    async def generate_image(prompt: str, style: str = "", ratio: str = "", negative_prompt: str = "",
                             seed: int = -1, count: int = 1, model: str = "") -> str:
        """用本地千问 Qwen-Image 2.1 生成图片，返回图片文件路径。

        Args:
            prompt: 画面主体描述，中英文都可以；图里要出现的文字用引号括起来。风格词交给 style。
            style: 预设风格名（如「电影大片」「国风水墨」「小红书封面」，见 list_styles），会自动补全风格提示词和推荐画幅。留空不加风格。
            ratio: 画幅比例：1:1、16:9、9:16、4:3、3:4、3:2、2:3，或 宽x高。留空用风格推荐的画幅。
            negative_prompt: 不想出现的内容，可留空。
            seed: 随机种子，-1 表示随机；想复现上一张图时传入上次的种子。
            count: 生成几张（1~8）。
            model: 出图模型名（见 list_styles 末尾的可用模型），留空用默认模型。用户没指定就留空。
        """
        r = await images.generate(prompt, ratio or None, negative_prompt or None,
                                  None if seed < 0 else seed, count, style=style or None, model=model or None)
        return _format(r, "出图")

    @mcp.tool()
    async def text_to_speech(text: str, voice: str = "", emotion: str = "", emo_alpha: float = -1) -> str:
        """用本地 IndexTTS2 把文字配成语音（克隆参考音色），返回 wav 文件路径。

        Args:
            text: 要念的文字。
            voice: 预设音色名（见 list_voices），支持模糊匹配；留空用默认音色。
            emotion: 预设情绪（开心、温柔、悲伤、生气、激动热情、低沉神秘…）或任意一句情绪描述；留空用音色的默认情绪。
            emo_alpha: 情绪强度 0~1，-1 表示用预设值。
        """
        r = await tts.speak(text, voice or None, emotion or None, None if emo_alpha < 0 else emo_alpha)
        return _format(r, "配音")

    @mcp.tool()
    async def list_voices() -> str:
        """列出可用的预设音色和情绪。"""
        p = Presets.load(cfg)
        voices = p.voices()
        if not voices:
            return f"还没有音色。把 3~10 秒干净人声的 wav 放进 {p.voices_dir}，文件名就是音色名。"
        lines = ["可用音色（第一个是默认）："]
        lines += [f"- {v.name}" + (f"：{v.desc}" if v.desc else "") for v in voices]
        lines.append("情绪预设：" + "、".join(e.name for e in p.emotions))
        return "\n".join(lines)

    @mcp.tool()
    async def list_styles() -> str:
        """列出出图的预设风格（按分类）和可用的出图模型。"""
        groups: dict[str, list[str]] = {}
        for s in Presets.load(cfg).styles:
            groups.setdefault(s.category or "其它", []).append(f"{s.name}({s.ratio or cfg.image['default_ratio']})")
        models = "、".join(m.name for m in ModelRegistry(cfg).available())
        return ("预设风格（括号里是推荐画幅）：\n" + "\n".join(f"- {k}：{'、'.join(v)}" for k, v in groups.items())
                + f"\n可用出图模型：{models}")

    return mcp


def main(config_path: str | None = None, transport: str = "stdio") -> None:
    build_server(Config.load(config_path)).run(transport=transport)
