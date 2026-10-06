"""MCP 服务：把出图 / 配音包装成 Hermes Agent 能直接调用的工具。

在 Hermes 里注册后（服务名 media），工具名为：
  mcp_media_generate_image / mcp_media_text_to_speech / mcp_media_list_voices
"""

from __future__ import annotations

from .backends import ImageGenerator, Result, TTSGenerator
from .config import Config

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """本地出图 / 配音工具（千问 Qwen-Image 2.1 + IndexTTS2）。
- 用户说「画/出图/生成图片/来张图」→ 调 generate_image。提示词可以直接用中文，尽量补充主体、风格、光线、构图等细节。
- 用户说「配音/念/朗读/读出来/转语音」→ 调 text_to_speech。用户提到情绪（开心、生气、温柔…）就填 emotion。
- 用户问有哪些声音/音色 → 调 list_voices。
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
    images = ImageGenerator(cfg)
    tts = TTSGenerator(cfg)
    mcp = _Server("hermes-media", instructions=INSTRUCTIONS)

    @mcp.tool()
    async def generate_image(prompt: str, ratio: str = "", negative_prompt: str = "",
                             seed: int = -1, count: int = 1) -> str:
        """用本地千问 Qwen-Image 2.1 生成图片，返回图片文件路径。

        Args:
            prompt: 画面描述，中英文都可以，越具体越好；图里要出现的文字用引号括起来。
            ratio: 画幅比例：1:1、16:9、9:16、4:3、3:4、3:2、2:3，或 宽x高。留空用默认。
            negative_prompt: 不想出现的内容，可留空。
            seed: 随机种子，-1 表示随机；想复现上一张图时传入上次的种子。
            count: 生成几张（1~8）。
        """
        r = await images.generate(prompt, ratio or None, negative_prompt or None,
                                  None if seed < 0 else seed, count)
        return _format(r, "出图")

    @mcp.tool()
    async def text_to_speech(text: str, voice: str = "", emotion: str = "", emo_alpha: float = -1) -> str:
        """用本地 IndexTTS2 把文字配成语音（克隆参考音色），返回 wav 文件路径。

        Args:
            text: 要念的文字。
            voice: 音色名（见 list_voices），支持模糊匹配；留空用默认音色。
            emotion: 情绪描述，如「开心」「愤怒地质问」「温柔地哄小孩」，留空则跟随参考音色。
            emo_alpha: 情绪强度 0~1，-1 表示用默认值（0.6）。
        """
        r = await tts.speak(text, voice or None, emotion or None, None if emo_alpha < 0 else emo_alpha)
        return _format(r, "配音")

    @mcp.tool()
    async def list_voices() -> str:
        """列出可用的参考音色（voices 目录里的音频文件）。"""
        voices = tts.list_voices()
        if not voices:
            return f"还没有音色。把 3~10 秒干净人声的 wav 放进 {tts.voices_dir}，文件名就是音色名。"
        default = cfg.tts.get("default_voice") or voices[0].stem
        return "可用音色：\n" + "\n".join(
            f"- {v.stem}{'（默认）' if v.stem == default else ''}" for v in voices)

    return mcp


def main(config_path: str | None = None, transport: str = "stdio") -> None:
    build_server(Config.load(config_path)).run(transport=transport)
