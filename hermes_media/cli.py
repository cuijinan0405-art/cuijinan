"""命令行入口。

  hermes-media serve                     # 启动 MCP 服务（给 Hermes 用）
  hermes-media image "一只橘猫" -r 16:9    # 直接出图
  hermes-media tts "你好呀" -v 女声 -e 开心  # 直接配音
  hermes-media voices                    # 列出音色
  hermes-media doctor                    # 检查各后端是否连得上
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from .backends import ImageGenerator, TTSGenerator
from .config import Config, find_config_file


async def _doctor(cfg: Config) -> int:
    ok = True
    file = find_config_file()
    print(f"配置文件: {file or '未找到，使用内置默认值'}")
    print(f"输出目录: {cfg.output_dir}")

    async def check(name: str, url: str):
        nonlocal ok
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(url)
            print(f"  [OK]  {name}: {url} -> {r.status_code}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [失败] {name}: {url} -> {e.__class__.__name__}: {e}")

    img = cfg.image
    print(f"出图后端: {img['backend']}")
    if img["backend"] == "comfyui":
        await check("ComfyUI", img["comfyui"]["url"].rstrip("/") + "/system_stats")
        wf = cfg.resolve_path(img["comfyui"]["workflow"])
        print(f"  {'[OK] ' if wf.is_file() else '[缺失]'} 工作流: {wf}")
        ok &= wf.is_file()
    else:
        await check("OpenAI 兼容出图", img["openai"]["base_url"].rstrip("/") + "/models")

    t = cfg.tts
    print(f"配音后端: {t['backend']}")
    if t["backend"] == "indextts2":
        await check("IndexTTS2 服务", t["indextts2"]["url"].rstrip("/") + "/health")
    elif t["backend"] == "comfyui":
        await check("ComfyUI", t["comfyui"]["url"].rstrip("/") + "/system_stats")
    else:
        await check("OpenAI 兼容配音", t["openai"]["base_url"].rstrip("/") + "/models")
    voices = TTSGenerator(cfg).list_voices()
    print(f"  音色: {', '.join(v.stem for v in voices) or '（空，请往 voices 目录放参考音频）'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hermes-media", description="Hermes Agent 本地出图/配音工具")
    p.add_argument("-c", "--config", help="配置文件路径（默认 ./config.yaml 或 ~/.hermes-media/config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动 MCP 服务")
    s.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])

    i = sub.add_parser("image", help="出图")
    i.add_argument("prompt")
    i.add_argument("-r", "--ratio", default=None)
    i.add_argument("-n", "--negative", default=None)
    i.add_argument("-s", "--seed", type=int, default=None)
    i.add_argument("--count", type=int, default=1)

    t = sub.add_parser("tts", help="配音")
    t.add_argument("text")
    t.add_argument("-v", "--voice", default=None)
    t.add_argument("-e", "--emotion", default=None)
    t.add_argument("-a", "--alpha", type=float, default=None)

    sub.add_parser("voices", help="列出音色")
    sub.add_parser("doctor", help="检查后端连通性")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from .server import main as serve
        serve(args.config, args.transport)
        return 0

    cfg = Config.load(args.config)
    try:
        if args.cmd == "image":
            r = asyncio.run(ImageGenerator(cfg).generate(args.prompt, args.ratio, args.negative, args.seed, args.count))
        elif args.cmd == "tts":
            r = asyncio.run(TTSGenerator(cfg).speak(args.text, args.voice, args.emotion, args.alpha))
        elif args.cmd == "voices":
            for v in TTSGenerator(cfg).list_voices():
                print(v.stem)
            return 0
        else:
            return asyncio.run(_doctor(cfg))
    except Exception as e:  # noqa: BLE001
        print(f"出错了: {e}", file=sys.stderr)
        return 1
    for f in r.files:
        print(f)
    print(r.info, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
