"""命令行入口。

  hermes-media ui                            # 打开节点画布界面（浏览器）
  hermes-media serve                         # 启动 MCP 服务（给 Hermes 用）
  hermes-media image "一只橘猫" -s 日系动漫     # 直接出图
  hermes-media tts "你好呀" -v 温柔女声 -e 开心  # 直接配音
  hermes-media presets                       # 列出风格 / 音色 / 情绪预设
  hermes-media models                        # 列出出图模型（含可下载的）
  hermes-media download qwen-image-2.1 --source hf-mirror   # 下载模型
  hermes-media doctor                        # 检查配置和模型状态
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from .backends import ImageGenerator, TTSGenerator
from .config import Config, find_config_file
from .downloads import SOURCES, DownloadManager
from .models import ModelRegistry
from .presets import Presets
from .workers import WorkerManager


async def _doctor(cfg: Config) -> int:
    ok = True
    file = find_config_file()
    print(f"配置文件: {file or '未找到，使用内置默认值（复制 config.example.yaml 为 config.yaml 来修改）'}")
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

    workers = WorkerManager(cfg)
    img = cfg.image
    print(f"出图后端: {img['backend']}")
    if img["backend"] == "local":
        w = workers["image"]
        print(f"  模型: {img['local']['model']}")
        print(f"  启动命令: {' '.join(w.argv)}")
        print(f"  状态: {w.status()['state']}（用到时自动启动；日志 {w.log_path}）")
    elif img["backend"] == "comfyui":
        await check("ComfyUI", img["comfyui"]["url"].rstrip("/") + "/system_stats")
    else:
        await check("OpenAI 兼容出图", img["openai"]["base_url"].rstrip("/") + "/models")

    t = cfg.tts
    print(f"配音后端: {t['backend']}")
    if t["backend"] == "local":
        d = t["local"].get("index_tts_dir")
        w = workers["tts"]
        if not d:
            ok = False
            print("  [缺失] tts.local.index_tts_dir 没填（你本地 index-tts 的目录）")
        else:
            exists = (Path(d).expanduser() / "indextts").is_dir()
            ok &= exists
            print(f"  {'[OK] ' if exists else '[失败]'} IndexTTS 目录: {d}")
            print(f"  启动命令: {' '.join(w.argv)}  (在 {d} 下执行)")
            print(f"  状态: {w.status()['state']}")
    elif t["backend"] == "comfyui":
        await check("ComfyUI", t["comfyui"]["url"].rstrip("/") + "/system_stats")
    else:
        await check("OpenAI 兼容配音", t["openai"]["base_url"].rstrip("/") + "/models")
    p = Presets.load(cfg)
    print(f"预设: {len(p.styles)} 种风格，{len(p.voices())} 个可用音色，{len(p.emotions)} 种情绪")
    missing = [v for v in p.voices(include_missing=True) if not v.available]
    if missing:
        print("  这些预设音色还缺参考音频（放进去就能用）: " + "、".join(f"{v.name} → {v.file}" for v in missing))
    return 0 if ok else 1


def _download(cfg: Config, model_id: str, source: str | None) -> int:
    import time

    dm = DownloadManager(cfg)
    t = dm.start(model_id, source)
    print(f"从 {SOURCES[t.source]} 下载 {model_id} → {dm.registry.models_dir / model_id}")
    while t.state == "running":
        time.sleep(2)
        st = dm.status()[model_id]
        total = f" / {st['total'] / 1024 ** 3:.1f} GB" if st["total"] else ""
        print(f"\r  已下载 {st['downloaded'] / 1024 ** 3:.2f} GB{total}   ", end="", flush=True)
    print()
    if t.state == "error":
        print(f"下载失败: {t.error}", file=sys.stderr)
        return 1
    print("下载完成，可以在界面「千问 Image 2.1」节点的「模型」里选择了。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hermes-media", description="Hermes Agent 本地出图/配音工具")
    p.add_argument("-c", "--config", help="配置文件路径（默认 ./config.yaml 或 ~/.hermes-media/config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("ui", help="打开节点画布界面")
    u.add_argument("--host", default=None)
    u.add_argument("--port", type=int, default=None)
    u.add_argument("--no-browser", action="store_true")

    s = sub.add_parser("serve", help="启动 MCP 服务")
    s.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])

    i = sub.add_parser("image", help="出图")
    i.add_argument("prompt")
    i.add_argument("-s", "--style", default=None, help="预设风格名")
    i.add_argument("-r", "--ratio", default=None)
    i.add_argument("-n", "--negative", default=None)
    i.add_argument("--seed", type=int, default=None)
    i.add_argument("--count", type=int, default=1)
    i.add_argument("-m", "--model", default=None, help="出图模型名或 id")

    t = sub.add_parser("tts", help="配音")
    t.add_argument("text")
    t.add_argument("-v", "--voice", default=None)
    t.add_argument("-e", "--emotion", default=None)
    t.add_argument("-a", "--alpha", type=float, default=None)

    sub.add_parser("presets", help="列出风格 / 音色 / 情绪预设")
    sub.add_parser("models", help="列出出图模型")
    d = sub.add_parser("download", help="下载出图模型")
    d.add_argument("model_id")
    d.add_argument("--source", choices=list(SOURCES), default=None)
    sub.add_parser("doctor", help="检查后端连通性")

    args = p.parse_args(argv)

    if args.cmd == "ui":
        from .webapp import serve as serve_ui
        serve_ui(Config.load(args.config), args.host, args.port, False if args.no_browser else None)
        return 0

    if args.cmd == "serve":
        from .server import main as serve
        serve(args.config, args.transport)
        return 0

    cfg = Config.load(args.config)
    try:
        if args.cmd == "image":
            r = asyncio.run(ImageGenerator(cfg, WorkerManager(cfg)).generate(
                args.prompt, args.ratio, args.negative, args.seed, args.count, style=args.style, model=args.model))
        elif args.cmd == "tts":
            r = asyncio.run(TTSGenerator(cfg, WorkerManager(cfg)).speak(args.text, args.voice, args.emotion, args.alpha))
        elif args.cmd == "models":
            reg = ModelRegistry(cfg)
            print(f"模型目录: {reg.models_dir}")
            for e in reg.entries():
                state = "可用" if e.installed else ("可下载" if e.sources else "不可用")
                print(f"  [{state}] {e.id:34} {e.name}")
            return 0
        elif args.cmd == "download":
            return _download(cfg, args.model_id, args.source)
        elif args.cmd == "presets":
            p = Presets.load(cfg)
            print("风格: " + "、".join(s.name for s in p.styles))
            print("音色: " + ("、".join(v.name for v in p.voices()) or "（空）"))
            print("情绪: " + "、".join(e.name for e in p.emotions))
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
