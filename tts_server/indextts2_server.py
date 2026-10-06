"""IndexTTS2 常驻 HTTP 服务（只用标准库，模型只加载一次）。

放在 index-tts 仓库的环境里运行（不需要额外装依赖）：

    cd /path/to/index-tts
    uv run python /path/to/hermes-media/tts_server/indextts2_server.py --model-dir checkpoints --port 7861

IndexTTS2.5 加 --version 2.5（可配 --lang ZH/EN）。

接口：
    GET  /health
    POST /tts   JSON:
        text          要念的文字（必填）
        voice_b64     参考音色音频 base64（或用 voice_path 传服务器本地路径）
        voice_name    参考音频文件名（只用来猜扩展名）
        emo_text      情绪描述，如 "开心"，为空则跟随参考音色
        emo_alpha     情绪强度，默认 0.6
        emo_vector    8 维情绪向量 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]（可选）
    返回 audio/wav
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TTS = None
LOCK = threading.Lock()  # GPU 推理串行执行
ARGS: argparse.Namespace
CACHE = Path(tempfile.gettempdir()) / "hermes_media_voices"


def load_model(args: argparse.Namespace):
    sys.path.insert(0, os.getcwd())
    model_dir = args.model_dir
    cfg_path = os.path.join(model_dir, "config.yaml")
    if args.version == "2.5":
        from indextts.infer_v2_5 import IndexTTS2
        return IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, use_bf16=not args.no_half,
                         use_qwen_emo=True)
    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, use_fp16=not args.no_half,
                     use_cuda_kernel=False, use_deepspeed=False)


def voice_file(body: dict) -> str:
    if body.get("voice_path"):
        return body["voice_path"]
    if not body.get("voice_b64"):
        raise ValueError("需要 voice_b64 或 voice_path（参考音色）")
    data = base64.b64decode(body["voice_b64"])
    suffix = Path(body.get("voice_name") or "voice.wav").suffix or ".wav"
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha1(data).hexdigest() + suffix)
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def synthesize(body: dict) -> bytes:
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("text 不能为空")
    kwargs = {
        "spk_audio_prompt": voice_file(body),
        "text": text,
        "emo_alpha": float(body.get("emo_alpha") or 0.6),
        "use_random": False,
        "verbose": False,
    }
    if body.get("emo_vector"):
        kwargs["emo_vector"] = [float(x) for x in body["emo_vector"]]
    elif body.get("emo_text"):
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = body["emo_text"]
    else:
        kwargs["emo_alpha"] = 1.0  # 没有情绪要求就完全跟随参考音色
    if ARGS.version == "2.5":
        kwargs["lang"] = body.get("lang") or ARGS.lang
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with LOCK:
            TTS.infer(output_path=out, **kwargs)
        return Path(out).read_bytes()
    finally:
        Path(out).unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "version": ARGS.version})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/tts":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            t0 = time.time()
            wav = synthesize(body)
            print(f">> {len(body.get('text', ''))} 字，用时 {time.time() - t0:.1f}s", flush=True)
            self._send(200, wav, "audio/wav")
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"{e.__class__.__name__}: {e}"})


def main():
    global TTS, ARGS
    p = argparse.ArgumentParser(description="IndexTTS2 HTTP 服务")
    p.add_argument("--model-dir", default="checkpoints")
    p.add_argument("--version", default="2", choices=["2", "2.5"])
    p.add_argument("--lang", default="ZH", help="IndexTTS2.5 默认语言")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--no-half", action="store_true", help="不用 fp16/bf16（显卡不支持时）")
    ARGS = p.parse_args()
    print(f">> 加载 IndexTTS{ARGS.version} 模型: {ARGS.model_dir}", flush=True)
    TTS = load_model(ARGS)
    print(f">> 已就绪: http://{ARGS.host}:{ARGS.port}", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
