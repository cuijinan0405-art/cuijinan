"""千问 Qwen-Image 2.1 常驻出图服务（diffusers，只用标准库起 HTTP）。

由 hermes-media 自动启动，也可以手动跑：
    python workers/image_worker.py --model Qwen/Qwen-Image-2.1 --port 7862

需要的环境：torch、diffusers（新版，支持 Qwen-Image 2.1）、transformers、accelerate。

接口：
    GET  /health   {"state": "loading" | "ready" | "error", "progress": {...}}
    POST /generate JSON {prompt, negative_prompt, width, height, steps, cfg, seed} → image/png
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _watchdog import watch_parent

STATE = {"state": "loading", "error": None, "progress": None, "model": None}
PIPE = None
CALL_PARAMS: set[str] = set()
LOCK = threading.Lock()
ARGS: argparse.Namespace


def load(args: argparse.Namespace) -> None:
    global PIPE, CALL_PARAMS
    try:
        import torch
        from diffusers import DiffusionPipeline

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        print(f">> 加载模型 {args.model}（{args.dtype}）…", flush=True)
        t0 = time.time()
        pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(args.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        PIPE = pipe
        CALL_PARAMS = set(inspect.signature(pipe.__call__).parameters)
        STATE.update(state="ready", model=f"{args.model} ({type(pipe).__name__})")
        print(f">> 模型就绪（{time.time() - t0:.0f}s）: {type(pipe).__name__}", flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        STATE.update(state="error", error=f"{e.__class__.__name__}: {e}")


def generate(body: dict) -> bytes:
    import torch

    steps = int(body.get("steps") or 50)
    kwargs = {
        "prompt": body["prompt"],
        "negative_prompt": body.get("negative_prompt") or " ",
        "width": int(body.get("width") or 1328),
        "height": int(body.get("height") or 1328),
        "num_inference_steps": steps,
    }
    cfg = float(body.get("cfg") or 4.0)
    # Qwen-Image 系列用 true_cfg_scale，其它管线用 guidance_scale
    kwargs["true_cfg_scale" if "true_cfg_scale" in CALL_PARAMS else "guidance_scale"] = cfg
    seed = int(body.get("seed") if body.get("seed") is not None else 0)
    gen_device = "cpu" if ARGS.offload else ARGS.device
    kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(seed)

    if "callback_on_step_end" in CALL_PARAMS:
        def on_step(pipe, step, timestep, cb_kwargs):
            STATE["progress"] = {"step": step + 1, "total": steps}
            return cb_kwargs
        kwargs["callback_on_step_end"] = on_step

    kwargs = {k: v for k, v in kwargs.items() if k in CALL_PARAMS or k == "prompt"}
    with LOCK:
        STATE["progress"] = {"step": 0, "total": steps}
        try:
            image = PIPE(**kwargs).images[0]
        finally:
            STATE["progress"] = None
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 不刷 /health 的访问日志
        pass

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
            return self._json(200, STATE)
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            return self._json(404, {"error": "not found"})
        if STATE["state"] != "ready":
            return self._json(503, {"error": f"模型还没就绪: {STATE['state']} {STATE['error'] or ''}"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if not (body.get("prompt") or "").strip():
                return self._json(400, {"error": "prompt 不能为空"})
            t0 = time.time()
            png = generate(body)
            print(f">> 出图 {body.get('width')}x{body.get('height')} 用时 {time.time() - t0:.1f}s", flush=True)
            self._send(200, png, "image/png")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"{e.__class__.__name__}: {e}"})


def main():
    global ARGS
    p = argparse.ArgumentParser(description="Qwen-Image 2.1 出图服务")
    p.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--offload", action="store_true", help="显存不够时把模型放内存，按需搬到显卡")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--parent-pid", type=int, default=None, help="该进程退出时自动退出")
    p.add_argument("--port", type=int, default=7862)
    ARGS = p.parse_args()
    watch_parent(ARGS.parent_pid)
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)  # 先开端口，加载时也能查询状态
    threading.Thread(target=load, args=(ARGS,), daemon=True).start()
    print(f">> 出图服务: http://{ARGS.host}:{ARGS.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
