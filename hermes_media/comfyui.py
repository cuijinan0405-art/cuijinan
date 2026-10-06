"""ComfyUI 工作流运行器。

参数注入沿用 Pixelle-MCP (https://github.com/ATH-MaaS/Pixelle-MCP, MIT) 的节点标题 DSL：

    $<参数名>.[~]<字段名>[!][:<描述>]

例如把 CLIPTextEncode 节点标题改成  ``$prompt.text!:正向提示词``，
调用时传 ``{"prompt": "一只猫"}`` 就会写进该节点的 ``inputs.text``。
``~`` 表示该字段是文件（本地路径会先上传到 ComfyUI），``!`` 表示必填。

没有标注的工作流也能用：会自动找到采样器连着的正/负提示词节点、
种子和 latent 尺寸节点（够覆盖官方 Qwen-Image 模板）。
"""

from __future__ import annotations

import asyncio
import copy
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_DSL = re.compile(r"^\$(?P<param>[A-Za-z_][\w]*)\.(?P<url>~)?(?P<field>[\w]+)(?P<req>!)?(?::(?P<desc>.*))?$")

SEED_FIELDS = ("seed", "noise_seed")


@dataclass
class ParamSpec:
    param: str
    node_id: str
    field: str
    required: bool = False
    is_file: bool = False
    description: str = ""


def parse_title(title: str) -> ParamSpec | None:
    m = _DSL.match((title or "").strip())
    if not m or m.group("param") == "output":
        return None
    return ParamSpec(
        param=m.group("param"),
        node_id="",
        field=m.group("field"),
        required=bool(m.group("req")),
        is_file=bool(m.group("url")),
        description=(m.group("desc") or "").strip(),
    )


def _title(node: dict) -> str:
    return (node.get("_meta") or {}).get("title", "")


def annotated_params(workflow: dict) -> list[ParamSpec]:
    specs = []
    for node_id, node in workflow.items():
        spec = parse_title(_title(node))
        if spec:
            spec.node_id = node_id
            specs.append(spec)
    return specs


def output_node_ids(workflow: dict) -> set[str]:
    """标题为 $output.xxx 的节点；没有则返回空集合（表示收集全部输出）。"""
    return {nid for nid, n in workflow.items() if _title(n).strip().startswith("$output.")}


def _link_target(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _text_node(workflow: dict, node_id: str | None, depth: int = 0) -> str | None:
    """沿连线找到真正存放提示词文本的节点（跳过 conditioning 中间节点）。"""
    if node_id is None or depth > 6 or node_id not in workflow:
        return None
    inputs = workflow[node_id].get("inputs", {})
    for key in ("text", "prompt"):
        if isinstance(inputs.get(key), str):
            return node_id
    for key in ("conditioning", "conditioning_1", "positive", "clip_conditioning"):
        found = _text_node(workflow, _link_target(inputs.get(key)), depth + 1)
        if found:
            return found
    return None


def autodetect(workflow: dict) -> list[ParamSpec]:
    """没有 DSL 标注时，按常见文生图结构推断 prompt / negative_prompt / seed / width / height。"""
    specs: list[ParamSpec] = []
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        if "positive" not in inputs:
            continue
        pos = _text_node(workflow, _link_target(inputs.get("positive")))
        neg = _text_node(workflow, _link_target(inputs.get("negative")))
        if pos:
            specs.append(ParamSpec("prompt", pos, "text" if "text" in workflow[pos]["inputs"] else "prompt"))
        if neg and neg != pos:
            specs.append(ParamSpec("negative_prompt", neg, "text" if "text" in workflow[neg]["inputs"] else "prompt"))
        latent = _link_target(inputs.get("latent_image"))
        if latent and latent in workflow:
            li = workflow[latent].get("inputs", {})
            if "width" in li and "height" in li:
                specs.append(ParamSpec("width", latent, "width"))
                specs.append(ParamSpec("height", latent, "height"))
        break
    # 种子可能在采样器上，也可能在单独的 RandomNoise 节点上
    for node_id, node in workflow.items():
        for f in SEED_FIELDS:
            if isinstance(node.get("inputs", {}).get(f), int):
                specs.append(ParamSpec("seed", node_id, f))
    return specs


def resolve_specs(workflow: dict) -> list[ParamSpec]:
    specs = annotated_params(workflow)
    have = {s.param for s in specs}
    # 标注优先；没标注的常用参数再用自动推断补齐
    for s in autodetect(workflow):
        if s.param not in have:
            specs.append(s)
    return specs


def apply_params(workflow: dict, params: dict[str, Any], specs: list[ParamSpec] | None = None) -> dict:
    wf = copy.deepcopy(workflow)
    specs = specs if specs is not None else resolve_specs(wf)
    missing = [s.param for s in specs if s.required and params.get(s.param) in (None, "")]
    if missing:
        raise ValueError(f"工作流缺少必填参数: {', '.join(sorted(set(missing)))}")
    if any(s.param == "seed" for s in specs) and params.get("seed") in (None, -1):
        params["seed"] = random.randint(0, 2**31 - 1)  # 回写，方便调用方告诉用户用了哪个种子
    for s in specs:
        if s.param in params and params[s.param] is not None:
            wf[s.node_id]["inputs"][s.field] = params[s.param]
    return wf


class ComfyUIError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(self, url: str, timeout: float = 600, transport: httpx.AsyncBaseTransport | None = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self.client_id = uuid.uuid4().hex

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.url, timeout=60, transport=self._transport)

    async def ping(self) -> dict:
        async with self._client() as c:
            r = await c.get("/system_stats")
            r.raise_for_status()
            return r.json()

    async def upload(self, path: Path) -> str:
        """上传本地文件（图片 / 音频都走 /upload/image），返回 ComfyUI 里的文件名。"""
        async with self._client() as c:
            with open(path, "rb") as f:
                r = await c.post("/upload/image", files={"image": (path.name, f)}, data={"overwrite": "true"})
            r.raise_for_status()
            info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    async def run(self, workflow: dict, out_dir: Path, prefix: str, poll_interval: float = 1.0) -> list[Path]:
        wanted = output_node_ids(workflow)
        async with self._client() as c:
            r = await c.post("/prompt", json={"prompt": workflow, "client_id": self.client_id})
            if r.status_code != 200:
                raise ComfyUIError(f"ComfyUI 拒绝了工作流 ({r.status_code}): {r.text[:500]}")
            prompt_id = r.json()["prompt_id"]

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            entry = None
            while loop.time() < deadline:
                h = await c.get(f"/history/{prompt_id}")
                h.raise_for_status()
                data = h.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyUIError(f"ComfyUI 执行失败: {status.get('messages')}")
                    if status.get("completed", True):
                        break
                await asyncio.sleep(poll_interval)
            else:
                raise ComfyUIError(f"等待 ComfyUI 超时（{self.timeout}s）")

            saved: list[Path] = []
            for node_id, out in (entry.get("outputs") or {}).items():
                if wanted and node_id not in wanted:
                    continue
                for kind in ("images", "audio", "gifs", "videos"):
                    for item in out.get(kind, []) or []:
                        if item.get("type") == "temp" and kind == "images" and not wanted:
                            continue  # 跳过预览图
                        v = await c.get("/view", params={
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        })
                        v.raise_for_status()
                        suffix = Path(item["filename"]).suffix or ".bin"
                        dest = out_dir / f"{prefix}_{len(saved) + 1}{suffix}"
                        dest.write_bytes(v.content)
                        saved.append(dest)
        if not saved:
            raise ComfyUIError("工作流跑完了但没有拿到任何输出文件，请检查是否有 SaveImage / SaveAudio 节点")
        return saved
