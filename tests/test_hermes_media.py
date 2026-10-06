import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from hermes_media.backends import ImageGenerator, TTSGenerator
from hermes_media.comfyui import apply_params, autodetect, parse_title, resolve_specs
from hermes_media.config import Config, DEFAULTS, _deep_merge

ROOT = Path(__file__).resolve().parents[1]
QWEN_WF = json.loads((ROOT / "workflows" / "qwen_image_2_1_api.json").read_text(encoding="utf-8"))


def make_cfg(tmp_path: Path, **override) -> Config:
    data = _deep_merge(DEFAULTS, {"output_dir": str(tmp_path / "out"), **override})
    data["image"]["comfyui"]["workflow"] = str(ROOT / "workflows" / "qwen_image_2_1_api.json")
    return Config(data, tmp_path)


def test_parse_title():
    s = parse_title("$prompt.text!:正向提示词")
    assert (s.param, s.field, s.required, s.is_file, s.description) == ("prompt", "text", True, False, "正向提示词")
    s = parse_title("$voice.~audio")
    assert s.is_file and not s.required and s.field == "audio"
    assert parse_title("$output.image") is None
    assert parse_title("KSampler") is None


def test_autodetect_unannotated_workflow():
    wf = json.loads(json.dumps(QWEN_WF))
    for node in wf.values():
        node["_meta"]["title"] = node["class_type"]  # 去掉所有标注
    found = {(s.param, s.node_id, s.field) for s in autodetect(wf)}
    assert ("prompt", "5", "text") in found
    assert ("negative_prompt", "6", "text") in found
    assert ("width", "7", "width") in found and ("height", "7", "height") in found
    assert ("seed", "8", "seed") in found


def test_apply_params_and_required():
    specs = resolve_specs(QWEN_WF)
    params = {"prompt": "一只猫", "negative_prompt": "模糊", "width": 1664, "height": 928, "seed": None}
    wf = apply_params(QWEN_WF, params, specs)
    assert wf["5"]["inputs"]["text"] == "一只猫"
    assert wf["6"]["inputs"]["text"] == "模糊"
    assert (wf["7"]["inputs"]["width"], wf["7"]["inputs"]["height"]) == (1664, 928)
    assert wf["8"]["inputs"]["seed"] == params["seed"] and params["seed"] is not None
    assert QWEN_WF["5"]["inputs"]["text"] == ""  # 原工作流不被修改
    with pytest.raises(ValueError, match="prompt"):
        apply_params(QWEN_WF, {"prompt": ""}, specs)


def test_image_sizes(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.image_size("16:9") == (1664, 928)
    assert cfg.image_size("9：16") == (928, 1664)
    assert cfg.image_size("800x600") == (800, 600)
    assert cfg.image_size(None) == (1328, 1328)
    with pytest.raises(ValueError):
        cfg.image_size("5:1")


def fake_comfyui():
    state = {"submitted": [], "polls": 0}
    png = b"\x89PNG fake"

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/prompt":
            state["submitted"].append(json.loads(req.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": f"p{len(state['submitted'])}"})
        if req.url.path.startswith("/history/"):
            pid = req.url.path.rsplit("/", 1)[1]
            state["polls"] += 1
            if state["polls"] % 2 == 1:
                return httpx.Response(200, json={})  # 第一次轮询还没跑完
            return httpx.Response(200, json={pid: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"10": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
            }})
        if req.url.path == "/view":
            return httpx.Response(200, content=png)
        return httpx.Response(404)

    return state, httpx.MockTransport(handler)


def test_comfyui_image_end_to_end(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr("hermes_media.comfyui.asyncio.sleep", lambda *_: real_sleep(0))
    state, transport = fake_comfyui()
    gen = ImageGenerator(make_cfg(tmp_path), transport)
    r = asyncio.run(gen.generate("海边的橘猫", ratio="16:9", seed=42, count=2))
    assert len(r.files) == 2 and all(f.read_bytes() == b"\x89PNG fake" for f in r.files)
    assert r.info["seeds"] == [42, 43]
    first = state["submitted"][0]
    assert first["5"]["inputs"]["text"] == "海边的橘猫"
    assert first["7"]["inputs"]["width"] == 1664 and first["8"]["inputs"]["seed"] == 42


def test_openai_image(tmp_path):
    def handler(req):
        body = json.loads(req.content)
        assert body["size"] == "928x1664" and body["n"] == 1
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"img").decode()}]})

    cfg = make_cfg(tmp_path)
    cfg.image["backend"] = "openai"
    r = asyncio.run(ImageGenerator(cfg, httpx.MockTransport(handler)).generate("猫", ratio="9:16"))
    assert r.files[0].read_bytes() == b"img"


def test_indextts2_tts(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "旁白男声.wav").write_bytes(b"RIFFvoice")
    (voices / "温柔女声.wav").write_bytes(b"RIFFvoice2")
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=b"RIFFout")

    cfg = make_cfg(tmp_path)
    tts = TTSGenerator(cfg, httpx.MockTransport(handler))
    assert [v.stem for v in tts.list_voices()] == ["旁白男声", "温柔女声"]
    r = asyncio.run(tts.speak("今天天气真好", voice="女", emotion="开心"))
    assert r.files[0].read_bytes() == b"RIFFout"
    assert seen["text"] == "今天天气真好" and seen["emo_text"] == "开心" and seen["emo_alpha"] == 0.6
    assert base64.b64decode(seen["voice_b64"]) == b"RIFFvoice2"
    assert r.info["voice"] == "温柔女声"
    with pytest.raises(ValueError, match="可用音色"):
        tts.resolve_voice("不存在")


def test_mcp_server_tools(tmp_path):
    from hermes_media.server import build_server

    mcp = build_server(make_cfg(tmp_path))
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in (tools.tools if hasattr(tools, "tools") else tools)}
    assert {"generate_image", "text_to_speech", "list_voices"} <= names
