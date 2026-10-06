import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from hermes_media.backends import ImageGenerator, TTSGenerator
from hermes_media.comfyui import apply_params, autodetect, parse_title, resolve_specs
from hermes_media.config import DEFAULTS, Config, _deep_merge
from hermes_media.downloads import DownloadManager
from hermes_media.models import ModelRegistry
from hermes_media.nodes import Job, build_nodes, execute
from hermes_media.presets import Presets
from hermes_media.workers import WorkerManager

ROOT = Path(__file__).resolve().parents[1]
STUBS = Path(__file__).resolve().parent / "stubs"
QWEN_WF = json.loads((ROOT / "workflows" / "qwen_image_2_1_api.json").read_text(encoding="utf-8"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_cfg(tmp_path: Path, **override) -> Config:
    voices = tmp_path / "voices"
    voices.mkdir(exist_ok=True)
    data = _deep_merge(DEFAULTS, {"output_dir": str(tmp_path / "out"), **override})
    data["tts"]["voices_dir"] = str(voices)
    data["image"]["comfyui"]["workflow"] = str(ROOT / "workflows" / "qwen_image_2_1_api.json")
    return Config(data, tmp_path)


class FakeWorkers:
    """不起进程，直接返回假地址（配合 httpx.MockTransport）。"""

    def image_ready(self, model=None):
        self.last_model = model
        return "http://image.test"

    class _W:
        def __init__(self, url):
            self.url = url

        def ensure_ready(self):
            return self.url

    def __getitem__(self, name):
        return self._W(f"http://{name}.test")


def fake_model_transport(seen: dict) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.setdefault(req.url.host, []).append(body)
        if req.url.path == "/generate":
            return httpx.Response(200, content=b"\x89PNG" + str(body["seed"]).encode())
        if req.url.path == "/tts":
            return httpx.Response(200, content=b"RIFFout")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


# ----------------------------------------------------------------------------- ComfyUI DSL


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
        node["_meta"]["title"] = node["class_type"]
    found = {(s.param, s.node_id, s.field) for s in autodetect(wf)}
    assert {("prompt", "5", "text"), ("negative_prompt", "6", "text"), ("width", "7", "width"),
            ("height", "7", "height"), ("seed", "8", "seed")} <= found


def test_apply_params_and_required():
    specs = resolve_specs(QWEN_WF)
    params = {"prompt": "一只猫", "negative_prompt": "模糊", "width": 1664, "height": 928, "seed": None}
    wf = apply_params(QWEN_WF, params, specs)
    assert wf["5"]["inputs"]["text"] == "一只猫" and wf["6"]["inputs"]["text"] == "模糊"
    assert (wf["7"]["inputs"]["width"], wf["7"]["inputs"]["height"]) == (1664, 928)
    assert wf["8"]["inputs"]["seed"] == params["seed"] is not None
    assert QWEN_WF["5"]["inputs"]["text"] == ""
    with pytest.raises(ValueError, match="prompt"):
        apply_params(QWEN_WF, {"prompt": ""}, specs)


# ----------------------------------------------------------------------------- 配置 / 预设


def test_image_sizes(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.image_size("16:9") == (1664, 928)
    assert cfg.image_size("9：16") == (928, 1664)
    assert cfg.image_size("800x600") == (800, 600)
    assert cfg.image_size(None) == (1328, 1328)
    with pytest.raises(ValueError):
        cfg.image_size("5:1")


def test_presets_shipped_with_project(tmp_path):
    p = Presets.load(make_cfg(tmp_path))
    assert len(p.styles) >= 15 and len({s.id for s in p.styles}) == len(p.styles)
    assert p.style("国风").name == "国风水墨"
    assert p.style("cyberpunk").ratio == "16:9"
    assert "江南" in p.style("国风水墨").apply("江南古镇")
    with pytest.raises(ValueError, match="可选"):
        p.style("不存在的风格")
    assert p.emotion("开心").text and p.emotion("happy").name == "开心"
    custom = p.emotion("咬牙切齿地")
    assert custom.id == "custom" and custom.text == "咬牙切齿地"
    assert p.emotion("") is None


def test_voice_presets(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "voices" / "温柔女声.wav").write_bytes(b"RIFF1")
    (tmp_path / "voices" / "我的声音.mp3").write_bytes(b"ID3")
    cfg.data["tts"]["local"]["index_tts_dir"] = str(STUBS / "fake_index_tts")
    # 预设里的 file 是相对配置目录的 voices/xxx.wav
    p = Presets.load(cfg)
    names = [v.name for v in p.voices()]
    assert names == ["温柔女声", "我的声音", "IndexTTS 示例 01"]
    assert len(p.voices(include_missing=True)) > len(names)  # 其它预设缺音频
    assert p.voice("女").name == "温柔女声"
    assert p.voice(None).name == "温柔女声"
    assert p.voice("示例 01").source == "example"
    with pytest.raises(ValueError, match="可用音色"):
        p.voice("不存在")


# ----------------------------------------------------------------------------- 后端


def test_local_image_with_style(tmp_path):
    seen = {}
    gen = ImageGenerator(make_cfg(tmp_path), FakeWorkers(), fake_model_transport(seen))
    r = asyncio.run(gen.generate("江南古镇", style="国风水墨", seed=7, count=2))
    assert [f.read_bytes() for f in r.files] == [b"\x89PNG7", b"\x89PNG8"]
    body = seen["image.test"][0]
    assert "江南古镇" in body["prompt"] and "水墨" in body["prompt"]
    assert (body["width"], body["height"]) == (1056, 1584)  # 风格推荐 2:3
    assert body["negative_prompt"] and body["steps"] == 50
    assert r.info["style"] == "国风水墨" and r.info["seeds"] == [7, 8]


def test_local_tts_with_presets(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "voices" / "温柔女声.wav").write_bytes(b"RIFFvoice")
    seen = {}
    tts = TTSGenerator(cfg, FakeWorkers(), fake_model_transport(seen))
    r = asyncio.run(tts.speak("今天天气真好", voice="温柔"))  # 没给情绪 → 用音色预设的默认情绪「温柔」
    body = seen["tts.test"][0]
    assert r.files[0].read_bytes() == b"RIFFout"
    assert body["text"] == "今天天气真好" and "温柔" in body["emo_text"] and body["emo_alpha"] == 0.6
    assert base64.b64decode(body["voice_b64"]) == b"RIFFvoice"
    asyncio.run(tts.speak("你好", voice="温柔女声", emotion="跟随音色"))
    assert seen["tts.test"][1]["emo_text"] is None and seen["tts.test"][1]["emo_alpha"] == 1.0
    asyncio.run(tts.speak("你好", voice="温柔女声", emotion="生气", emo_alpha=0.3))
    assert "愤怒" in seen["tts.test"][2]["emo_text"] and seen["tts.test"][2]["emo_alpha"] == 0.3


def test_comfyui_backend_still_works(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr("hermes_media.comfyui.asyncio.sleep", lambda *_: real_sleep(0))
    submitted = []

    def handler(req):
        if req.url.path == "/prompt":
            submitted.append(json.loads(req.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": "p1"})
        if req.url.path.startswith("/history/"):
            return httpx.Response(200, json={"p1": {"status": {"status_str": "success", "completed": True},
                                  "outputs": {"10": {"images": [{"filename": "a.png", "type": "output"}]}}}})
        return httpx.Response(200, content=b"png")

    cfg = make_cfg(tmp_path)
    cfg.image["backend"] = "comfyui"
    r = asyncio.run(ImageGenerator(cfg, None, httpx.MockTransport(handler)).generate("猫", ratio="16:9", seed=1))
    assert r.files[0].read_bytes() == b"png" and submitted[0]["7"]["inputs"]["width"] == 1664


def test_openai_image(tmp_path):
    def handler(req):
        body = json.loads(req.content)
        assert body["size"] == "928x1664"
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"img").decode()}]})

    cfg = make_cfg(tmp_path)
    cfg.image["backend"] = "openai"
    r = asyncio.run(ImageGenerator(cfg, None, httpx.MockTransport(handler)).generate("猫", ratio="9:16"))
    assert r.files[0].read_bytes() == b"img"


# ----------------------------------------------------------------------------- 节点图


def image_and_tts_graph(style="电影大片", ratio="跟随风格"):
    return {
        "1": {"type": "TextInput", "widgets": {"text": "海边的橘猫"}},
        "2": {"type": "StylePreset", "widgets": {"style": style}, "inputs": {"subject": ["1", 0]}},
        "3": {"type": "QwenImage", "widgets": {"ratio": ratio, "steps": 8, "cfg": 3.5, "seed": 5, "count": 1},
              "inputs": {"prompt": ["2", 0], "negative": ["2", 1], "ratio_in": ["2", 2]}},
        "4": {"type": "PreviewImage", "inputs": {"images": ["3", 0]}},
        "5": {"type": "TextInput", "widgets": {"text": "欢迎收看"}},
        "6": {"type": "VoicePreset", "widgets": {"voice": "温柔女声"}},
        "7": {"type": "EmotionPreset", "widgets": {"emotion": "开心", "custom": "", "strength": 0.4}},
        "8": {"type": "IndexTTS2", "inputs": {"text": ["5", 0], "voice": ["6", 0], "emotion": ["7", 0]}},
        "9": {"type": "PreviewAudio", "inputs": {"audio": ["8", 0]}},
    }


def test_node_definitions(tmp_path):
    nodes = build_nodes(make_cfg(tmp_path))
    assert {"TextInput", "StylePreset", "QwenImage", "PreviewImage", "VoicePreset", "EmotionPreset",
            "IndexTTS2", "PreviewAudio"} == set(nodes)
    assert nodes["QwenImage"].to_json()["widgets"][0]["options"] == ["配置文件里的模型"]
    style_widget = nodes["StylePreset"].to_json()["widgets"][0]
    assert "国风水墨" in style_widget["options"]
    json.dumps([n.to_json() for n in nodes.values()], ensure_ascii=False)


def test_execute_graph(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "voices" / "温柔女声.wav").write_bytes(b"RIFFvoice")
    seen = {}
    t = fake_model_transport(seen)
    job = asyncio.run(execute(image_and_tts_graph(), cfg, ImageGenerator(cfg, FakeWorkers(), t),
                              TTSGenerator(cfg, FakeWorkers(), t), Job()))
    assert job.status == "done", job.error
    assert job.results["4"]["images"][0].startswith("/files/") and job.results["9"]["audio"][0].endswith(".wav")
    img = seen["image.test"][0]
    assert (img["width"], img["height"], img["steps"], img["cfg"], img["seed"]) == (1664, 928, 8, 3.5, 5)
    assert "电影剧照" in img["prompt"]
    tts = seen["tts.test"][0]
    assert "开心" in tts["emo_text"] and tts["emo_alpha"] == 0.4


def test_execute_graph_errors(tmp_path):
    cfg = make_cfg(tmp_path)
    g = image_and_tts_graph()
    del g["3"]["inputs"]["prompt"]
    job = asyncio.run(execute({k: g[k] for k in "1234"}, cfg, ImageGenerator(cfg, FakeWorkers()), None, Job()))
    assert job.status == "error" and "没有连线" in job.error and job.error_node == "3"
    job = asyncio.run(execute({"1": g["1"]}, cfg, None, None, Job()))
    assert job.status == "error" and "输出节点" in job.error


# ----------------------------------------------------------------------------- 真实进程：模型常驻服务 + 网页接口


@pytest.fixture
def live_cfg(tmp_path, monkeypatch):
    """用真实的 workers/*.py 子进程，但模型换成 tests/stubs 里的假模块。"""
    monkeypatch.setenv("PYTHONPATH", str(STUBS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cfg = make_cfg(tmp_path)
    cfg.data["image"]["local"].update(python=sys.executable, model="fake/qwen", device="cpu",
                                       port=free_port(), startup_timeout=60)
    cfg.data["tts"]["local"].update(python=sys.executable, index_tts_dir=str(STUBS / "fake_index_tts"),
                                     port=free_port(), startup_timeout=60)
    workers = WorkerManager(cfg)
    yield cfg, workers
    workers.stop_all()


def test_workers_autostart_and_generate(live_cfg):
    cfg, workers = live_cfg
    img = asyncio.run(ImageGenerator(cfg, workers).generate("测试", style="写实摄影", seed=3))
    assert img.files[0].read_bytes().startswith(b"\x89PNG")
    tts = asyncio.run(TTSGenerator(cfg, workers).speak("你好世界", voice="示例 01", emotion="悲伤"))
    payload = json.loads(tts.files[0].read_bytes()[4:])
    assert payload["text"] == "你好世界" and payload["use_emo_text"] and "悲伤" in payload["emo_text"]
    assert workers.status() == {"image": {"state": "ready", "error": None, "progress": None,
                                          "model": "fake/qwen (DiffusionPipeline)", "model_id": "default"},
                                "tts": {"state": "ready", "error": None, "version": "2"}}


def test_worker_exits_with_parent(live_cfg, tmp_path):
    cfg, _ = live_cfg
    port = cfg.image["local"]["port"]
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = subprocess.Popen([sys.executable, str(ROOT / "workers" / "image_worker.py"), "--model", "x",
                              "--device", "cpu", "--port", str(port), "--parent-pid", str(parent.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        parent.kill()
        parent.wait()
        child.wait(timeout=15)
        assert child.returncode == 0
    finally:
        child.kill()


def test_web_api_end_to_end(live_cfg):
    from http.server import ThreadingHTTPServer

    from hermes_media.webapp import App, make_handler

    cfg, workers = live_cfg
    (Path(cfg.tts["voices_dir"]) / "温柔女声.wav").write_bytes(b"RIFFvoice")
    app = App(cfg, workers)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with httpx.Client(base_url=base, timeout=30) as c:
            assert "litegraph" in c.get("/").text
            assert c.get("/static/vendor/litegraph.core.min.js").status_code == 200
            assert c.get("/static/../config.py").status_code == 404
            info = c.get("/api/nodes").json()
            assert "温柔女声" in [v["name"] for v in info["voices"] if v["available"]]
            r = c.post("/api/voices/upload?name=我的声音&ext=.wav", content=b"RIFFmine").json()
            assert r["name"] == "我的声音"
            assert c.get("/api/voices/audio", params={"name": "我的声音"}).content == b"RIFFmine"

            job_id = c.post("/api/run", json={"graph": image_and_tts_graph()}).json()["job_id"]
            for _ in range(200):
                job = c.get(f"/api/jobs/{job_id}").json()
                if job["status"] in ("done", "error"):
                    break
                time.sleep(0.1)
            assert job["status"] == "done", job["error"]
            assert c.get(job["results"]["4"]["images"][0]).content.startswith(b"\x89PNG")
            assert c.get(job["results"]["9"]["audio"][0]).content.startswith(b"RIFF")
            assert len(c.get("/api/history").json()) == 2
            assert c.get("/api/status").json()["workers"]["image"]["state"] == "ready"
    finally:
        server.shutdown()


# ----------------------------------------------------------------------------- 模型下载 / 切换


@pytest.fixture
def fake_hub(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("huggingface_hub", STUBS / "huggingface_hub" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return mod


def wait_download(dm, model_id):
    for _ in range(100):
        t = dm.tasks[model_id]
        if t.state != "running":
            return t
        time.sleep(0.05)
    raise AssertionError("download did not finish")


def test_download_and_resolve_models(tmp_path, fake_hub):
    cfg = make_cfg(tmp_path)
    reg = ModelRegistry(cfg)
    assert [e.id for e in reg.available()] == ["default"]
    with pytest.raises(ValueError, match="还没下载"):
        reg.get("qwen-image-2.1-uncensored-q8")

    dm = DownloadManager(cfg)
    t = wait_download(dm, dm.start("qwen-image-2.1-uncensored-q8", "hf-mirror").model_id)
    assert t.state == "done" and t.total == 1024
    with pytest.raises(ValueError, match="基础模型"):
        reg.load_spec("qwen-image-2.1-uncensored-q8")  # 量化版要先有官方版

    t = wait_download(dm, dm.start("qwen-image-2.1").model_id)
    assert t.state == "done" and t.total == t.downloaded - len((reg.models_dir / "qwen-image-2.1" / ".hm_complete").read_bytes())
    assert fake_hub.CALLS[0][1] == "https://hf-mirror.com" and fake_hub.CALLS[1][1] is None

    entry, spec = reg.load_spec("千问 Image 2.1 Uncensored · GGUF Q8（社区）")
    assert spec.model == str(reg.models_dir / "qwen-image-2.1")
    assert spec.transformer.endswith("qwen-image-2.1-Q8_0.gguf")  # 只下了匹配的文件
    assert not list((reg.models_dir / "qwen-image-2.1-uncensored-q8").glob("*Q4*"))
    assert {e.name for e in reg.available()} >= {"千问 Image 2.1 官方版", entry.name}

    # 配置里的模型目录不存在时，自动用已下载的官方版
    cfg.data["image"]["local"]["model"] = str(tmp_path / "nope")
    assert ModelRegistry(cfg).get(None).id == "qwen-image-2.1"

    t = wait_download(dm, dm.start("qwen-image-2.1-uncensored-hf").model_id)
    assert t.state == "error" and "Not Found" in t.error


def test_custom_models_in_config(tmp_path):
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"x")
    cfg = make_cfg(tmp_path, image={"models": [{"name": "我的模型", "model": "Qwen/Qwen-Image-2.1",
                                                 "lora": "style.safetensors", "lora_scale": 0.8}],
                                    "default_model": "我的模型"})
    entry, spec = ModelRegistry(cfg).load_spec(None)
    assert entry.name == "我的模型" and spec.model == "Qwen/Qwen-Image-2.1"
    assert spec.args() == ["--model", "Qwen/Qwen-Image-2.1", "--lora", str(lora), "--lora-scale", "0.8"]
    nodes = build_nodes(cfg)
    model_widget = nodes["QwenImage"].to_json()["widgets"][0]
    assert model_widget["default"] == "我的模型" and "配置文件里的模型" in model_widget["options"]


def test_switch_image_model_live(live_cfg, fake_hub):
    cfg, workers = live_cfg
    dm = DownloadManager(cfg)
    for mid in ("qwen-image-2.1", "qwen-image-2.1-uncensored-q8"):
        assert wait_download(dm, dm.start(mid).model_id).state == "done"
    gen = ImageGenerator(cfg, workers)
    asyncio.run(gen.generate("猫", seed=1))
    assert workers["image"].health()["model_id"] == "default"
    r = asyncio.run(gen.generate("猫", seed=1, model="qwen-image-2.1-uncensored-q8"))
    assert r.files[0].read_bytes().startswith(b"\x89PNG")
    assert workers["image"].health()["model_id"] == "qwen-image-2.1-uncensored-q8"
    log = workers["image"].log_path.read_text(encoding="utf-8")
    assert "替换扩散模型" in log and "gguf=True" in log and "parts=['transformer']" in log


def test_mcp_server_tools(tmp_path):
    from hermes_media.server import build_server

    mcp = build_server(make_cfg(tmp_path))
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in (tools.tools if hasattr(tools, "tools") else tools)}
    assert {"generate_image", "text_to_speech", "list_voices", "list_styles"} <= names
