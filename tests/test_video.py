import asyncio
import json
import re
import struct
import subprocess
import wave
import zlib
from pathlib import Path

import httpx
import pytest

from hermes_media.backends import ImageGenerator, TTSGenerator
from hermes_media.cli import main as cli_main
from hermes_media.storyboard import Storyboard, VideoComposer
from hermes_media.video import (Line, RenderOptions, Shot, build_command, build_timeline, find_ffmpeg,
                                split_sentences, wrap_text, write_ass, write_srt)

from test_hermes_media import FakeWorkers, make_cfg


def png_bytes(w=36, h=64, rgb=(200, 60, 40)) -> bytes:
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def wav_bytes(seconds: float, rate=22050) -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def fake_media_transport(seen: dict) -> httpx.MockTransport:
    """假模型：出图返回真 PNG，配音返回真 wav（每个字 0.1 秒）。"""
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.setdefault(req.url.path, []).append(body)
        if req.url.path == "/generate":
            return httpx.Response(200, content=png_bytes())
        if req.url.path == "/tts":
            return httpx.Response(200, content=wav_bytes(0.1 * len(body["text"])))
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def ffmpeg_or_skip() -> str:
    try:
        return find_ffmpeg()
    except FileNotFoundError:
        pytest.skip("没有 ffmpeg")


def probe(ffmpeg: str, path: Path) -> str:
    return subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr


# ----------------------------------------------------------------------------- 纯计算


def test_split_and_wrap():
    assert split_sentences("清晨的乌镇。青石板路！要去吗？\n下次见") == ["清晨的乌镇。", "青石板路！", "要去吗？", "下次见"]
    assert split_sentences("  。。 ") == []
    assert wrap_text("青石板路，白墙黛瓦，小桥流水人家。", 10) == ["青石板路，白墙黛瓦", "小桥流水人家"]
    assert all(len(x) <= 5 for x in wrap_text("一二三四五六七八九十十一", 5))


def test_timeline_math(tmp_path):
    img = tmp_path / "a.png"
    shots = [
        Shot(img, [Line("第一句。", tmp_path / "1.wav", 1.0), Line("第二句。", tmp_path / "2.wav", 2.0)]),
        Shot(img, [Line("第三句。", tmp_path / "3.wav", 1.5)], hold=1.0),
        Shot(img, []),
    ]
    tl = build_timeline(shots, lead=0.3, tail=0.5, gap=0.15, empty_hold=2.0)
    assert tl.durations == pytest.approx([0.3 + 1 + 0.15 + 2 + 0.5, 0.3 + 1.5 + 0.5 + 1, 2.0])
    assert tl.starts == pytest.approx([0, 3.95, 7.25])
    assert tl.total == pytest.approx(9.25)
    assert [round(s, 2) for _, s in tl.audio] == [0.3, 1.45, 4.25]
    assert [(round(c.start, 2), round(c.end, 2), c.lines) for c in tl.cues] == [
        (0.3, 1.3, ["第一句"]), (1.45, 3.45, ["第二句"]), (4.25, 5.75, ["第三句"])]


def test_long_sentence_splits_into_cues(tmp_path):
    text = "一二三四五六七八九十，一二三四五六七八九十，一二三四五六七八九十。"
    tl = build_timeline([Shot(tmp_path / "a.png", [Line(text, None, 3.0)])], lead=0, max_chars=10)
    assert [c.lines for c in tl.cues] == [["一二三四五六七八九十", "一二三四五六七八九十"], ["一二三四五六七八九十"]]
    assert tl.cues[0].end == pytest.approx(2.0) and tl.cues[1].end == pytest.approx(3.0)


def test_subtitle_files(tmp_path):
    tl = build_timeline([Shot(tmp_path / "a.png", [Line("你好{世界}。", None, 1.25)])], lead=61)
    ass = write_ass(tl.cues, tmp_path / "s.ass", 1080, 1920, "Microsoft YaHei").read_text(encoding="utf-8")
    assert "PlayResY: 1920" in ass and "Microsoft YaHei" in ass
    assert "Dialogue: 0,0:01:01.00,0:01:02.25,Default,,0,0,0,,你好｛世界｝" in ass
    srt = write_srt(tl.cues, tmp_path / "s.srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:01:01,000 --> 00:01:02,250\n你好{世界}")


def test_build_command(tmp_path):
    shots = [Shot(tmp_path / f"{i}.png", [Line("句子。", tmp_path / f"{i}.wav", 1.0)], motion=m, transition=t)
             for i, (m, t) in enumerate([("zoom_in", "fade"), ("pan_left", "slideleft"), ("zoom_out", "random")])]
    tl = build_timeline(shots)
    args = build_command("ffmpeg", tl, tmp_path / "out.mp4",
                         RenderOptions(subtitles="subtitles.ass", fonts_dir=Path("D:/fonts")), seed="t")
    fc = args[args.index("-filter_complex") + 1]
    assert "-loop" not in args  # -loop 1 会让 zoompan 每帧都生成一整段
    assert fc.count("zoompan=") == 3 and fc.count("xfade=") == 2
    assert f"xfade=transition=slideleft:duration=0.5:offset={tl.starts[1]:.3f}" in fc
    # 前两个镜头多渲染 0.5 秒垫在转场下面，最后一个不用
    frames = [int(x) for x in re.findall(r":d=(\d+):", fc)]
    assert frames == [round((tl.durations[0] + 0.5) * 30), round((tl.durations[1] + 0.5) * 30),
                      round(tl.durations[2] * 30)]
    assert r"fontsdir='D\:/fonts'" in fc
    assert "adelay=300:all=1" in fc and "amix=inputs=3" in fc and "loudnorm=I=-14.0" in fc
    assert args[-1] == str(tmp_path / "out.mp4")


def test_build_command_without_transitions(tmp_path):
    shots = [Shot(tmp_path / f"{i}.png", [], transition="none") for i in range(2)]
    args = build_command("ffmpeg", build_timeline(shots), tmp_path / "o.mp4", RenderOptions())
    fc = args[args.index("-filter_complex") + 1]
    assert "concat=n=2" in fc and "xfade" not in fc and "anullsrc" in " ".join(args)


def test_storyboard_parsing(tmp_path):
    (tmp_path / "x.jpg").write_bytes(png_bytes())
    sb = Storyboard.from_dict({
        "title": "测试", "voice": "女声", "motion": "pan_right",
        "scenes": ["只有旁白。", {"text": "第二镜。", "image_file": "x.jpg", "motion": "none", "voice": "男声"}],
    }, tmp_path)
    assert sb.ratio == "9:16" and len(sb.scenes) == 2
    assert (sb.scenes[0].voice, sb.scenes[0].motion) == ("女声", "pan_right")
    assert (sb.scenes[1].voice, sb.scenes[1].motion, sb.scenes[1].image_file) == ("男声", "none", tmp_path / "x.jpg")
    with pytest.raises(ValueError, match="没有镜头"):
        Storyboard.from_dict({"scenes": []})
    with pytest.raises(ValueError, match="不认识的字段: bgmm"):
        Storyboard.from_dict({"bgmm": 1, "scenes": ["a"]})
    with pytest.raises(ValueError, match="运镜"):
        Storyboard.from_dict({"scenes": [{"text": "a", "motion": "spin"}]})
    with pytest.raises(ValueError, match="图片不存在"):
        Storyboard.from_dict({"scenes": [{"image_file": "nope.png"}]}, tmp_path)


# ----------------------------------------------------------------------------- 端到端（真 ffmpeg，假模型）


def video_cfg(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "voices" / "温柔女声.wav").write_bytes(wav_bytes(1))
    cfg.data["video"]["preset"] = "ultrafast"
    cfg.data["video"]["fps"] = 15
    return cfg


def composer(cfg, seen, progress=None):
    t = fake_media_transport(seen)
    return VideoComposer(cfg, ImageGenerator(cfg, FakeWorkers(), t), TTSGenerator(cfg, FakeWorkers(), t),
                         progress)


def test_compose_video_end_to_end(tmp_path):
    ffmpeg = ffmpeg_or_skip()
    cfg = video_cfg(tmp_path)
    sb = Storyboard.from_dict({
        "title": "江南", "ratio": "1:1", "style": "国风水墨", "transition": "random",
        "scenes": [
            {"text": "清晨的乌镇。还笼罩在薄雾里。", "image": "晨雾中的水乡"},
            {"text": "青石板路，白墙黛瓦。", "motion": "pan_left", "emotion": "开心"},
            {"text": "下次见。", "hold": 0.5, "motion": "zoom_out"},
        ],
    }, tmp_path)
    seen, msgs = {}, []
    r = asyncio.run(composer(cfg, seen, msgs.append).compose(sb))
    # 4 句配音、3 张图；画面提示词套用了风格，第二镜没写 image 就用旁白
    assert [b["text"] for b in seen["/tts"]] == ["清晨的乌镇。", "还笼罩在薄雾里。", "青石板路，白墙黛瓦。", "下次见。"]
    assert len(seen["/generate"]) == 3 and "青石板路" in seen["/generate"][1]["prompt"]
    assert (seen["/generate"][0]["width"], seen["/generate"][0]["height"]) == (1328, 1328)
    expected = (0.3 + 0.6 + 0.15 + 0.8 + 0.5) + (0.3 + 1.0 + 0.5) + (0.3 + 0.4 + 0.5 + 0.5)
    assert r.duration == pytest.approx(expected)
    assert r.video.is_file() and r.video.suffix == ".mp4"
    info = probe(ffmpeg, r.video)
    assert "1080x1080" in info and "h264" in info and "aac" in info and "48000 Hz" in info
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    assert float(s) == pytest.approx(expected, abs=0.1)
    assert r.srt.read_text(encoding="utf-8").count("-->") == 4
    assert any(x.startswith("合成视频") for x in msgs)
    assert r.info["tts"] == 4 and r.info["image"] == 3

    # 改一句旁白再跑：只重新配那一句；第一镜单独写了画面提示词，图也不用重出
    sb.scenes[0].text = "清晨的乌镇。还笼罩在一层薄雾里。"
    seen2 = {}
    r2 = asyncio.run(composer(cfg, seen2).compose(sb))
    assert [b["text"] for b in seen2["/tts"]] == ["还笼罩在一层薄雾里。"] and "/generate" not in seen2
    assert (r2.info["tts_cached"], r2.info["image_cached"]) == (3, 3)


def test_cli_video(tmp_path, monkeypatch, capsys):
    ffmpeg_or_skip()
    cfg = video_cfg(tmp_path)
    (tmp_path / "pic.png").write_bytes(png_bytes(rgb=(20, 80, 200)))
    script = tmp_path / "片子.yaml"
    script.write_text("title: 命令行\nratio: '16:9'\nsubtitles: false\nscenes:\n  - image_file: pic.png\n    hold: 1\n",
                      encoding="utf-8")
    monkeypatch.setattr("hermes_media.cli.Config.load", classmethod(lambda cls, path=None: cfg))
    out = tmp_path / "成片.mp4"
    assert cli_main(["video", str(script), "-o", str(out)]) == 0
    assert out.is_file() and capsys.readouterr().out.strip() == str(out)


def test_missing_voice_fails_before_any_work(tmp_path):
    ffmpeg_or_skip()
    cfg = video_cfg(tmp_path)
    sb = Storyboard.from_dict({"scenes": [{"text": "好。", "voice": "不存在的音色"}]})
    seen = {}
    with pytest.raises(ValueError, match="第 1 个镜头：找不到音色"):
        asyncio.run(composer(cfg, seen).compose(sb))
    assert not seen
