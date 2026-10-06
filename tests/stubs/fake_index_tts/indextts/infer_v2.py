"""测试用的假 IndexTTS2：把收到的参数写进输出文件。"""
import json


class IndexTTS2:
    def __init__(self, **kw):
        pass

    def infer(self, spk_audio_prompt, text, output_path, **kw):
        with open(output_path, "wb") as f:
            f.write(b"RIFF" + json.dumps({"text": text, **kw}, ensure_ascii=False).encode())
