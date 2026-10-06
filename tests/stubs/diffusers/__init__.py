"""测试用的假 diffusers：返回一张按种子着色的小 PNG。"""
import struct
import zlib


class _Img:
    def __init__(self, w, h, seed):
        self.w, self.h, self.seed = max(1, w // 16), max(1, h // 16), seed

    def save(self, buf, format="PNG"):
        c = bytes(((self.seed * 37) % 256, (self.seed * 91) % 256, 160))
        rows = b"".join(b"\x00" + c * self.w for _ in range(self.h))

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

        buf.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class _Out:
    def __init__(self, img):
        self.images = [img]


class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, model, torch_dtype=None, **parts):
        print(f">> stub pipeline {model} parts={sorted(parts)}", flush=True)
        return cls()

    def load_lora_weights(self, path, adapter_name=None):
        print(f">> stub lora {path}", flush=True)

    def set_adapters(self, names, adapter_weights=None):
        pass

    def to(self, device):
        return self

    def __call__(self, prompt, negative_prompt=None, width=1024, height=1024, num_inference_steps=4,
                 true_cfg_scale=4.0, generator=None, callback_on_step_end=None):
        for i in range(num_inference_steps):
            if callback_on_step_end:
                callback_on_step_end(self, i, 0, {})
        return _Out(_Img(width, height, generator.seed))


LOADED = {}  # 测试检查用：记录加载了哪些替换组件


class GGUFQuantizationConfig:
    def __init__(self, compute_dtype=None):
        self.compute_dtype = compute_dtype


class QwenImageTransformer2DModel:
    @classmethod
    def from_single_file(cls, path, **kw):
        print(f">> stub transformer from_single_file {path} gguf={'quantization_config' in kw}", flush=True)
        return cls()

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls()
