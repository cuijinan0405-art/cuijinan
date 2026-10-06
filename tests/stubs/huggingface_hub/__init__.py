"""测试用的假 huggingface_hub：按 FAKE_REPOS 生成文件。"""
import fnmatch
import os
from dataclasses import dataclass

FAKE_REPOS = {
    "Qwen/Qwen-Image-2.1": {"model_index.json": b'{"transformer": ["diffusers", "QwenImageTransformer2DModel"]}',
                            "transformer/diffusion_pytorch_model.safetensors": b"x" * 2048},
    "0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF": {"qwen-image-2.1-Q8_0.gguf": b"g" * 1024,
                                                   "qwen-image-2.1-Q4_K_M.gguf": b"q" * 512},
}
CALLS = []


@dataclass
class DryRunFileInfo:
    filename: str
    file_size: int


def snapshot_download(repo_id, local_dir=None, allow_patterns=None, endpoint=None, dry_run=False, **kw):
    if repo_id not in FAKE_REPOS:
        raise RuntimeError("404 Client Error: Repository Not Found")
    files = {k: v for k, v in FAKE_REPOS[repo_id].items()
             if not allow_patterns or any(fnmatch.fnmatch(k, p) for p in allow_patterns)}
    if dry_run:
        return [DryRunFileInfo(k, len(v)) for k, v in files.items()]
    CALLS.append((repo_id, endpoint, allow_patterns))
    for k, v in files.items():
        path = os.path.join(local_dir, k)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(v)
    return local_dir
