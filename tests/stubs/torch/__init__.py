"""测试用的假 torch，只够 image_worker 跑通。"""
bfloat16 = "bf16"
float16 = "f16"
float32 = "f32"


class Generator:
    def __init__(self, device="cpu"):
        self.seed = 0

    def manual_seed(self, s):
        self.seed = s
        return self
