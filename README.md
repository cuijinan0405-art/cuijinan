# hermes-media：Hermes Agent 一句话出图 / 配音

在 Hermes Agent 里直接说一句话，就能调用你**本地**的：

- **千问 Qwen-Image 2.1** 出图（通过 ComfyUI，或任何 OpenAI 兼容的出图接口）
- **IndexTTS2 / 2.5** 配音（克隆参考音色，可以用一句话控制情绪）

```
/img 一只戴墨镜的橘猫在海边冲浪，赛博朋克风，横版
/voice 用旁白的声音，低沉地说：故事，要从三十年前说起
```

也可以不加斜杠，直接说「帮我画张…」「把这段读出来」，Hermes 会自己调用工具，然后把图片 / 音频直接发给你（CLI、Telegram、微信、飞书等网关都支持）。

---

## 一、GitHub 调研结论（2026-10）

| 项目 | ⭐ | 做了什么 | 是否适合直接用 |
|---|---|---|---|
| [ATH-MaaS/Pixelle-MCP](https://github.com/ATH-MaaS/Pixelle-MCP) | ~1.1k | 把 ComfyUI 工作流自动变成 MCP 工具，节点标题写 `$prompt.text!` 就是参数，MIT 协议 | 思路最好，但体量大（自带 Web UI、LLM、多用户），**不针对 Hermes，不管 IndexTTS2** |
| [artokun/comfyui-mcp](https://github.com/artokun/comfyui-mcp) | ~790 | 178 个工具的 ComfyUI 全功能控制台 | 太重，「几句话出图」用不上这么多工具，反而让模型选错工具 |
| [jau123/MeiGen-AI-Design-MCP](https://github.com/jau123/MeiGen-AI-Design-MCP) | ~1.8k | 主打云端 GPT Image / Midjourney 等，附带 ComfyUI 模式 | 以云端为主，跟本地千问 + 配音需求不对口 |
| [ph1lb4/imagegen-mac](https://github.com/ph1lb4/imagegen-mac) 等 | <10 | 本地 Qwen-Image 2.1 + MCP | 只支持 Mac / 星太少 |
| [1guguo/indextts2-api](https://github.com/1guguo/indextts2-api) 等 | ~30 | IndexTTS2 加 HTTP API | 只有配音，不能直接接 Agent |

**结论：没有现成的「Hermes + 本地千问出图 + IndexTTS2 配音」一体化项目，高星项目都太重或者偏云端。**
所以本项目做法是：**抄 Pixelle-MCP 最好用的部分**——ComfyUI 节点标题 DSL（`$参数.字段!:描述`，语法完全兼容），
再补上 IndexTTS2 配音、Hermes 的斜杠命令 skill 和 `MEDIA:` 文件回传，整体只有 3 个工具，模型不会选错。

## 二、整体结构

```
你在 Hermes 里说话
   │  /img、/voice 两个 skill 负责把一句话扩写成好的参数
   ▼
Hermes Agent ──MCP(stdio)──▶ hermes-media（本项目）
                                ├─ generate_image  ──▶ ComfyUI + 千问 Image 2.1 工作流
                                ├─ text_to_speech  ──▶ tts_server/indextts2_server.py（IndexTTS2 常驻）
                                └─ list_voices     ──▶ voices/ 目录里的参考音频
   ◀── 返回 MEDIA:/xxx.png / MEDIA:/xxx.wav，Hermes 自动把文件发给你
```

```
hermes_media/          MCP 服务 + 命令行
  comfyui.py           ComfyUI 工作流运行器（DSL 参数注入 + 自动识别提示词/种子/尺寸）
  backends.py          出图 / 配音后端（comfyui、openai 兼容、indextts2）
  server.py            MCP 工具定义
  cli.py               hermes-media 命令
tts_server/indextts2_server.py   IndexTTS2 常驻服务（纯标准库，放进 index-tts 环境运行）
workflows/qwen_image_2_1_api.json  千问 Image 2.1 示例工作流（API 格式）
skills/img、skills/voice           Hermes skill → /img 和 /voice 斜杠命令
```

## 三、安装

### 1. 安装本项目

```bash
git clone https://github.com/cuijinan0405-art/cuijinan.git hermes-media
cd hermes-media
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
cp config.example.yaml config.yaml
mkdir -p ~/.hermes-media && cp config.yaml ~/.hermes-media/config.yaml   # 或者用 HERMES_MEDIA_CONFIG 指定
```

### 2. 准备千问 Image 2.1 工作流

1. 打开 ComfyUI，载入你**已经能正常出图**的千问 Image 2.1 工作流。
2. 菜单 → 工作流 → **导出 (API)**，保存为 `workflows/qwen_image_2_1_api.json`（覆盖示例文件）。
3. 不用改任何节点也能用：程序会自动找到采样器连着的正/反提示词、种子和画布尺寸。
   想精确指定，就把节点标题改成 Pixelle 风格，例如：
   - 正向提示词节点：`$prompt.text!:正向提示词`
   - 反向提示词节点：`$negative_prompt.text`
   - 输出节点：`$output.image`

> 如果你的千问 Image 不是 ComfyUI，而是一个 OpenAI 兼容服务（`/v1/images/generations`），把 `image.backend` 改成 `openai` 并填 `base_url` 即可。

### 3. 启动 IndexTTS2 常驻服务

模型只加载一次，之后每次配音只要几秒：

```bash
cd /path/to/index-tts               # 你本地的 IndexTTS 仓库
uv run python /path/to/hermes-media/tts_server/indextts2_server.py --model-dir checkpoints --port 7861
# IndexTTS2.5：加 --version 2.5
```

然后把 3~10 秒干净人声的参考音频放进 `voices/`，**文件名就是音色名**，比如 `voices/旁白男声.wav`、`voices/温柔女声.wav`。

> 已经在 ComfyUI 里用 IndexTTS2 节点？把 `tts.backend` 改成 `comfyui`，导出 API 工作流，文本节点标题写 `$text.text!`，参考音频节点写 `$voice.~audio`（字段名按你的节点实际名称）。

### 4. 自检

```bash
hermes-media doctor                         # 检查 ComfyUI / IndexTTS2 / 工作流 / 音色
hermes-media image "一只橘猫，水彩风" -r 16:9   # 不经过 Hermes，直接出图
hermes-media tts "你好呀" -v 女声 -e 开心       # 直接配音
```

### 5. 接入 Hermes Agent

在 `~/.hermes/config.yaml` 里加上（服务名一定要叫 `media`，skill 里写的工具名是 `mcp_media_*`）：

```yaml
mcp_servers:
  media:
    command: "/path/to/hermes-media/.venv/bin/hermes-media"   # Windows: ...\.venv\Scripts\hermes-media.exe
    args: ["serve"]
    env:
      HERMES_MEDIA_CONFIG: "/home/你/.hermes-media/config.yaml"
    timeout: 900          # 出图/配音可能比较慢，给足时间
```

或者用命令：`hermes mcp add media --command /path/to/.venv/bin/hermes-media --args serve`

再把两个 skill 装进去，就有了 `/img`、`/voice` 斜杠命令：

```bash
mkdir -p ~/.hermes/skills/creative
cp -r skills/img skills/voice ~/.hermes/skills/creative/
```

重启 Hermes（或 `/reload-mcp`），试试：

```
/img 小红书封面，标题"周末去哪玩"，清新插画风
/voice 欢迎收看本期节目
帮我写一段 30 字的咖啡店开业口播，然后用温柔女声配音
```

## 四、工具说明

| 工具 | 参数 | 说明 |
|---|---|---|
| `generate_image` | `prompt`、`ratio`(1:1/16:9/9:16/4:3/3:4/3:2/2:3/宽x高)、`negative_prompt`、`seed`、`count` | 按千问官方推荐分辨率出图，返回种子方便复现 |
| `text_to_speech` | `text`、`voice`(模糊匹配)、`emotion`(一句话描述情绪)、`emo_alpha` | 有 `emotion` 时走 IndexTTS2 的文本情绪控制 |
| `list_voices` | — | 列出 `voices/` 里的音色 |

## 五、开发

```bash
pip install -e ".[dev]"
pytest
```

测试用假的 ComfyUI / IndexTTS2 HTTP 服务覆盖了：DSL 解析、无标注工作流自动识别、参数注入、ComfyUI 轮询下载、OpenAI 兼容出图、IndexTTS2 配音、MCP 工具注册。

## 致谢

- ComfyUI 节点标题 DSL 来自 [Pixelle-MCP](https://github.com/ATH-MaaS/Pixelle-MCP)（MIT）
- [IndexTTS](https://github.com/index-tts/index-tts)、[Qwen-Image](https://github.com/QwenLM/Qwen-Image)、[Hermes Agent](https://github.com/NousResearch/hermes-agent)
