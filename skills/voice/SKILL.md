---
name: voice
description: 一句话配音。用户想把文字配成语音、朗读、念出来、做旁白/口播时，用本地 IndexTTS2（mcp_media_text_to_speech）克隆音色合成，并把音频直接发给用户。
version: 0.1.0
metadata:
  hermes:
    tags: [tts, voice, indextts2, audio]
    category: creative
---

# 一句话配音（IndexTTS2）

## When to Use
- 用户输入 `/voice <文字>`，或者说「帮我配个音」「把这段读出来」「用xx的声音念…」「做段口播」。
- 优先使用本地工具 `mcp_media_text_to_speech`，不要用 Hermes 自带的在线 TTS。

## 怎么做
1. 分清楚「要念的内容」和「要求」：
   - `/voice 用温柔女声，开心地说：今天天气真好` → text = "今天天气真好"，voice = "女"，emotion = "开心"
   - 用户只给了一段话 → 整段都是 text，不要改写用户的原文。
   - 用户让你「写一段…并配音」→ 先写好文案，再把文案作为 text。
2. 音色 `voice`：用户点名了音色（如「男声」「旁白」「小美」）就填，支持模糊匹配；不确定有哪些时先调 `mcp_media_list_voices`。没说就留空用默认音色。
3. 情绪 `emotion`：用户描述了语气（开心、悲伤、愤怒、惊讶、温柔、激动、低沉…）就用一句简短的中文描述填进去；没说就留空（跟随参考音色本身的语气）。情绪太夸张时可以把 `emo_alpha` 调低到 0.4。
4. 文本很长（超过 500 字）时按段落拆成几次调用，分别返回。
5. 把返回结果里的 `MEDIA:/路径` **原样**放进回复，再用一句话说明用的音色和情绪。

## 示例
- `/voice 欢迎收看本期节目` → text = "欢迎收看本期节目"
- `/voice 用旁白的声音，低沉地说：故事，要从三十年前说起` → text = "故事，要从三十年前说起"，voice = "旁白"，emotion = "低沉、缓慢、有悬念感"
