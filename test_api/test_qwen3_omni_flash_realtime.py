import os
import base64
import numpy as np
import sounddevice as sd
from openai import OpenAI

content = "你会说南京话吗，说几句"

# 1. 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 2. 初始化播放器 (Qwen3-Omni 默认音频通常是 24000Hz)
# 注意：如果返回的是 wav 格式，前几个 chunk 会包含 wav 头，raw 播放可能会有一声“咔哒”
sample_rate = 24000 
player_stream = sd.RawOutputStream(
    samplerate=sample_rate,
    blocksize=2400, # 每次处理 100ms 的数据
    channels=1,
    dtype='int16'
)
player_stream.start()

print("正在请求并准备播放...")

completion = client.chat.completions.create(
    model="qwen3-omni-flash",
    messages=[{"role": "user", "content": content}],
    modalities=["text", "audio"],
    # 注意：为了流式播放方便，建议使用 pcm16 格式（如果 API 支持），
    # 如果用 wav，代码需要处理掉文件头
    audio={"voice": "Li", "format": "wav"}, 
    stream=True,
    stream_options={"include_usage": True},
)

try:
    for chunk in completion:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            
            # 处理文本部分
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='', flush=True)
            
            # 处理音频部分
            if hasattr(delta, 'audio') and delta.audio:
                # 获取 base64 编码的音频 delta
                if 'data' in delta.audio:
                    b64_data = delta.audio['data']
                    raw_audio = base64.b64decode(b64_data)
                    
                    # 如果是 WAV 格式，第一个 chunk 包含 44 字节的头，
                    # 简单处理可以跳过它，或者直接写入（会有轻微杂音）
                    player_stream.write(raw_audio)
        
        elif hasattr(chunk, 'usage') and chunk.usage:
            print(f"\n\n[Token Usage]: {chunk.usage}")

finally:
    # 停止播放
    player_stream.stop()
    player_stream.close()
    print("\n播放结束。")