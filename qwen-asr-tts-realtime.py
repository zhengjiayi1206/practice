import os
import dashscope
import sounddevice as sd
import numpy as np
import base64
import time

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 声卡初始化
sd.default.samplerate = 24000
sd.default.channels = 1

text = "你好啊，我是通义千问"
response = dashscope.MultiModalConversation.call(
    api_key="sk-dfb5d9295ee94adeafc438d18c7d4900",
    model="qwen3-tts-flash",
    text=text,
    voice="Cherry",
    language_type="Chinese",
    stream=True
)

for chunk in response:
    if chunk.output is not None:
        audio = chunk.output.audio
        if audio.data is not None:
            wav_bytes = base64.b64decode(audio.data)
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
            sd.play(audio_np, 24000)
            sd.wait()
        if chunk.output.finish_reason == "stop":
            print("finish at: {} ", chunk.output.audio.expires_at)

time.sleep(0.8)

