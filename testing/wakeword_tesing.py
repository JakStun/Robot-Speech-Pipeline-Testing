from livekit.wakeword import WakeWordModel
from liveaudio import Microphone

import asyncio
import numpy as np

WAKEWORD_WINDOW = 32000
WAKEWORD_STRIDE = 16000

async def main():
    mic = Microphone()

    model = WakeWordModel(models=["hey_livekit.onnx"])

    audio_buffer = np.zeros(0, dtype=np.int16)

    print("Listening")

    while True:
        audio_frame = await mic.read_frame()

        audio_buffer = np.concatenate((audio_buffer, audio_frame.samples_16k))

        if len(audio_buffer) < WAKEWORD_WINDOW:
            continue

        chunk = audio_buffer[:WAKEWORD_WINDOW]

        audio_buffer = audio_buffer[WAKEWORD_STRIDE:]

        scores = model.predict(chunk)
        score = scores["hey_livekit"]

        print(f"Score: {score:.4f}", end="\r")

        if score > 0.35:
            print("\nWake word detected!")
            

asyncio.run(main())
