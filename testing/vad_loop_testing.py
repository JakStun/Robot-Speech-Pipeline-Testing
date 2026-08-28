import asyncio
import numpy as np
from liveaudio import Microphone, SpeechDetector

import sounddevice as sd
import soundfile as sf

mic = Microphone()
vad = SpeechDetector()

async def main():
    print("started loop")
    while True:
        frame = await mic.read_frame()
        vad.process(frame)

        if vad.started():
            print("started talking")

        if vad.ended():
            print("finished talking")

asyncio.run(main())