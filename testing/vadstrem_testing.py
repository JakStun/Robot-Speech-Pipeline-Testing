from silero_vad import load_silero_vad, VADIterator
from liveaudio import Microphone

import asyncio
import numpy as np
import torch

async def main():
    mic = Microphone()
    model = load_silero_vad()

    buffer = np.zeros(0, dtype=np.int16)

    speaking = False
    silence_samples = 0

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512

    START_THRESHOLD = 0.9
    END_THRESHOLD = 0.3

    END_SILENCE_SAMPLES = int(SAMPLE_RATE * 0.5)

    print("Listening for speech")

    try:
        while True:
            frame = await mic.read_frame()

            buffer = np.concatenate(
                (buffer, frame.samples_16k)
            )

            while len(buffer) >= CHUNK_SIZE:
                chunk = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]

                audio = torch.from_numpy(chunk).float() / 32768.0
                audio = audio - audio.mean()

                confidence = model(audio, SAMPLE_RATE).item()

                if not speaking:
                    if confidence >= START_THRESHOLD:
                        speaking = True
                        silence_samples = 0
                        
                        print("Speech started ", confidence)
                else:
                    if confidence < END_THRESHOLD:
                        silence_samples += CHUNK_SIZE
                    else:
                        silence_samples = 0

                    if silence_samples >= END_SILENCE_SAMPLES:
                        speaking = False
                        silence_samples = 0

                        print("Speech ended ", confidence)
    finally:
        mic.close()

asyncio.run(main())