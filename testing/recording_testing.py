import asyncio
import numpy as np

from liveaudio import Microphone


async def main():
    mic = Microphone(device="dmic_sv")

    print("Silence")

    try:
        for _ in range(80):
            frame = await mic.read_frame()

            x = frame.samples.astype(np.float64)

            low8 = x & 0xFF

            print(
                "low8:",
                "min =", low8.min(),
                "max =", low8.max(),
                "unique =", np.unique(low8)[:20],
            )
            # centered = x - x.mean()
            # print(
            #     f"min: {x.min():12.0f}, max: {x.max():12.0f}, mean: {x.mean():12.0f}, rms={np.sqrt(np.mean(x*x)):12.0f}, centered rms={np.sqrt(np.mean(centered*centered)):10.0f}"
            # )

    finally:
        mic.close()

asyncio.run(main())