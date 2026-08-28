import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

device = 'mic_sv'
samplerate = 48000
channels = 2
dtype = 'int32'
blocksize = 1024

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.zeros(blocksize))
ax.set_ylim(-1.0, 1.0)
ax.set_xlim(0, blocksize)
ax.set_title("Real-time Microphone Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude (normalized)")

def callback(indata, frames, time, status):
    mono = indata[:, 0]                     # raw int32
    mono_float = mono.astype(np.float32) / (2**31)  # normalize to -1..1

    line.set_ydata(mono_float)
    fig.canvas.draw()
    fig.canvas.flush_events()

with sd.InputStream(device=device,
                    channels=channels,
                    samplerate=samplerate,
                    dtype=dtype,
                    blocksize=blocksize,
                    callback=callback):
    print("Listening...")
    while True:
        plt.pause(0.01)