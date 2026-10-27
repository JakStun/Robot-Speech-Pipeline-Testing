
import subprocess
import numpy as np
import wave
import time

# Microphone settings
DEVICE = "hw:1,0"
SAMPLE_RATE = 16000
CHANNELS = 2
SECONDS = 10
OUTPUT_FILE = "mic_test.wav"

print("Checking microphone hardware rate...")

# Capture raw S32_LE audio directly from ALSA.
command = [
    "arecord",
    "-D", DEVICE,
    "-f", "S32_LE",
    "-r", str(SAMPLE_RATE),
    "-c", str(CHANNELS),
    "-t", "raw",
    "-q"
]

try:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"Recording for {SECONDS} seconds. Speak into the microphone.")

    # Read exactly the requested amount of raw audio.
    bytes_needed = SAMPLE_RATE * SECONDS * CHANNELS * 4
    raw_audio = bytearray()

    start = time.time()

    while len(raw_audio) < bytes_needed:
        chunk = process.stdout.read(
            min(16384, bytes_needed - len(raw_audio))
        )

        if not chunk:
            break

        raw_audio.extend(chunk)

    process.terminate()
    process.wait(timeout=3)

    if len(raw_audio) == 0:
        error = process.stderr.read().decode(errors="replace")
        raise RuntimeError(f"No audio received.\n{error}")

    # Interpret audio as signed 32-bit PCM.
    samples = np.frombuffer(raw_audio, dtype=np.int32)

    # Two channels: inspect channel 0.
    stereo = samples[:len(samples) // CHANNELS * CHANNELS].reshape(-1, CHANNELS)
    mic = stereo[:, 0].copy()

    print("\nMicrophone statistics — channel 0")
    print("Samples:", len(mic))
    print("Duration:", round(len(mic) / SAMPLE_RATE, 3), "seconds")
    print("Minimum:", int(mic.min()))
    print("Maximum:", int(mic.max()))
    print("Mean:", round(float(mic.mean()), 2))
    print("RMS:", round(float(np.sqrt(np.mean(mic.astype(np.float64) ** 2))), 2))

    # Save both channels as a 32-bit PCM WAV.
    with wave.open(OUTPUT_FILE, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(4)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(raw_audio[:len(stereo.tobytes())])

    print(f"\nSaved recording to {OUTPUT_FILE}")

except Exception as error:
    print("\nMicrophone test failed:")
    print(error)
    print("\nCheck whether hw:1,0 supports 16000 Hz natively.")

finally:
    if "process" in locals() and process.poll() is None:
        process.terminate()
        process.wait()
