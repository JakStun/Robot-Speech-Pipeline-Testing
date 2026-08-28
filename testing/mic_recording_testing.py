import sounddevice as sd
import soundfile as sf


print(sd.query_devices())
duration = 5
samplerate = 48000
channels = 2
dtype='int32'
device = "dmic_sv"

stream = sd.InputStream(
    device=device,
    samplerate=samplerate,
    channels=channels,
    dtype=dtype,
)

stream.start()
print("Recording...")
audio = stream.read(int(samplerate * 10))[0]
stream.stop()

# print("Recording...")
# audio = sd.rec(
#     int(duration * samplerate), 
#     samplerate=samplerate, 
#     channels=channels, 
#     dtype=dtype,
#     device=device
# )
# sd.wait()
sf.write("python_stereo_recording_50cm.wav", audio, samplerate)
print("Finished recording. Saved to python_stereo_recording_50cm.wav")