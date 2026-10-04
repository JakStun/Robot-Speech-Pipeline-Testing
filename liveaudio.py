import sounddevice as sd
import numpy as np
import httpx
import queue
import asyncio
import torch
import time
import io
import wave
import tempfile
import os

from silero_vad import load_silero_vad

from livekit.wakeword import WakeWordModel

from dataclasses import dataclass


SAMPLE_RATE = 16000

WAKEWORD_WINDOW = 32000      # 2 seconds
WAKEWORD_STRIDE = SAMPLE_RATE      # 1 second

CHUNK_SIZE = 512

SILENCE_TIMEOUT = 1  # seconds

@dataclass
class AudioFrame:
    samples: np.ndarray
    samples_16k: np.ndarray
    overflow: bool


class Microphone:

    def __init__(self, device="plughw:1,0", sample_rate=16000, channels=2) -> None:
        self.device = device

        self.sample_rate = sample_rate
        self.output_sample_rate = 16000
        
        self.channels = channels

        self.frame_size = 512 # 512 samples at 16 kHz = 32 ms

        self.queue = queue.Queue(maxsize=10)

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int32",
            blocksize=self.frame_size,
            callback=self._audio_callback,
        )

        self.stream.start()

    async def read_frame(self) -> AudioFrame:
        return await asyncio.to_thread(
            self.queue.get
        )

    def pause(self) -> None:
        if self.stream.active:
            self.stream.stop()

    def resume(self) -> None:
        if not self.stream.active:
            self.stream.start()

    def close(self) -> None:
        self.pause()
        self.stream.close()

    def clear(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _audio_callback(self, indata, frames, time, status) -> None:
        # take one mic channel -> TODO: Check what it means

        # taking channel 0
        ch0 = indata[:, 0]

        frame = ch0.copy()

        frame_16k = frame.astype(np.int32)

        audio_frame = AudioFrame(
            samples=frame,
            samples_16k=frame_16k,
            overflow=status.input_overflow
        )

        try:
            self.queue.put_nowait(audio_frame)

        except queue.Full:
            self.queue.get_nowait()
            self.queue.put_nowait(audio_frame)


class UtteranceRecorder:

    def __init__(self, silence_timeout=1.0) -> None:
        self.silence_timeout = silence_timeout

        self.recording = False
        self.silence_start = None

        self.audio = []

    def start(self) -> None:
        self.recording = True
        self.silence_start = None
        self.audio = []

        print("Wakeword!")

    def process(self, frame: AudioFrame) -> None:

        if self.recording:
            #TODO: Fix this part, high pitches in recordings
            audio = frame.samples.astype(np.float32)

            audio -= audio.mean()

            audio = np.clip(audio, -32768, 32767).astype(np.int16)

            self.audio.append(audio.copy())

    def speech_started(self) -> None:
        self.silence_start = None

    def speech_ended(self) -> None:
        self.silence_start = time.monotonic()

    def update(self) -> np.ndarray | None:

        if not self.recording:
            return None

        if self.silence_start is None:
            return None

        if time.monotonic() - self.silence_start >= self.silence_timeout:

            self.recording = False
            self.silence_start = None

            audio = np.concatenate(self.audio)

            self.audio = []

            return audio

        return None


class WakeWordDetector:

    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

        self.model = WakeWordModel(models=["hey_livekit.onnx"])
        self.buffer = np.zeros(0, dtype=np.int16)

    def process(self, frame: AudioFrame, speech_active: bool) -> bool:
        self.buffer = np.concatenate(
            (self.buffer, frame.samples_16k)
        )[-WAKEWORD_WINDOW:]

        if not speech_active:
            return False

        if len(self.buffer) < WAKEWORD_WINDOW:
            return False

        chunk = self.buffer[:WAKEWORD_WINDOW]

        self.buffer = self.buffer[WAKEWORD_STRIDE:]

        scores = self.model.predict(chunk)

        print(f"score: {scores['hey_livekit']:.3f}")

        return scores["hey_livekit"] > self.threshold


class SpeechDetector:

    def __init__(self, start_threshold=0.9, end_threshold=0.3) -> None:
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.model = load_silero_vad()

        self.speech_active = False
        self.event = None

        # self.vad = VADIterator(
        #     self.model,
        #     sampling_rate=16000,
        #     threshold=0.8,
        #     min_silence_duration_ms=300,
        # )

        # self.event = None

        self.buffer = np.zeros(0, dtype=np.int16)

        self.end_silence_samples = int(
            SAMPLE_RATE * 0.5
        )

        self.silence_samples = 0

    def process(self, frame: AudioFrame) -> None:
        self.event = None

        self.buffer = np.concatenate(
            (self.buffer, frame.samples_16k)
        )

        while len(self.buffer) >= CHUNK_SIZE:
            chunk = self.buffer[:CHUNK_SIZE]

            self.buffer = self.buffer[CHUNK_SIZE:]

            audio = torch.from_numpy(chunk).float() / 32768.0
            audio = audio - audio.mean() # need to change audio data -> remove DC offset

            confidence = self.model(audio, SAMPLE_RATE).item()

            if not self.speech_active:

                if confidence >= self.start_threshold:
                    self.speech_active = True
                    self.silence_samples = 0
                    self.event = {"start": confidence}

            else:

                if confidence < self.end_threshold:
                    self.silence_samples += CHUNK_SIZE
                else:
                    self.silence_samples = 0

                if self.silence_samples >= self.end_silence_samples:
                    self.speech_active = False
                    self.silence_samples = 0
                    self.event = {"end": confidence}

    def started(self) -> bool:
        return (
            self.event is not None
            and "start" in self.event
        )

    def ended(self) -> bool:
        return (
            self.event is not None
            and "end" in self.event
        )


class SpeechClient:

    def __init__(self, server_url: str = "http://192.168.1.136:8000", robot_id: str = "mHY4kxrUKqInYVcFXwiZM") -> None:
        self.serve_url = server_url
        self.robot_id = robot_id

        self.task = None

        self.client = httpx.AsyncClient(
            timeout=1200
        )

    def send(self, audio: np.ndarray):
        """Ignore new requests while waiting for server response"""

        if self.task is not None:
            return

        # wav = self._audio_to_wav(audio)

        # with open("debug_recording.wav", "wb") as f:
        #     f.write(wav.read())

        self.task = asyncio.create_task(
            self._process(audio)
        )

    def update(self):

        if self.task is None:
            return None

        if not self.task.done():
            return None

        response = self.task.result()

        self.task = None

        return response

    async def _process(self, audio):
        wav = self._audio_to_wav(audio)

        try:
            response = await self.client.post(
                f"{self.serve_url}/api/v1/audio/post",

                headers={
                    "X-Robot-Id": self.robot_id,
                },

                files={
                    "audio_file": (
                        "speech.wav",
                        wav,
                        "audio/wav",
                    )
                }
            )

            response.raise_for_status()

            return response.content
        
        except Exception as e:
            print("Error receiving answer")

    def _audio_to_wav(self, audio: np.ndarray) -> io.BytesIO:
        wav = io.BytesIO()

        with wave.open(wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(48000)

            f.writeframes(audio.tobytes())

        wav.seek(0)

        return wav


class SpeechPlayer:

    def __init__(self):
        self.task = None

    @property
    def is_playing(self) -> bool:
        '''Needed for LED animation'''
        return self.task is not None

    def play(self, audio: bytes) -> None:

        if self.task is not None:
            return

        self.task = asyncio.create_task(
            self._play(audio)
        )

    async def _play(self, audio: bytes) -> None:
        temp_path = None

        try:
            # temp wav file:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as file:
                file.write(audio)
                temp_path = file.name

            # Windows logic:
            # process = await asyncio.create_subprocess_exec(
            #     "powershell",
            #     "-c",
            #     f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()'
            # )

            # RPi Logic:
            process = await asyncio.create_subprocess_exec(
                "aplay",
                temp_path,
            )

            await process.wait()

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def update(self) -> bool:

        if self.task is None:
            return False

        if not self.task.done():
            return False

        self.task = None

        return True


async def main():
    microphone = Microphone()
    vad = SpeechDetector()
    wakeword = WakeWordDetector()
    utterance = UtteranceRecorder()
    speech = SpeechClient()
    player = SpeechPlayer()

    spinner = "|/-\\"
    spinner_index = 0

    last_spinner = time.monotonic()

    print("Listening for speech")
    
    while True:
        # TEMP thinking/speaking Animation
        if speech.task is not None or player.is_playing:

            if time.monotonic() - last_spinner > 0.1:

                print(
                    spinner[spinner_index],
                    end="\r",
                    flush=True
                )

                spinner_index = (spinner_index + 1) % len(spinner)

                last_spinner = time.monotonic()

        
        frame = await microphone.read_frame()
        # print("read frame", end="\r")

        vad.process(frame)
        # print("processed vad", end="\r")
        utterance.process(frame)
        # print("processed utterance", end="\r")

        if vad.started():
            print("started talking")
            utterance.speech_started()

        if vad.ended():
            print("finished talking")
            utterance.speech_ended()

        if not utterance.recording and wakeword.process(frame, vad.speech_active):
            utterance.start()

        audio = utterance.update()

        if audio is not None:
            print(f"Recorded {len(audio)} frames -> sending to server")

            speech.send(audio)


        response_audio = speech.update()
        
        if response_audio is not None:
            print("Received response -> Playing:")
            player.play(response_audio)

        if player.update():
            print("Finished speaking")


if __name__ == "__main__":
    asyncio.run(main())