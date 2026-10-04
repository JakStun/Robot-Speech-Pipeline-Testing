# Speech detection and streaming pipeline learning

- this is a subproject to Gneral-Kalani-Project

## Learning goals:

- how VADs work (basics) - DONE
- how WAKEWORD logic works - DONE
- how to record audio locally/on rpi - DONE
- how to send requests to server - DONE
- how mic + RPi logic works together - DONE

### After you've cloned the rep and installed all needed packages, download hey_livekit.onnx separately (too big for github commits)

## Notes:

- SileroVAD is much better than webrtc -> stomps, whistling, background noises are not recognized as human speech -> ```Need to downgrade pytorch and pyaudio to 2.8.0```
- LiveKit is better for my main project than openWakeWord -> wakeword is recognized only once, no more multiple same actions after each other (no problems with downloading)
- Sounddevice package is running smoothly on rpi as well
- mic needed more tweaking -> recordings were very silent (vad couldn't recognize speech from silence), for further info see official adafruit docs for mems mics
- Asyncio must be used without question. Without it, only one process can run at one time

## Components:

- RPi 4 B 4GB
- 2x Adafruit I2S MEMS Microphone Breakout - SPH0645LM4H -> discontinued
- 2x INMP441 I2S MICs -> can record in 16kHz (needed for wakeword recognition)
- Adafruit Stereo Enclosed Speaker Set - 3W 4 Ohm
- 2x MAX98357 I2S mono amplifier 3W
- Breadboard + M-M & M-F cables for connecting everything up

## Architecture:

1. Microphone:
    - always listening (is paused only during processing)
    - uses sounddevice InputStream class to stream frames (32 ms audio frames)
    - sounddevice continuously receives audio from the microphone
    - passes audio frames to other classes
    - funcs:
        - read_frame() -> returns newest AudioFrame
        - pause() -> stops InputStream
        - resume() -> starts InputStream
        - close() -> stops InputStream (maybe not needed anymore)
        - clear() -> clears stream using Queue.get_nowait()
        - _audio_callback() -> callback func for InputStream, updates queue with new AudioFrame

2. SpeechDetector:
    - uses silero-vad model and its confidence generator to determine speech
    - consumes modified frames passed from Microphone.read_frame()

3. UtteranceDetector:
     - records commands for brain after wakeword is said
     - consumes/records frames passed from microphone class
     - funcs:
          - start() -> starts recording
          - 

4. WakeWordDetector:
     - checks for wake word detection

5. SpeechClient:
     - handles server (brain) communication

6. SpeechPlayer:
     - plays retrieved auido (response)


                    ┌─────────────────────┐
                    │      Microphone     │
                    │  sounddevice / ALSA │
                    └──────────┬──────────┘
                               │
                         AudioFrame
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       SpeechDetector    WakeWordDetector   UtteranceRecorder
        (Silero-vad)        (LiveKit)            (buffer)
              │                │                │
              │                │                │
              └────────────────┼────────────────┘
                               │
                         complete audio
                               │
                               ▼
                       ┌──────────────┐
                       │ SpeechClient │
                       │    HTTP      │
                       └──────┬───────┘
                              │
                         WAV upload
                              │
                              ▼
                         FastAPI server
                              │
                       STT → LLM → TTS
                              │
                         WAV response
                              │
                              ▼
                       ┌──────────────┐
                       │ SpeechPlayer │
                       └──────┬───────┘
                              │
                         speaker output


0 bcm2835 Headphones: - (hw:0,0), ALSA (0 in, 8 out)
> 1 snd_rpi_googlevoicehat_soundcar: Google voiceHAT SoundCard HiFi voicehat-hifi-0 (hw:1,0), ALSA (2 in, 0 out)
  2 sysdefault, ALSA (0 in, 128 out)
  3 speakerbonnet, ALSA (2 in, 0 out)
  4 dmixer, ALSA (0 in, 2 out)
  5 softvol, ALSA (0 in, 2 out)
  6 dmic_hw, ALSA (2 in, 0 out)
  7 dmic_sv, ALSA (2 in, 0 out)
  8 dmix, ALSA (0 in, 2 out)
< 9 default, ALSA (0 in, 128 out)