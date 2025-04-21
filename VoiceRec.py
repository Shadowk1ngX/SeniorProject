import pvporcupine
import pyaudio
import struct
import json
import whisper
import wave
import tempfile
import numpy as np
import time
import pyttsx3
from datetime import datetime
from AssistantCore import handle_command
import ChatbotGpt

# === Load Access Key ===
with open('Key.json', 'r') as file:
    data = json.load(file)
ACCESS_KEY = data["PvporcupineKey"]

# === Load Models ===
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=["WindowsKeyWord.ppn"]  # Replace with Pi version when deploying
)
whisper_model = whisper.load_model("base")
tts = pyttsx3.init()
gpt = ChatbotGpt.ChatbotGPT()

# === Audio Config ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_RECORD_SECONDS = 30
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2

# PyAudio setup
pa = pyaudio.PyAudio()

def speak(text):
    print("🗣️ Speaking:", text)
    tts.say(text)
    tts.runAndWait()

def rms(data):
    shorts = np.frombuffer(data, dtype=np.int16)
    if len(shorts) == 0:
        return 0
    return np.sqrt(np.mean(shorts.astype(np.float32) ** 2))

def record_until_silence():
    print("🎤 Listening for your command (max 30s)...")

    record_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    start_time = time.time()
    last_voice_time = time.time()
    grace_start = time.time()
    GRACE_PERIOD = 5
    user_started_speaking = False

    while True:
        data = record_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        volume = rms(data)
        #print(f"Volume: {volume}")

        current_time = time.time()

        if volume > SILENCE_THRESHOLD:
            last_voice_time = current_time
            user_started_speaking = True

        if not user_started_speaking and current_time - grace_start > GRACE_PERIOD:
            print("😐 No speech detected during grace period, stopping...")
            speak("I didn't hear anything.")
            break

        if user_started_speaking and current_time - last_voice_time > SILENCE_DURATION:
            print("🤫 Silence detected, stopping...")
            break

        if current_time - start_time > MAX_RECORD_SECONDS:
            print("⏰ Max time reached, stopping...")
            break

    record_stream.stop_stream()
    record_stream.close()

    return b"".join(frames)

def save_temp_wav(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wf = wave.open(tmpfile.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()
        return tmpfile.name

def transcribe_audio(file_path):
    print("🧠 Transcribing with Whisper...")
    result = whisper_model.transcribe(file_path)
    text = result["text"].strip()
    print("📝 Text:", text)
    return text

def listen_loop(running):
    print("🧠 listen_loop() was called")

    try:
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        print("🎧 Microphone stream opened successfully!")
    except Exception as e:
        print("❌ Failed to open microphone stream:", e)
        return

    print("🎤 Voice Assistant is listening for 'Hey Jarvis'...")

    while running.value:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("✅ Wake word detected!")
            speak("How can I help?")
            command_audio = record_until_silence()
            wav_path = save_temp_wav(command_audio)
            text = transcribe_audio(wav_path)
            if text:
                handle_command(text, speak, gpt_instance=gpt)
            print("🎤 Listening for wake word again...")

    print("🛑 Stopping voice assistant...")
    stream.stop_stream()
    stream.close()
    porcupine.delete()
    pa.terminate()
