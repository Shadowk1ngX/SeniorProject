import pvporcupine
import pyaudio
import struct
import json
import whisper
import wave
import tempfile
import numpy as np
import time
import webrtcvad
from datetime import datetime
from AssistantCore import handle_command
import ChatbotGpt
import Sounds



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
gpt = ChatbotGpt.ChatbotGPT()

# === Audio Config ===
CHUNK_DURATION_MS = 30  # must be 10, 20, or 30 for webrtcvad
RATE = 16000
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)  # = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_RECORD_SECONDS = 30

# PyAudio setup
pa = pyaudio.PyAudio()

#def speak(text):
#    print("🗣️ Speaking:", text)
#    def run():
#        tts.say(text)
#        tts.runAndWait()
#    threading.Thread(target=run, daemon=True).start()
#def speak(text):
   # _speech_queue.put(text)



def record_until_silence(tts_queue):
    print("🎤 Listening for your command (VAD enabled)...")

    vad = webrtcvad.Vad(3)  # 0–3: higher = more aggressive (cuts noise better)

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
    GRACE_PERIOD = 5  # Max time allowed before user starts speaking
    user_started_speaking = False

    while True:
        data = record_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        is_speaking = vad.is_speech(data, RATE)
        current_time = time.time()

        # Optional visual debug
        print("🎙️ Speaking" if is_speaking else "…", end="\r")

        if is_speaking:
            last_voice_time = current_time
            user_started_speaking = True

        if not user_started_speaking and current_time - grace_start > GRACE_PERIOD:
            print("\n😐 No speech detected during grace period, stopping...")
            tts_queue.put("I didn't hear anything.")
            break

        if user_started_speaking and current_time - last_voice_time > 1.0:
            print("\n🤫 Silence detected, stopping...")
            break

        if current_time - start_time > MAX_RECORD_SECONDS:
            print("\n⏰ Max time reached, stopping...")
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

def transcribe_audio(file_path,LimitCheck = True):
    print("🧠 Transcribing with Whisper...")
    result = whisper_model.transcribe(file_path)
    text = result["text"].strip()
    print("📝 Text:", text)

    # Filter out empty or nonsense transcripts
    if LimitCheck:
        if len(text.split()) < 2:
            print("🤫 Likely noise or too short to be valid.")
            return ""

    return text

def listen_loop(running, assistant_state=None, state_lock=None, shared_frame=None, frame_lock=None, tts_queue=None):
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
            assistant_state["in_command"] = True
            print("VOICE")
            print("✅ Wake word detected!")
            tts_queue.put("How can I help?")
            time.sleep(1.2)
            Sounds.play_sound("ActiveRecordSound.mp3")
            command_audio = record_until_silence(tts_queue)
            Sounds.play_sound("EndRecordSound.mp3")
            wav_path = save_temp_wav(command_audio)
            text = transcribe_audio(wav_path)
            if text:
                handle_command(text, gpt_instance=gpt, assistant_state=assistant_state, lock=state_lock,shared_frame=shared_frame, frame_lock=frame_lock, tts_queue= tts_queue)

            assistant_state["in_command"] = False
            print("🎤 Listening for wake word again...")

    print("🛑 Stopping voice assistant...")
    stream.stop_stream()
    stream.close()
    porcupine.delete()
    pa.terminate()
