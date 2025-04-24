import threading
import queue
import pyttsx3

_speech_queue = queue.Queue()
_stop_signal = threading.Event()
_speech_thread = None

def _speech_worker():
    print("🔊 TTS Worker running")
    while not _stop_signal.is_set():
        try:
            text = _speech_queue.get(timeout=0.1)
            if text:
                print("🗣️ Speaking:", text)
                engine = pyttsx3.init()  # 👈 create a fresh engine each time
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except queue.Empty:
            continue
        except Exception as e:
            print("❌ TTS error:", e)
    print("🛑 TTS Worker shutting down")

def start():
    global _speech_thread
    if _speech_thread is None or not _speech_thread.is_alive():
        _speech_thread = threading.Thread(target=_speech_worker, daemon=True)
        _speech_thread.start()

def speak(text):
    _speech_queue.put(text)
    print("ADD TO QUEUE")

def stop():
    _stop_signal.set()
    if _speech_thread:
        _speech_thread.join()
