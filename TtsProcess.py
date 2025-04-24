
import pyttsx3
import time

def tts_loop(queue, running):
    engine = pyttsx3.init()
    while running.value:
        if not queue.empty():
            text = queue.get()
            print("🗣️ Speaking:", text)
            engine.say(text)
            engine.runAndWait()
        time.sleep(0.1)
