import ChatbotGpt as Chat
import FaceRec as Face
import ObjectDetection as Object
import VoiceRec as Voice

import multiprocessing
import cv2
import time
import numpy as np

def capture_and_share_frames(shared_frame, lock, running):
    cap = cv2.VideoCapture(0)
    while running.value:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            shared_frame[:] = frame.flatten()
        time.sleep(0.03) #30 fps
    cap.release()

def run_face_recognizer(shared_frame, lock, running):
    import numpy as np
    recognizer = Face.FaceRecognizer(fast_mode=True)
    while running.value:
        with lock:
            frame_np = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).copy().reshape((480, 640, 3))
        recognizer.process_frame(frame_np)


def run_object_detector(shared_frame, lock, running):
    detector = Object.ObjectDetector(confidence_threshold=0.7, fast_mode=True)
    print("Object detector started")
    while running.value:
        with lock:
            frame_np = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).copy().reshape((480, 640, 3))
        detector.process_frame(frame_np)

def start_voice_detection(running):
    print("Voice recognition started")
    Voice.listen_loop(running)  # Make sure this blocks or handles running flag

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Better compatibility on Windows/macOS

    # Shared memory and lock
    frame_shape = (480, 640, 3)  # adjust this to your camera's resolution
    shared_array = multiprocessing.Array('B', frame_shape[0] * frame_shape[1] * frame_shape[2])
    lock = multiprocessing.Lock()
    running = multiprocessing.Value('b', True)

    # Create processes
    capture_proc = multiprocessing.Process(target=capture_and_share_frames, args=(shared_array, lock, running))
    face_proc = multiprocessing.Process(target=run_face_recognizer, args=(shared_array, lock, running))
    object_proc = multiprocessing.Process(target=run_object_detector, args=(shared_array, lock, running))
    voice_proc = multiprocessing.Process(target=Voice.listen_loop, args=(running,))

    # Start processes
    voice_proc.start()
    capture_proc.start()
    face_proc.start()
    object_proc.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running.value = False
        print("Shutting down...")

    # Join processes
    for proc in [voice_proc, capture_proc, face_proc, object_proc]:
        proc.join()