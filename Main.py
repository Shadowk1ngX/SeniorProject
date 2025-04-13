import ChatbotGpt as Chat
import FaceRec as Face
import ObjectDetection as Object

import threading
import cv2
import time

# Shared frame and lock
shared_frame = None
frame_lock = threading.Lock()
running = True

# Initialize webcam once
cap = cv2.VideoCapture(0)

def capture_frames():
    global shared_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            shared_frame = frame.copy()
        time.sleep(0.01)  # Optional: reduce CPU usage

def run_face_recognizer():
    global shared_frame, running
    recognizer = Face.FaceRecognizer()
    print("Face recognizer started")
    while running:
        with frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else None
        if frame is not None:
            recognizer.process_frame(frame)  # You’ll need to expose this

def run_object_detector():
    global shared_frame, running
    detector = Object.ObjectDetector(confidence_threshold=0.5)
    print("Object detector started")
    while running:
        with frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else None
        if frame is not None:
            detector.process_frame(frame)  # You’ll need to expose this

if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames)
    face_thread = threading.Thread(target=run_face_recognizer)
    object_thread = threading.Thread(target=run_object_detector)

    capture_thread.start()
    face_thread.start()
    object_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        cap.release()
        print("Shutting down...")
