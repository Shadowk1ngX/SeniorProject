import ChatbotGpt as Chat
import FaceRec as Face
import ObjectDetection as Object
import VoiceRec as Voice
from AssistantCore import update_faces
import cv2
import time
import numpy as np
from multiprocessing import Queue
from TtsProcess import tts_loop


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

def run_face_recognizer(shared_frame, frame_lock, running,assistant_state, state_lock, tts_queue):
    #recognizer = Face.FaceRecognizer(fast_mode=True)
    while running.value:
        Face.face_loop(shared_frame, frame_lock, running,assistant_state, state_lock, tts_queue)
        #with frame_lock:
            #raw = shared_frame.get_obj()
            #frame = np.frombuffer(raw, dtype=np.uint8).reshape((480,640,3))
        #names = recognizer.process_frame(frame)
        #update_faces(names, assistant_state, state_lock)


#def run_object_detector(shared_frame, lock, running):
 #   detector = Object.ObjectDetector(confidence_threshold=0.7, fast_mode=True)
  #  print("Object detector started")
   # while running.value:
   #     with lock:
   #         frame_np = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).copy().reshape((480, 640, 3))
   #     detector.process_frame(frame_np)

def start_voice_detection(running):
    print("Voice recognition started")
    Voice.listen_loop(running)  # Make sure this blocks or handles running flag


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # Moved inside, and added force=True

    tts_queue = multiprocessing.Queue()

    manager = multiprocessing.Manager()
    assistant_state = manager.dict({
        "people_seen": manager.list(),
        "objects_detected": manager.list(),
        "in_command": False,
        "awaiting_name": False,
        "awaiting_confirmation": False,
        "pending_face_frame": None,
    })

    state_lock = multiprocessing.Lock()

    # Shared memory and lock
    frame_shape = (480, 640, 3)
    shared_array = multiprocessing.Array('B', frame_shape[0] * frame_shape[1] * frame_shape[2])
    lock = multiprocessing.Lock()
    running = multiprocessing.Value('b', True)

    # Create processes
    capture_proc = multiprocessing.Process(target=capture_and_share_frames, args=(shared_array, lock, running))
    face_proc = multiprocessing.Process(target=run_face_recognizer, args=(shared_array, lock, running, assistant_state, state_lock, tts_queue))
    voice_proc = multiprocessing.Process(target=Voice.listen_loop, args=(running, assistant_state, state_lock, shared_array, lock, tts_queue))
    tts_proc = multiprocessing.Process(target=tts_loop, args=(tts_queue, running))

    # Start processes
    capture_proc.start()
    voice_proc.start()
    face_proc.start()
    tts_proc.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running.value = False
        print("Shutting down...")

    for proc in [voice_proc, capture_proc, face_proc, tts_proc]:
        proc.join()
