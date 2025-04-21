import cv2
import face_recognition
import os
import pickle
import numpy as np
from AssistantCore import update_faces

KNOWN_FACES_FILE = "known_faces.pkl"

class FaceRecognizer:
    def __init__(self, fast_mode=False):
        self.known_faces = self.load_known_faces()
        self.fast_mode = fast_mode

    def load_known_faces(self):
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, "rb") as f:
                return pickle.load(f)
        return {"encodings": [], "names": []}

    def save_known_faces(self):
        with open(KNOWN_FACES_FILE, "wb") as f:
            pickle.dump(self.known_faces, f)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces["encodings"], face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = matches.index(True)
                name = self.known_faces["names"][best_match_index]
            names.append(name)

        update_faces(names)

        for (top, right, bottom, left), name in zip(face_locations, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if self.fast_mode:
                #print(f"[Face] {name}")
                ...
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if not self.fast_mode:
            cv2.imshow("Face Recognition", frame)
            cv2.waitKey(1)

# === Multiprocessing Entry Point ===
def face_loop(shared_frame, lock, running):
    print("[FaceRec] Face recognition process started")
    recognizer = FaceRecognizer(fast_mode=True)

    width, height = 640, 480  # Match the camera resolution used in main.py

    while running.value:
        with lock:
            frame_data = shared_frame[:]
        frame = bytearray(frame_data)
        frame_np = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame_np is not None:
            recognizer.process_frame(frame_np)

    print("[FaceRec] Face recognition process stopping...")
