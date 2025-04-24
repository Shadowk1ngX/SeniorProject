import cv2
import face_recognition
import os
import pickle
import time
import numpy as np
import AssistantCore
from AssistantCore import update_faces
import VoiceRec as Voice
import Sounds


KNOWN_FACES_FILE = "known_faces.pkl"



def face_loop(shared_frame, lock, running, assistant_state, state_lock,tts_queue):
    recognizer = FaceRecognizer(fast_mode=False)
    name_prompt_delay = 10.0
    first_unknown_time = None

    while running.value:
        now = time.time()
        # ─── Read raw bytes directly from the Array ─────────
        with lock:
            raw = shared_frame.get_obj()  # ctypes buffer

        # ─── Build the image ───────────────────────────────
        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((480, 640, 3))

        # ─── Run the recognizer & update shared state ─────
        names = recognizer.process_frame(frame)
        update_faces(names, assistant_state, state_lock)
        #print("Detected names:", names)
        #print(assistant_state)

        # ─── Unknown‐face timer logic ──────────────────────
        if "Unknown" in names:
            #if assistant_state["in_command"]:
            #    print("⏳ Delaying face prompt — assistant is busy.")
             #   first_unknown_time = None  # 🧠 Reset timer while command is active
             #   continue
            if first_unknown_time is None:
                first_unknown_time = now
                
            elif (now - first_unknown_time) > name_prompt_delay and not assistant_state["in_command"]:
                print("FACE")
                with state_lock:
                    assistant_state["awaiting_name"] = True
                    assistant_state["pending_face_frame"] = bytes(raw)

                name_confirmed = False
                Name = ""

                while not name_confirmed:
                    tts_queue.put("I don’t recognize you. What’s your name?")
                    Sounds.play_sound("ActiveRecordSound.mp3")
                    Recording = Voice.record_until_silence(tts_queue)
                    Sounds.play_sound("EndRecordSound.mp3")
                    wav_path = Voice.save_temp_wav(Recording)
                    Name = Voice.transcribe_audio(wav_path, LimitCheck=False).strip()

                    if not Name or len(Name.split()) > 4:
                        tts_queue.put("Sorry, I didn't catch that.")
                        continue
                    
                    tts_queue.put(f"Your name is {Name}. Is that correct?")
                    Sounds.play_sound("ActiveRecordSound.mp3")
                    ConfirmationRecording = Voice.record_until_silence(tts_queue)
                    Sounds.play_sound("EndRecordSound.mp3")
                    ConfirmationWav_path = Voice.save_temp_wav(ConfirmationRecording)
                    ConfirmationText = Voice.transcribe_audio(ConfirmationWav_path, LimitCheck=False).lower()

                    if "yes" in ConfirmationText or "correct" in ConfirmationText or "yeah" in ConfirmationText or "incorrect" in ConfirmationText:
                        name_confirmed = True
                    elif "no" in ConfirmationText:
                        tts_queue.put("Alright, let's try again.")
                    else:
                        tts_queue.put("I didn't understand that. Please say yes or no.")

                # 🧠 Register only if name is confirmed
                success = recognizer.register_face(frame, Name)
                if success:
                    first_unknown_time = None
                    print("✅ Registered new face for", Name)
                    tts_queue.put(f"Thanks, {Name}. I’ll remember you.")
                    recognizer.known_faces = recognizer.load_known_faces()
                    # Optional: Add to seen list here
                    # with state_lock:
                    #     assistant_state["people_seen"].append(Name)
                else:
                    print("❌ Failed to register face; no encoding found.")
                    tts_queue.put("Sorry, I couldn’t see your face clearly—let’s try again.")
                    first_unknown_time = None  # Reset timer so it’ll re-prompt later


        cv2.waitKey(1)

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

        for (top, right, bottom, left), name in zip(face_locations, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if self.fast_mode:
                ...
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if not self.fast_mode:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        return names
    
    def register_face(self, frame, name):
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        encs  = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return False
        self.known_faces["encodings"].append(encs[0])
        self.known_faces["names"].append(name)
        self.save_known_faces()
        return True


# === Multiprocessing Entry Point ===
 