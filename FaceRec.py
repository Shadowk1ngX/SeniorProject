import cv2
import face_recognition
import os
import pickle

KNOWN_FACES_FILE = "known_faces.pkl"

class FaceRecognizer:
    def __init__(self):
        self.known_faces = self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, "rb") as f:
                return pickle.load(f)
        return {"encodings": [], "names": []}

    def save_known_faces(self):
        with open(KNOWN_FACES_FILE, "wb") as f:
            pickle.dump(self.known_faces, f)

    def recognize_and_track(self, on_frame=None):
        video = cv2.VideoCapture(0)
        print("Press 's' to save a new face, 'q' to quit.")

        while True:
            ret, frame = video.read()
            if not ret:
                break

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

            # Draw rectangles
            for (top, right, bottom, left), name in zip(face_locations, names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if on_frame:
                on_frame(frame, names)

            cv2.imshow("Face Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and face_encodings:
                name = input("Enter name for the face: ")
                self.known_faces["encodings"].append(face_encodings[0])
                self.known_faces["names"].append(name)
                self.save_known_faces()
                print(f"Saved {name}'s face.")

            elif key == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()
