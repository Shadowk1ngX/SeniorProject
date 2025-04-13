import cv2
import face_recognition
import os
import pickle

# Paths
KNOWN_FACES_FILE = "known_faces.pkl"

# Load known faces
def load_known_faces():
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

# Save known faces
def save_known_faces(data):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(data, f)

# Initialize data
known_faces = load_known_faces()
video = cv2.VideoCapture(0)

print("Press 's' to save a new face, 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Faster processing
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            best_match_index = matches.index(True)
            name = known_faces["names"][best_match_index]

        # Draw box and label
        top, right, bottom, left = [v * 4 for v in face_location]  # Resize back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognizer", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save new face
    if key == ord("s") and face_encodings:
        name = input("Enter name for the face: ")
        known_faces["encodings"].append(face_encodings[0])
        known_faces["names"].append(name)
        save_known_faces(known_faces)
        print(f"Saved {name}'s face.")

    elif key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
