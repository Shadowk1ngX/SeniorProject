import face_recognition
import pickle
import os

# File to store face encodings and names
ENCODINGS_FILE = "face_encodings.pkl"

# Initialize known faces and names
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
    print(f"Debug: Loaded {len(known_face_encodings)} known face encodings.")
else:
    known_face_encodings = []
    known_face_names = []
    print("Debug: No existing face encodings found. Starting fresh.")

def save_encodings():
    """
    Saves known face encodings and names to a file.
    """
    try:
        with open(ENCODINGS_FILE, "wb") as file:
            pickle.dump((known_face_encodings, known_face_names), file)
        print(f"Debug: Saved {len(known_face_encodings)} face encodings to file.")
    except Exception as e:
        print(f"Error: Failed to save face encodings. {e}")

def add_new_face(frame, name):
    """
    Adds a new face to the known encodings.

    Parameters:
        frame (numpy array): A single frame from the video feed.
        name (str): The name of the person associated with the face.

    Returns:
        bool: True if the face was added successfully, False otherwise.
    """
    try:
        # Convert the frame from BGR (OpenCV format) to RGB
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        print(f"Debug: RGB frame shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Debug: Detected face locations - {face_locations}")

        if not face_locations:
            print("No face detected. Ensure your face is clearly visible.")
            return False

        # Validate the face_locations output
        for loc in face_locations:
            if len(loc) != 4 or not all(isinstance(coord, int) for coord in loc):
                print(f"Error: Invalid face location format: {loc}")
                return False

        # Compute face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Debug: Found {len(face_encodings)} face encodings.")

        if face_encodings:
            # Add the first face encoding and associated name
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

            # Save encodings to the file
            save_encodings()

            print(f"Successfully added face for {name}.")
            return True
        else:
            print("Error: No face encodings returned.")
            return False

    except Exception as e:
        print(f"Error while computing face encodings: {e}")
        return False


def recognize_faces(frame):
    """
    Recognizes faces in the given frame.

    Parameters:
        frame (numpy array): A single frame from the video feed.

    Returns:
        list: List of tuples containing names and face locations.
    """
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"Debug: Detected face locations - {face_locations}")

    if not face_locations:
        return []

    try:
        # Get face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Debug: Found {len(face_encodings)} face encodings.")
    except Exception as e:
        print(f"Error in face_encodings: {e}")
        return []

    recognized_faces = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        print(f"Debug: Matches found - {matches}")

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            print(f"Debug: Recognized as {name}.")
        else:
            print("Debug: Face not recognized.")

        recognized_faces.append((name, (top, right, bottom, left)))

    return recognized_faces
