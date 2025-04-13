import ChatbotGpt as Chat
import FaceRec as Face
import ObjectDetection as Object

import threading



def run_face_recognizer():
    face = Face()
    face.recognize_and_track()

def run_object_detector():
    detector = Object(confidence_threshold=0.5)
    detector.detect_from_webcam()

if __name__ == "__main__":
    face_thread = threading.Thread(target=run_face_recognizer)
    object_thread = threading.Thread(target=run_object_detector)

    face_thread.start()
    object_thread.start()

    face_thread.join()
    object_thread.join()