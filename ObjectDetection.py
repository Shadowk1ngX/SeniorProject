import cv2
import numpy as np
from ultralytics import YOLO
from AssistantCore import update_objects

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.7, fast_mode=False):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.fast_mode = fast_mode
        self.model.overrides['verbose'] = False

    def process_frame(self, frame):
        results = self.model(frame)[0]
        labels_detected = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            labels_detected.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if self.fast_mode:
                #print(f"[Object] {label} ({conf:.2f})")
                ...
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        update_objects(labels_detected)

        if not self.fast_mode:
            cv2.imshow("Object Detection", frame)
            cv2.waitKey(1)

# === Multiprocessing Entry Point ===
def detect_loop(shared_frame, lock, running):
    print("[ObjectDetection] Object detector process started")
    detector = ObjectDetector(confidence_threshold=0.5, fast_mode=True)

    width, height = 640, 480  # match your camera resolution

    while running.value:
        with lock:
            frame_data = shared_frame[:]
        frame = bytearray(frame_data)
        frame_np = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame_np is not None:
            detector.process_frame(frame_np)

    print("[ObjectDetection] Object detector process stopping...")
