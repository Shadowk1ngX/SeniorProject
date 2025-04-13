import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(0)

    def detect_from_webcam(self, on_detect=None):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)[0]
            detections = []

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detection = {
                    "label": label,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                }
                detections.append(detection)

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Optional callback to process detections
            if on_detect:
                on_detect(detections, frame)

            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
