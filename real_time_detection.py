import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
best_model = YOLO("C:\\Users\\Rohit\\objec\\Object-Detection-in-Adverse-Weather-Conditions\\runs\\detect\\train2\\weights\\best.pt")

# Capture video from the laptop's webcam (0 is usually the default camera)
cap = cv2.VideoCapture("http://100.72.117.24:8080/video")


# Set the resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = best_model(frame)
    result = results[0]

    # Extract bounding boxes
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

    # Draw bounding boxes
    for bbox in bboxes:
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

    # Display the real-time video with detections
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
