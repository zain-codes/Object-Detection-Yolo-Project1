from typing import List, Tuple, Any

from ultralytics import YOLO
import cv2
import math
import numpy as np

cap = cv2.VideoCapture("../Videos/ScenicDrive1.mp4")

model = YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracked_centroids = {}
max_frames_not_detected = 5


def match_centroids(old_centroids, new_centroids, max_distance=30):
    matched_pairs = []
    unmatched_new_centroids = []

    for new_key, new_value in new_centroids.items():
        matched = False
        min_distance = max_distance
        matched_key = None

        for old_key, old_value in old_centroids.items():
            distance = np.sqrt((new_key[0] - old_key[0]) ** 2 + (new_key[1] - old_key[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                matched_key = old_key

        if matched_key is not None:
            matched_pairs.append((matched_key, new_key))
            del old_centroids[matched_key]
        else:
            unmatched_new_centroids.append(new_key)

    return matched_pairs, unmatched_new_centroids


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    current_centroids = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"]:
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                current_centroids[centroid] = currentClass

    matched_pairs, unmatched_new_centroids = match_centroids(tracked_centroids.copy(), current_centroids)

    for old_key, new_key in matched_pairs:
        prev_centroid_x = old_key[0]
        direction = ""

        if new_key[0] < prev_centroid_x:
            direction = "incoming"
        elif new_key[0] > prev_centroid_x:
            direction = "outgoing"

        tracked_centroids[new_key] = tracked_centroids.get(old_key, {"centroid": new_key, "class": current_centroids [new_key], "direction": ""})
        tracked_centroids[new_key]["centroid"] = new_key
        tracked_centroids[new_key]["direction"] = direction
        del tracked_centroids[old_key]

    for new_key in unmatched_new_centroids:
        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": "",
                                      "frames_not_detected": 0}

    centroids_to_remove = []
    for key, value in tracked_centroids.items():
        if key not in current_centroids:
            value["frames_not_detected"] += 1
            if value["frames_not_detected"] >= max_frames_not_detected:
                centroids_to_remove.append(key)
        else:
            value["frames_not_detected"] = 0

    for key in centroids_to_remove:
        del tracked_centroids[key]

    for key, value in tracked_centroids.items():
        x1, y1 = key[0] - w // 2, key[1] - h // 2
        x2, y2 = x1 + w, y1 + h

        if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike":
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Draw label and confidence
            label = f'{currentClass} {conf} {value["direction"]} '
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - label_height), (x1 + label_width, y1), (255, 0, 255), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
