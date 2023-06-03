'''

///////////////////////////
    THIS IS A KIND OF WORKING CODE
//////////////////////////

from typing import List, Tuple, Any

from ultralytics import YOLO
import cv2
import math
import numpy as np

cap = cv2.VideoCapture("../Videos/ScenicDrive1.mp4")
# cap = cv2.VideoCapture(0)

model = YOLO('../yolo-weights/yolov8m.pt')

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

KNOWN_WIDTH = 1.8  # Average car width is 1.8 meters
FOCAL_LENGTH = 700  # Focal length of your camera

def calculate_distance(pixel_width: float) -> float:
    # Calculate the distance to the object
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return distance

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
    current_pixel_widths = {}
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

                # Calculate the pixel width of the detected vehicle
                pixel_width = x2 - x1

                # Store the pixel width
                current_pixel_widths[centroid] = pixel_width

    matched_pairs, unmatched_new_centroids = match_centroids(tracked_centroids.copy(), current_centroids)

    for old_key, new_key in matched_pairs:
        prev_centroid_x = old_key[0]
        direction = ""

        if new_key[0] < prev_centroid_x:
            direction = "Incoming"
        elif new_key[0] > prev_centroid_x:
            direction = "Outgoing"

        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])

        tracked_centroids[new_key] = tracked_centroids.get(old_key, {"centroid": new_key, "class": current_centroids[new_key], "direction": "", "frames_not_detected": 0, "distance": distance})
        del tracked_centroids[old_key]

    for new_key in unmatched_new_centroids:
        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])

        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": "", "frames_not_detected": 0, "distance": distance}

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

            # Draw label with distance
            label = f'{currentClass} {conf} {value["direction"]} Distance: {value["distance"]:.2f}m'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - label_height), (x1 + label_width, y1), (255, 0, 255), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

///////////////////////////
    THIS IS A KIND OF WORKING CODE
//////////////////////////
'''

# //////////////////////////
    # THIS IS THE BEST ONE SO FAR WITH ALL THE LABELS APPEARING INCLUDING DISTANCE
# /////////////////////////
from typing import List, Tuple, Any

from ultralytics import YOLO
import cv2
import math
import numpy as np

cap = cv2.VideoCapture("../Videos/ScenicDrive1.mp4")
# cap = cv2.VideoCapture(0)

model = YOLO('../yolo-weights/yolov8m.pt')

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

KNOWN_WIDTH = 1.8  # Average car width is 1.8 meters
FOCAL_LENGTH = 700  # Focal length of your camera

def calculate_distance(pixel_width: float) -> float:
    # Calculate the distance to the object
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return distance

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
    current_pixel_widths = {}
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

                # Calculate the pixel width of the detected vehicle
                pixel_width = x2 - x1

                # Store the pixel width
                current_pixel_widths[centroid] = pixel_width

    matched_pairs, unmatched_new_centroids = match_centroids(tracked_centroids.copy(), current_centroids)

    for old_key, new_key in matched_pairs:
        prev_centroid_x = old_key[0]
        direction = ""

        if new_key[0] < prev_centroid_x:
            direction = "Incoming"
        elif new_key[0] > prev_centroid_x:
            direction = "Outgoing"

        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])
        print(distance)

        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": direction,
                                      "frames_not_detected": 0, "distance": distance}
        del tracked_centroids[old_key]

    for new_key in unmatched_new_centroids:
        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])

        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": "", "frames_not_detected": 0, "distance": distance}

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

            # Draw label with distance and direction
            # Draw label with distance
            label = f'{value["class"]} {value["direction"]} Distance: {value["distance"]:.2f}m'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Move labels below the bounding box
            cv2.rectangle(img, (x1, y2), (x1 + label_width, y2 + label_height), (255, 0, 255), -1)
            cv2.putText(img, label, (x1, y2 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

'''

from ultralytics import YOLO
import cv2
import math
import numpy as np
import time

cap = cv2.VideoCapture("../Videos/ScenicDrive1.mp4")
# cap = cv2.VideoCapture(0)

model = YOLO('../yolo-weights/yolov8m.pt')

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
PIXEL_TO_METERS = 0.5  # Estimation of conversion from pixels to meters

KNOWN_WIDTH = 1.8  # Average car width is 1.8 meters
# THE VALUE WAS 700
FOCAL_LENGTH = 1000  # Focal length of your camera

def calculate_distance(pixel_width: float) -> float:
    # Calculate the distance to the object
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return distance

def calculate_distance_moved(old_position, new_position):
    pixel_distance = np.sqrt((new_position[0] - old_position[0]) ** 2 + (new_position[1] - old_position[1]) ** 2)
    return pixel_distance * PIXEL_TO_METERS

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
    current_pixel_widths = {}
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

                # Calculate the pixel width of the detected vehicle
                pixel_width = x2 - x1

                # Store the pixel width
                current_pixel_widths[centroid] = pixel_width

    matched_pairs, unmatched_new_centroids = match_centroids(tracked_centroids.copy(), current_centroids)

    for old_key, new_key in matched_pairs:
        prev_centroid_x = old_key[0]
        direction = ""

        if new_key[0] < prev_centroid_x:
            direction = "Incoming"
        elif new_key[0] > prev_centroid_x:
            direction = "Outgoing"

        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])
        print(distance)

        # Calculate speed
        distance_moved = calculate_distance_moved(old_key, new_key)
        speed = distance_moved / (1/30)  # Assuming 30 FPS

        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": direction,
                                      "frames_not_detected": 0, "distance": distance, "speed": speed}
        del tracked_centroids[old_key]

    for new_key in unmatched_new_centroids:
        # Calculate the distance
        distance = calculate_distance(current_pixel_widths[new_key])

        tracked_centroids[new_key] = {"centroid": new_key, "class": current_centroids[new_key], "direction": "", "frames_not_detected": 0, "distance": distance, "speed": 0}

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

            # Draw label with distance and direction
            # Draw label with distance
            label = f'{value["class"]} {value["direction"]} Distance: {value["distance"]:.2f}m Speed: {value["speed"]:.2f}m/s'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Move labels below the bounding box
            cv2.rectangle(img, (x1, y2), (x1 + label_width, y2 + label_height), (255, 0, 255), -1)
            cv2.putText(img, label, (x1, y2 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1) '''