from ultralytics import YOLO
# used to display detections'
import cv2
import cvzone
# serves the rounding off of the confidence values
import math

# Webcam object - 0 indicates the number of cameras being used to capture the footage
# cap = cv2.VideoCapture(0)
# 3 = propID number 3, next to it is the width
# cap.set(3, 1280)
# 4 = propID number 4, next to it is the height
# cap.set(4, 720)

# To import videos and use on YOLO
cap = cv2.VideoCapture("../Videos/TDPOV (online-video-cutter.com).mp4")


# create the Model
model = YOLO('../yolo-weights/yolov8l.pt')

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

mask = cv2.imread("Mask-CV-project1.png")  # mask

# This should run the webcam by default
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    # check for individual bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # bounding box
            # to find the coordinates of each bounding box
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
            # creating the rectangle using cv2 library'
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # to use this fancy bounding box from CVZONE library, you have to change the variable and its values
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class names
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # text on top of the rectangle
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike":
                # draw bounding box and label for the object
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=5)

    cv2.imshow("Image", img)
    cv2.imshow("Image region", imgRegion)
    # delay time of the video capture
    cv2.waitKey(1)
