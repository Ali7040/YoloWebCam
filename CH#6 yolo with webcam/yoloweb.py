import cv2
from ultralytics import YOLO
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../yolo-Weight/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag" "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich"
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "sofa", "potted plant", "bed",
              "dining-table", "toilet", "monitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush", "sticky notes", "Pen", "Guitar"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            # CLASS NAME
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)))

    cv2.imshow("Image", img)

    cv2.waitKey(1)

# import cv2
# from ultralytics import YOLO
# import cvzone
# import math
#
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
#
# model = YOLO("../yolo-Weight/yolov8n.pt")
#
# # Define the proper class names for the YOLO model
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#               "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#               "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#               "sofa", "potted plant", "bed",
#               "dining-table", "toilet", "monitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#               "hair drier", "toothbrush", "sticky notes", "Pen", "Guitar", "screen"]
#
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # CLASS NAME
#             cls = int(box.cls[0])
#
#             # Ensure the class index is within the classNames list range
#             if cls < 0 or cls >= len(classNames):
#                 continue
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)))
#
#     cv2.imshow("Image", img)
#
#     if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
#         break
#
# cap.release()
# cv2.destroyAllWindows()
