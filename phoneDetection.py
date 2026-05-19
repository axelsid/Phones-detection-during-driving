import cv2
import supervision as sv
from ultralytics import YOLO

# YOLO11s pretrained on COCO — cell phone is class 67
model = YOLO("yolo11s.pt")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def detect_phones(image_path: str):
    image = cv2.imread(image_path)

    # classes=[67] filters to cell phone only, conf=0.15 catches low-confidence detections
    results = model.predict(image, classes=[67], conf=0.15)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"cell phone {conf:.2f}"
        for conf in detections.confidence
    ]

    annotated = box_annotator.annotate(image, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    cv2.imshow("Phone Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return not detections.is_empty()