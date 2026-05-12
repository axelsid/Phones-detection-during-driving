import argparse
import cv2
import kornia as K
import torch
from kornia.contrib import FaceDetector, FaceDetectorResult
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def anonymize(img_path: Path, score_threshold: float = 0.6):
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # <-- cast Path to str
    if img_raw is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    h, w = img_raw.shape[:2]

    img = K.image_to_tensor(img_raw, keepdim=False).to(device, dtype)
    img = K.color.bgr_to_rgb(img)

    face_detection = FaceDetector().to(device, dtype)

    with torch.no_grad():
        dets = face_detection(img)

    dets = [FaceDetectorResult(o) for o in dets[0]]

    img_out = img_raw.copy()

    for face in dets:
        if face.score < score_threshold:
            continue

        x1, y1 = face.top_left.int().tolist()
        x2, y2 = face.bottom_right.int().tolist()

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        face_region = img_out[y1:y2, x1:x2]
        kw = max(3, (x2 - x1) // 3 | 1)
        kh = max(3, (y2 - y1) // 3 | 1)
        img_out[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (kw, kh), 30)

        print(f"Anonymized face: score={face.score:.2f}, box=({x1},{y1})->({x2},{y2})")

    return img_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=Path)
    parser.add_argument("--score_threshold", type=float, default=0.6)
    args = parser.parse_args()
    anonymize(args.img_path, args.score_threshold)