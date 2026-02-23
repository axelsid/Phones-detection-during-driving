import cv2
import kornia as K
import numpy as np
import matplotlib.pyplot as plt
import torch
from kornia.contrib import FaceDetector, FaceDetectorResult

device = torch.device("cpu")
dtype = torch.float32

img_cv = cv2.imread(r"C:\Users\alexa\PycharmProjects\Anonimizer\people.jpg")

img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

img = torch.from_numpy(img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0


img_vis = K.tensor_to_image(img.byte())

plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.show()

face_detection = FaceDetector().to(device, dtype)

with torch.no_grad():
    dets = face_detection(img)

dets = [FaceDetectorResult(o) for o in dets]

k: int = 21
s: float = 35.0


def apply_blur_face(img: torch.Tensor, img_vis: np.ndarray, x1, y1, x2, y2):
    roi = img[..., y1:y2, x1:x2]

    roi = K.filters.gaussian_blur2d(roi, (k, k), (s, s))
    img_vis[y1:y2, x1:x2] = K.tensor_to_image(roi)

for b in dets:
    top_left = b.top_left.int().tolist()
    bottom_right = b.bottom_right.int().tolist()
    scores = b.score.tolist()

    for score, tp, br in zip(scores, top_left, bottom_right):
        x1, y1 = tp
        x2, y2 = br

        if score < 0.7:
            continue
        img_vis = cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        apply_blur_face(img, img_vis, x1, y1, x2, y2)

plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.show()



