import cv2
import kornia as K
import numpy as np
import matplotlib.pyplot as plt
import torch
from kornia.contrib import FaceDetector
from PIL import Image

# --- konfiguracja ---
device = torch.device("cpu")
dtype = torch.float32
k, s = 21, 35.0  # kernel i sigma do gaussian blur

# --- wczytanie obrazu ---
with Image.open("ezgif.com-gif-maker-3.jpg") as im:
    im = im.convert("RGB")
    img_np = np.array(im)

# konwersja do tensora [B, C, H, W] i skala 0-1
img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# --- funkcja do blur twarzy ---
def apply_blur_face(img: torch.Tensor, x1, y1, x2, y2):
    h, w = img.shape[-2:]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    roi = img[..., y1:y2, x1:x2]
    if roi.numel() == 0:
        return
    roi_blur = K.filters.gaussian_blur2d(roi, (k, k), (s, s))
    img[..., y1:y2, x1:x2] = roi_blur

# --- wykrywanie twarzy ---
face_detector = FaceDetector().to(device, dtype)
with torch.no_grad():
    dets = face_detector(img)

# dets może być listą tensorów lub tensor 2D [num_faces, N]
faces = []
if isinstance(dets, list):
    faces = dets
else:  # tensor
    for i in range(dets.shape[0]):
        faces.append(dets[i])

# --- przetwarzanie każdej twarzy ---
for det in faces:
    det = det.squeeze()
    vals = []

    # spłaszczamy wszystkie listy wewnątrz tensora
    for v in det.tolist():
        if isinstance(v, list):
            vals.extend(v)
        else:
            vals.append(v)

    # jeśli mamy klasyczny format Kornia [15] -> [x1, y1, x2, y2, ..., score]
    if len(vals) >= 15:
        x1, y1, x2, y2 = map(lambda v: max(0, int(v)), vals[0:4])
        score = float(vals[14])
    elif len(vals) == 3:  # np. [x, y, score]
        x1, y1, score = map(float, vals)
        x2, y2 = x1 + 50, y1 + 50
    else:
        continue  # pomiń dziwne przypadki

    if score < 0.7:
        continue

    apply_blur_face(img, x1, y1, x2, y2)

# --- konwersja tensora na obraz i wyświetlenie ---
img_vis = K.tensor_to_image(img)

plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.show()