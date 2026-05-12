from Anonimizer import anonymize
from phoneDetection import detect_phones
import cv2

file_path = "man-sitting-in-his-car-texting-with-his-cell-phone-side-view-of-a-young-man-sitting-inside-car-using-mobile-phone-young-man-sitting-in-a-car-and-looking-at-a-mobile-phone-photo.jpg"
score_threshold = 0.6

anonimized_image = anonymize(file_path, score_threshold)
cv2.imwrite("image_anonimized.jpg", anonimized_image)
isPhonesDetected = detect_phones("image_anonimized.jpg")
if isPhonesDetected:
    print("Phones detected!!!!!")
else:
    print("No phones detected!!!")


