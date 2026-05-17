import argparse
from Anonimizer import anonymize
from phoneDetection import detect_phones
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Anonymize image and detect phones.")
    parser.add_argument(
        "--file_path",
        type=str,
        default="phone5.jfif",
        help="Path to the input image file (default: phone5.jfif)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.6,
        help="Score threshold for anonymization (default: 0.6)"
    )
    parser.add_argument(
        "--anonimized_file",
        type=str,
        default="image_anonimized.jpg",
        help="Name/path of the output anonymized file (default: image_anonimized.jpg)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    anonimized_image = anonymize(args.file_path, args.score_threshold)
    cv2.imwrite(args.anonimized_file, anonimized_image)

    isPhonesDetected = detect_phones(args.anonimized_file)
    if isPhonesDetected:
        print("Phones detected!!!!!")
    else:
        print("No phones detected!!!")

if __name__ == "__main__":
    main()