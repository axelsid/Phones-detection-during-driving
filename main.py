import argparse
import logging
from Anonimizer import anonymize
from phoneDetection import detect_phones
import cv2
import piexif

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Anonymize image and detect phones.")
    parser.add_argument("--file_path", type=str, help="Path to the input image file")
    parser.add_argument("--score_threshold", type=float, help="Score threshold for anonymization")
    parser.add_argument("--anonimized_file", type=str, help="Name/path of the output anonymized file")
    return parser.parse_args()


def get_gps_coordinates(file_path):
    try:
        exif_data = piexif.load(file_path)
        gps_info = exif_data.get("GPS", {})

        if not gps_info:
            logger.info("No GPS data found in EXIF.")
            return

        def to_decimal(coord, ref):
            degrees, minutes, seconds = coord
            decimal = degrees[0] / degrees[1] + minutes[0] / (minutes[1] * 60) + seconds[0] / (seconds[1] * 3600)
            if ref in [b"S", b"W"]:
                decimal = -decimal
            return decimal

        lat = to_decimal(gps_info[piexif.GPSIFD.GPSLatitude], gps_info[piexif.GPSIFD.GPSLatitudeRef])
        lon = to_decimal(gps_info[piexif.GPSIFD.GPSLongitude], gps_info[piexif.GPSIFD.GPSLongitudeRef])

        return lat, lon

    except Exception as e:
        logger.error(f"Failed to extract EXIF GPS data: {e}")
        return


def main():
    args = parse_args()

    anonimized_image = anonymize(args.file_path, args.score_threshold)
    cv2.imwrite(args.anonimized_file, anonimized_image)

    isPhonesDetected = detect_phones(args.anonimized_file)
    if isPhonesDetected:
        logger.warning("Phones detected!")
        coordinates = get_gps_coordinates(args.file_path)
        if coordinates:
            lat, lon = coordinates
            logger.info(f"Photo coordinates — Latitude: {lat:.6f}, Longitude: {lon:.6f}")
        else:
            logger.info("No coordinates available for this photo.")
    else:
        logger.info("No phones detected.")


if __name__ == "__main__":
    main()