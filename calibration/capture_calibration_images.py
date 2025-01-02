import cv2
import numpy as np
import os
from datetime import datetime
import argparse
from src.camera import Camera
from src.enums import RobotType


def capture_calibration_images(robot_type: RobotType):
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')

    # Initialize camera
    camera = Camera(robot_type)
    try:
        captured_frames = 0
        required_frames = 15

        while captured_frames < required_frames:
            # Capture image
            img = camera.grab_image()
            if img is None or img.size == 0:
                print("Failed to capture image")
                continue

            # Display image
            cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
            cv2.imshow('Capture', img)

            # Wait for Enter key
            print(
                f"\nPress Enter to capture image {captured_frames + 1}/{required_frames} (or 'q' to quit)")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('\r') or key == ord('\n'):  # Enter key
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"calibration_images/calib_{captured_frames + 1:02d}_{timestamp}.png"
                    Camera.save_image(img, filename)
                    print(f"Saved image as: {filename}")
                    captured_frames += 1
                    break
                elif key == ord('q'):
                    print("Capture cancelled by user")
                    return

            if captured_frames == required_frames:
                print("\nAll calibration images captured!")
                print("Images are saved in the 'calibration_images' directory")

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-type", type=str,
                        default="RV6S", help="Type of the robot")
    args = parser.parse_args()
    print("This script will capture 15 images for camera calibration.")
    print("Position the calibration grid and press Enter for each capture.")
    print("Try to capture the grid from different angles and positions.")
    print("\nPress Enter to start capturing...")
    input()
    capture_calibration_images(RobotType[args.robot_type])
