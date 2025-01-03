import cv2
import numpy as np
import os
from datetime import datetime
import argparse
from src.camera import Camera
from src.enums import RobotType
from src.camera_image import CameraImage


def capture_calibration_images(robot_type: RobotType):
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists('calibration/calibration_images'):
        os.makedirs('calibration/calibration_images')

    # Initialize camera
    camera = Camera(robot_type)
    try:
        captured_frames = 0
        required_frames = 15
        quit_program = False

        while captured_frames < required_frames and not quit_program:
            # Capture image
            img = camera.grab_image()
            if img.image is None:
                print("Failed to capture image")
                continue

            # Display image
            img.set_display_size(720)
            print(f"\nPress Enter to capture image {captured_frames + 1}/{required_frames} (or 'q' to quit)")
            
            while True:
                img.display(block=False)  # Non-blocking display
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('\r') or key == ord('\n'):  # Enter key
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"calibration_images/calib_{captured_frames + 1:02d}_{timestamp}.png"
                    img.save_image(filename)
                    print(f"Saved image as: {filename}")
                    captured_frames += 1
                    cv2.destroyWindow("Image Scene")
                    break
                elif key == ord('q'):
                    print("Capture cancelled by user")
                    quit_program = True
                    cv2.destroyWindow("Image Scene")
                    break

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
