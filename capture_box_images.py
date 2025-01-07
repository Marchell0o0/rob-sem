import cv2
import os
from datetime import datetime
import argparse
from src.camera import Camera
from src.enums import RobotType

def capture_boards_images(robot_type: RobotType):
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists('boards_images_classified'):
        os.makedirs('boards_images_classified')

    captured_frames = 0
    required_frames = 120

    # Initialize camera
    camera = Camera(robot_type)
    classes = [
        "flat_low_empty",
        "flat_low_cubes",
        "flat_high_empty",
        "flat_high_cubes",
        "rotated_7_empty",
        "rotated_7_cubes",
        "rotated_15_empty",
        "rotated_15_cubes"
    ]
    
    try:
        while captured_frames < required_frames:
            # Capture image
            img = camera.grab_image()
            if img is None or img.image.size == 0:
                print("Failed to capture image")
                continue

            # Display the image for feedback
            # cv2.imshow('Capture', img.image)

            # Wait for user to press Enter to capture the image
            key = cv2.waitKey(0)  # Wait indefinitely for a key press
            print(f"Press Enter to capture image {captured_frames + 1}/{required_frames} (or 'q' to quit)")

            if key == ord('\r') or key == ord('\n'):  # Enter key
                # Save the image with a timestamp and class name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"boards_images_classified/board_{captured_frames + 1}_{timestamp}.png"
                cv2.imwrite(filename, img.image)
                print(f"Saved image as: {filename}")
                captured_frames += 1
            elif key == ord('q'):
                print("Capture cancelled by user")
                return

        print("All board images captured!")
        print("Images are saved in the 'boards_images_classified' directory")

    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-type", type=str, default="RV6S", help="Type of the robot")
    args = parser.parse_args()
    
    print("This script will capture 15 images for camera calibration.")
    print("Position the calibration grid and press Enter for each capture.")
    print("Try to capture the grid from different angles and positions.")
    print("\nPress Enter to start capturing...")
    
    input()  # Wait for user to press Enter before starting the capture
    capture_boards_images(RobotType[args.robot_type])
