import cv2
import numpy as np
from basler_camera import BaslerCamera
import os
from datetime import datetime

def capture_calibration_images():
    # Create directory for calibration images if it doesn't exist
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')

    # Initialize camera
    camera = BaslerCamera()
    camera.connect_by_ip("192.168.137.107")
    camera.connect_by_name("camera-crs93")
    camera.open()
    camera.set_parameters()

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
            print(f"\nPress Enter to capture image {captured_frames + 1}/{required_frames} (or 'q' to quit)")
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('\r') or key == ord('\n'):  # Enter key
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"calibration_images/calib_{captured_frames + 1:02d}_{timestamp}.png"
                    cv2.imwrite(filename, img)
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
    print("This script will capture 15 images for camera calibration.")
    print("Position the calibration grid and press Enter for each capture.")
    print("Try to capture the grid from different angles and positions.")
    print("\nPress Enter to start capturing...")
    input()
    capture_calibration_images() 