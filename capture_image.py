import cv2
import os
from basler_camera import BaslerCamera
from datetime import datetime

def capture_single_image(robot_type: str):
    # Create directory for images if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Initialize camera
    camera = BaslerCamera()
    if robot_type == "crs93":
        camera.connect_by_ip("192.168.137.107")
        camera.connect_by_name("camera-crs93")
    elif robot_type == "crs97":
        camera.connect_by_ip("192.168.137.106")
        camera.connect_by_name("camera-crs97")
    elif robot_type == "rv6s":
        camera.connect_by_ip("192.168.137.109")
        camera.connect_by_name("camera-rv6s")
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")
    camera.open()
    camera.set_parameters()

    try:
        # Capture image
        img = camera.grab_image()
        if img is None or img.size == 0:
            print("Failed to capture image")
            return

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/capture_{timestamp}.png"
        cv2.imwrite(filename, img)
        print(f"Image saved as: {filename}")

    finally:
        camera.close()

if __name__ == '__main__':
    robot_type = "rv6s"  # Change this to your desired robot type
    capture_single_image(robot_type)