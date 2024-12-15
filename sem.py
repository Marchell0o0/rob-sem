from ctu_crs import CRS93 # or CRS97
import cv2
from basler_camera import BaslerCamera
import math
import os
from datetime import datetime
import numpy as np
import csv

class Board:
    def __init__(self, csv_file):
        self.aruco_ids = []
        self.hole_positions = []
        self.load_from_csv(csv_file)
        
        # Print loaded data
        print(f"Loaded ArUco IDs: {self.aruco_ids}")
        print(f"Loaded hole positions relative to marker {min(self.aruco_ids)}:")
        for i, pos in enumerate(self.hole_positions):
            print(f"Hole {i+1}: (x={pos[0]}, y={pos[1]}) mm")
        
        # ArUco dictionary for DICT_4X4_50
        self.aruco_dict = cv2.aruco.Dictionary_4X4_50
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        
    def load_from_csv(self, csv_file):
        """Load ArUco marker IDs and hole positions from CSV file."""
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            # First row contains ArUco marker IDs
            self.aruco_ids = [int(id_) for id_ in next(csv_reader)]
            # Remaining rows contain hole positions relative to the marker with smallest ID
            self.hole_positions = []
            for row in csv_reader:
                self.hole_positions.append((float(row[0]), float(row[1])))
                
    def detect_markers(self, image):
        """Detect ArUco markers in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        # Draw detected markers on the image
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            print(f"Detected ArUco markers with IDs: {ids.flatten()}")
        else:
            print("No ArUco markers detected")
            
        return image, corners, ids

def main():
    # Initialize robot
    robot = CRS93()
    robot.initialize() 

    # Move the first joint by 90 degrees (pi/2 radians)
    q = robot.get_q()
    robot.move_to_q(q + [math.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.wait_for_motion_stop()

    # Initialize board with CSV file
    board = Board('positions_plate_01-02.csv')

    # Initialize and setup camera
    camera = BaslerCamera()
    camera.connect_by_ip("192.168.137.107")
    camera.connect_by_name("camera-crs93")
    camera.open()
    camera.set_parameters()

    # Take image and process it
    img = camera.grab_image()
    if (img is not None) and (img.size > 0):
        # Create images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # Detect ArUco markers
        processed_img, corners, ids = board.detect_markers(img)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/capture_{timestamp}.png"
        
        # Save the processed image with detected markers
        cv2.imwrite(filename, processed_img)
        print(f"Image saved as: {filename}")
    else:
        print("The image was not captured.")

    # Clean up camera
    camera.close()

    # Clean up robot
    robot.soft_home()
    robot.close()

if __name__ == '__main__':
    main()