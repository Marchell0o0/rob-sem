import cv2
import numpy as np
import csv
import os

class Board:
    def __init__(self, aruco_ids=None):
        self.aruco_ids = aruco_ids
        self.hole_positions = []
        
        # ArUco dictionary for DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
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
        
        print(f"Loaded board with ArUco IDs: {self.aruco_ids}")
        print(f"Loaded hole positions relative to marker {min(self.aruco_ids)}:")
        for i, pos in enumerate(self.hole_positions):
            print(f"Hole {i+1}: (x={pos[0]}, y={pos[1]}) mm")
                
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

def find_board_files(marker_ids):
    """Find all board files that match any of the detected marker IDs."""
    if marker_ids is None or len(marker_ids) == 0:
        return []
        
    # Convert marker_ids from numpy array to list of ints
    marker_ids = marker_ids.flatten().tolist()
    matching_files = []
    
    # Check each board file
    board_files = [f for f in os.listdir('boards') if f.startswith('positions_plate_') and f.endswith('.csv')]
    
    for board_file in board_files:
        with open(os.path.join('boards', board_file), 'r') as f:
            first_line = f.readline().strip()
            file_ids = [int(id_) for id_ in first_line.split(',')]
            
            # Check if any of the detected markers match this board's markers
            if any(marker_id in file_ids for marker_id in marker_ids):
                matching_files.append(os.path.join('boards', board_file))
    
    return matching_files

def main():
    # Create a board instance without loading CSV yet
    board = Board()

    # Read the saved image
    img = cv2.imread('images/capture_20241215_151731.png')
    if img is None:
        print("Error: Could not read the image file")
        return

    # First detect markers in the image
    processed_img, corners, ids = board.detect_markers(img)
    
    # Find and load all matching board files
    if ids is not None:
        board_files = find_board_files(ids)
        if board_files:
            print(f"\nFound {len(board_files)} matching board configurations:")
            boards = []
            for board_file in board_files:
                print(f"\nLoading board configuration from: {board_file}")
                board = Board()
                board.load_from_csv(board_file)
                boards.append(board)
        else:
            print("No matching board configuration found for detected markers")
    
    # Display the image
    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 