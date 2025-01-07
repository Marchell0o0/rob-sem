import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.se3 import SE3
from src.so3 import SO3

def create_custom_board(board_width=200, board_height=240, marker_size=36):
    """Create a custom board with two ArUco markers on diagonal positions."""
    
    # Define marker positions (in mm)
    marker_positions = [
        (0, 0),           # First marker (ID 3) at origin
        (140, 180)        # Second marker (ID 4) offset diagonally
    ]
    
    # Define marker corners for each marker relative to their positions
    all_corners = []
    all_ids = []
    
    for i, (mx, my) in enumerate(marker_positions):
        # Define corners for this marker
        corners = np.array([
            [mx - marker_size/2, my + marker_size/2, 0],  # top-left
            [mx + marker_size/2, my + marker_size/2, 0],  # top-right
            [mx + marker_size/2, my - marker_size/2, 0],  # bottom-right
            [mx - marker_size/2, my - marker_size/2, 0]   # bottom-left
        ], dtype=np.float32)
        
        all_corners.append(corners)
        all_ids.append(i + 3)  # IDs will be 3 and 4
    
    return np.array(all_corners), np.array(all_ids)

def estimate_board_pose(image_path, camera_matrix, dist_coeffs):
    """Detect markers and estimate board pose."""
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)   
    
    if ids is None:
        print("No markers detected!")
        return None, img
    
    # Get board definition
    board_corners, board_ids = create_custom_board()
    
    # Create objPoints and imgPoints arrays for solvePnP
    objPoints = []
    imgPoints = []
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in board_ids:
            board_idx = np.where(board_ids == marker_id)[0][0]
            objPoints.extend(board_corners[board_idx])
            imgPoints.extend(corners[i][0])
    
    if not objPoints:
        print("No valid board markers found!")
        return None, img
    
    # Convert to numpy arrays
    objPoints = np.array(objPoints, dtype=np.float32)
    imgPoints = np.array(imgPoints, dtype=np.float32)
    
    # Estimate pose
    success, rvec, tvec = cv2.solvePnP(
        objPoints, imgPoints, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("Could not estimate board pose!")
        return None, img
    
    # Draw results
    img_markers = img.copy()
    cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)
    
    # Draw board axes
    cv2.drawFrameAxes(img_markers, camera_matrix, dist_coeffs, rvec, tvec, 100)
    
    return (rvec, tvec), img_markers

def main():
    # Load camera calibration
    camera_matrix = np.load("calibration/calibration_data/camera_matrix.npy")
    dist_coeffs = np.load("calibration/calibration_data/dist_coeffs.npy")
    
    # Get all images from the dataset directory
    dataset_path = Path("images/boards_dataset")
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = sorted([f for f in dataset_path.iterdir() if f.suffix.lower() in image_extensions])
    
    if not images:
        print(f"No images found in {dataset_path}")
        return
        
    print(f"Found {len(images)} images")
    
    for image_path in images:
        print(f"\nProcessing {image_path.name}")
        result, img_with_axes = estimate_board_pose(image_path, camera_matrix, dist_coeffs)
        
        if result is not None:
            rvec, tvec = result
            print("Board rotation vector:", rvec.flatten())
            print("Board translation vector:", tvec.flatten())
            
            # Convert rotation vector to matrix for SE3
            R, _ = cv2.Rodrigues(rvec)
            board_pose = SE3(
                translation=tvec.flatten(),
                rotation=SO3(rotation_matrix=R)
            )
            print("Board pose:", board_pose)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_axes, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Board with Coordinate Axes - {image_path.name}")
        plt.axis('off')
        
        # Show the plot non-blocking
        plt.show(block=True)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Close current figure
        plt.close()
        
        # If 'q' is pressed, exit
        if key == ord('q'):
            print("\nExiting on user request")
            break
            
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()