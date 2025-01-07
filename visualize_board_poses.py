import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.se3 import SE3
from src.so3 import SO3

def create_custom_board(ids, marker_size=36):
    """Create a custom board with two ArUco markers and board edges."""
    
    # Define marker positions (in mm)
    marker_positions = [
        (0, 0),           # First marker (ID 3) at origin
        (140, 180)        # Second marker (ID 4) offset diagonally
    ]
    
    # Define board corners (30mm padding from markers)
    board_corners = np.array([
        [210, -30, 0],    # bottom-right
        [210, 170, 0],    # top-right
        [-30, 170, 0],    # top-left
        [-30, -30, 0],    # bottom-left
    ], dtype=np.float32)
    
    # Define marker corners for each marker relative to their positions
    all_corners = []
    all_ids = []
    
    for i, (mx, my) in enumerate(marker_positions):
        corners = np.array([
            [mx - marker_size/2, my + marker_size/2, 0],  # top-left
            [mx + marker_size/2, my + marker_size/2, 0],  # top-right
            [mx + marker_size/2, my - marker_size/2, 0],  # bottom-right
            [mx - marker_size/2, my - marker_size/2, 0],  # bottom-left
        ], dtype=np.float32)
        
        all_corners.append(corners)
    
    all_ids = np.sort(ids.flatten())
    
    return np.array(all_corners), np.array(all_ids), board_corners

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort an image using camera parameters."""
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)

def detect_board_edges(image, corners):
    """Detect board edges using contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Project marker corners to image coordinates to get ROI
    all_corners = np.vstack([corner[0] for corner in corners])
    min_xy = np.min(all_corners, axis=0).astype(int) - 600
    max_xy = np.max(all_corners, axis=0).astype(int) + 600
    
    # Ensure bounds are within image
    min_xy = np.maximum(min_xy, [0, 0])
    max_xy = np.minimum(max_xy, [gray.shape[1], gray.shape[0]])
    
    # Extract ROI
    roi = gray[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]
    
    # Preprocessing
    denoised = cv2.fastNlMeansDenoising(roi, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 70)
    
    # Connect edges
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    small_kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    
    # Erode to thin out the contours
    eroded = cv2.erode(closed, small_kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug visualization
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    # Draw all contours in red
    cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]
    board_contour = max(valid_contours, key=cv2.contourArea)
    
    # Draw largest contour in green
    cv2.drawContours(debug_img, [board_contour], -1, (0, 255, 0), 3)
    
    # Approximate polygon
    epsilon = 0.03 * cv2.arcLength(board_contour, True)
    approx = cv2.approxPolyDP(board_contour, epsilon, True)
    
    # Draw approximated polygon corners in blue
    for point in approx:
        cv2.circle(debug_img, tuple(point[0]), 5, (255, 0, 0), -1)
    
    # # Show debug images - undistort just before showing
    # edges_undist = undistort_image(edges, camera_matrix, dist_coeffs)
    # closed_undist = undistort_image(closed, camera_matrix, dist_coeffs)
    # debug_undist = undistort_image(debug_img, camera_matrix, dist_coeffs)
    
    cv2.imshow("Edges", edges)
    cv2.imshow("Dilated", dilated)
    cv2.imshow("Closed", closed)
    cv2.imshow("Eroded", eroded)
    cv2.waitKey(1)
    
    # Get corners and adjust for ROI offset
    corners = approx.reshape(-1, 2) + min_xy
    return corners

def estimate_board_pose(img, camera_matrix):
    """Detect markers and estimate board pose."""
    # Read and undistort image

    
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
    
    # Get board definition including board corners
    board_corners, board_ids, board_edge_points = create_custom_board(ids)
    
    # Detect board edges
    detected_edges = detect_board_edges(img, corners)
    
    # Create objPoints and imgPoints arrays for solvePnP
    objPoints = []
    imgPoints = []
    
    # Draw results
    img_markers = img.copy()
    cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)
    
    # Add ArUco corners
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in board_ids:
            board_idx = np.where(board_ids == marker_id)[0][0]
            marker_corners_board = board_corners[board_idx]
            marker_corners_img = corners[i][0]
            
            objPoints.extend(marker_corners_board)
            imgPoints.extend(marker_corners_img)
            
            # Add coordinate labels for each corner
            for j in range(4):
                coord_text = f"({marker_corners_board[j][0]:.0f}, {marker_corners_board[j][1]:.0f})"
                text_pos = (int(marker_corners_img[j][0]) + 10, int(marker_corners_img[j][1]) + 10)
                
                # Draw point and text
                cv2.circle(img_markers, (int(marker_corners_img[j][0]), int(marker_corners_img[j][1])), 5, (0, 255, 0), -1)
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img_markers, 
                            (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                            (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                            (255, 255, 255), -1)
                cv2.putText(img_markers, coord_text, text_pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add board edge points if detected
    if detected_edges is not None:
        # Draw lines between corners
        for i in range(4):
            pt1 = tuple(detected_edges[i].astype(int))
            pt2 = tuple(detected_edges[(i + 1) % 4].astype(int))
            cv2.line(img_markers, pt1, pt2, (0, 0, 255), 2)  # Red lines
            
        # Draw and label corners
        for i, (edge_point_3d, edge_point_2d) in enumerate(zip(board_edge_points, detected_edges)):
            # Draw corner point
            cv2.circle(img_markers, tuple(edge_point_2d.astype(int)), 5, (255, 0, 0), -1)  # Blue dots
            
            # Add coordinate labels for edge points
            coord_text = f"({edge_point_3d[0]:.0f}, {edge_point_3d[1]:.0f})"
            text_pos = (int(edge_point_2d[0]) + 10, int(edge_point_2d[1]) + 10)
            
            # Draw text with background
            text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_markers, 
                        (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                        (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                        (255, 255, 255), -1)
            cv2.putText(img_markers, coord_text, text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    if not objPoints:
        print("No valid board markers found!")
        return None, img
    
    # Convert to numpy arrays
    objPoints = np.array(objPoints, dtype=np.float32)
    imgPoints = np.array(imgPoints, dtype=np.float32)
    
    # Estimate pose
    success, rvec, tvec = cv2.solvePnP(
        objPoints, imgPoints, camera_matrix, distCoeffs = np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("Could not estimate board pose!")
        return None, img
    
    # Draw board axes
    cv2.drawFrameAxes(img_markers, camera_matrix, np.zeros(5), rvec, tvec, 100)
    
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
    count = 0
    for image_path in images:
        count += 1
        if count == 1:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        
        undistorted = undistort_image(img, camera_matrix, dist_coeffs)
        print(f"\nProcessing {image_path.name}")

        result, img_with_axes = estimate_board_pose(undistorted, camera_matrix)
        
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