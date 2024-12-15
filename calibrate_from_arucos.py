import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def calibrate_camera():
    # Configuration
    CHECKERBOARD = (20, 13)  # internal corners (height, width) - 13 rows × 20 columns
    SQUARE_SIZE = 20.0  # mm
    
    # Get calibration images
    images = sorted(Path('calibration_images').glob('*.png'))  # Process all images
    
    # Create 3D points pattern
    corners3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # Create grid in row-major order (left-to-right, top-to-bottom)
    idx = 0
    for y in range(CHECKERBOARD[1]):  # rows (13)
        for x in range(CHECKERBOARD[0]):  # columns (20)
            # Start from max x and decrease to 0 to match physical board layout (origin at top-left)
            corners3d[idx] = [x * SQUARE_SIZE, y * SQUARE_SIZE, 0]  # x goes from 380mm to 0, y goes from 0 to 240mm
            idx += 1
    
    # Lists to store points
    pts2d = []
    pts3d = []
    
    # Process each image
    for i, fname in enumerate(images):
        print(f"\nProcessing image {i}: {fname}")
        
        # Read and convert image
        img = cv2.imread(str(fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        success, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if success:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store points
            pts2d.append(corners)
            pts3d.append(corners3d)
            
            # Draw corners and coordinates
            img_display = img.copy()
            cv2.drawChessboardCorners(img_display, CHECKERBOARD, corners, success)
            
            # Add text for each corner
            for j, (corner, point3d) in enumerate(zip(corners, corners3d)):
                x, y = corner.ravel()
                # Draw a small circle at the corner
                cv2.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # Format coordinates more compactly
                text_2d = f"2D:({int(x)},{int(y)})"
                text_3d = f"3D:({int(point3d[0])},{int(point3d[1])})"
                text_id = f"ID: {j}"
                
                # Print to console for verification
                # print(f"Corner {j}: 3D({point3d[0]:.1f},{point3d[1]:.1f},{point3d[2]:.1f}) -> 2D({int(x)},{int(y)})")
                
                # Place coordinates with better spacing
                cv2.putText(img_display, text_2d, (int(x)-40, int(y)-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.putText(img_display, text_3d, (int(x)-40, int(y)+10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.putText(img_display, text_id, (int(x)-40, int(y)+20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Display the image
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
            plt.title(f'Calibration Image {i} with Corner Coordinates')
            plt.axis('off')
            plt.show()
            
            # Calculate pattern metrics
            pattern_width_px = np.linalg.norm(corners[CHECKERBOARD[1]-1] - corners[0])
            pattern_height_px = np.linalg.norm(corners[-1] - corners[0])
            perspective_ratio = (pattern_height_px/CHECKERBOARD[0]) / (pattern_width_px/CHECKERBOARD[1])
            
            print(f"Perspective ratio (height/width): {perspective_ratio:.2f}")
            if perspective_ratio > 1.2 or perspective_ratio < 0.8:
                print("WARNING: Large perspective distortion detected!")
        else:
            print(f"No chessboard found in image {i}")
    
    if not pts2d:
        print("No chessboard patterns found!")
        return None
    
    # Calibrate camera
    print(f"\nCalibrating with {len(pts2d)} images...")
    h, w = gray.shape
    flags = cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_PRINCIPAL_POINT
    
    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        pts3d, pts2d, (w, h), None, None, flags=flags
    )
    
    # Print results
    print(f"\nRMS re-projection error: {err} pixels")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients:\n{dist}")
    
    # Calculate per-image errors
    for i in range(len(pts2d)):
        imgpoints2, _ = cv2.projectPoints(pts3d[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(pts2d[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        print(f"Image {i} error: {error} pixels")
    
    # Save calibration results
    np.save('camera_matrix.npy', K)
    np.save('dist_coeffs.npy', dist)
    print("\nCalibration results saved to camera_matrix.npy and dist_coeffs.npy")
    
    return err, K, dist, rvecs, tvecs

if __name__ == '__main__':
    calibrate_camera()