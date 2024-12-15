# OpenCV library for image processing
import cv2
# Our Basler camera interface
from basler_camera import BaslerCamera
import os
from datetime import datetime
import numpy as np

def detect_aruco_markers(image_path):
    # Load camera calibration
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
        
    # Get optimal new camera matrix and undistort the image
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
    # Create ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers on undistorted image
    corners, ids, rejected = detector.detectMarkers(undistorted_img)
    
    if ids is not None:
        # Draw detected markers
        img_display = undistorted_img.copy()
        cv2.aruco.drawDetectedMarkers(img_display, corners, ids)
        
        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 35.0, new_camera_matrix, dist_coeffs)  # 35mm marker size
            
        # Draw axes for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(img_display, new_camera_matrix, dist_coeffs, 
                            rvecs[i], tvecs[i], 30)  # 30mm axis length
            
            # Print marker information
            marker_id = ids[i][0]
            position = tvecs[i][0]
            rotation = rvecs[i][0]
            print(f"\nMarker {marker_id}:")
            print(f"Position (mm): X={position[0]:.1f}, Y={position[1]:.1f}, Z={position[2]:.1f}")
            print(f"Rotation (rad): X={rotation[0]:.2f}, Y={rotation[1]:.2f}, Z={rotation[2]:.2f}")
            
        # Display both original and undistorted images
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', img)
        cv2.namedWindow('Undistorted Image with Markers', cv2.WINDOW_NORMAL)
        cv2.imshow('Undistorted Image with Markers', img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ArUco markers detected")

def main():
    # Original camera capture code
    camera: BaslerCamera = BaslerCamera()
 
    # Camera can be connected based on its' IP or name:
    # Camera for robot CRS 93
    camera.connect_by_ip("192.168.137.107")
    camera.connect_by_name("camera-crs93")
    # Camera for robot CRS 97
    #   camera.connect_by_ip("192.168.137.106")
    #   camera.connect_by_name("camera-crs97")
    # camera.connect_by_name("camera-crs97")
 
    # Open the communication with the camera
    camera.open()
    # Set capturing parameters from the camera object.
    # The default parameters (set by constructor) are OK.
    # When starting the params should be send into the camera.
    camera.set_parameters()
 
    # Take one image from the camera
    img = camera.grab_image()
    # If the returned image has zero size,
    # the image was not captured in time.
    if (img is not None) and (img.size > 0):
        # Create images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/capture_{timestamp}.png"
        
        # Save the image
        cv2.imwrite(filename, img)
        print(f"Image saved as: {filename}")
        
        # Process the captured image for ArUco markers
        detect_aruco_markers(filename)
    else:
        print("The image was not captured.")
 
    # Close communication with the camera before finish.
    camera.close()
 
if __name__ == '__main__':
    main()