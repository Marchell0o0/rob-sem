import cv2
import numpy as np
from pathlib import Path
from src.scene3d import Scene3D
from src.se3 import SE3
from src.so3 import SO3

def load_camera_params(calib_dir: str = "calibration/calibration_data"):
    """Load camera matrix and distortion coefficients from calibration files."""
    camera_matrix = np.load(Path(calib_dir) / "camera_matrix.npy")
    dist_coeffs = np.load(Path(calib_dir) / "dist_coeffs.npy")
    return camera_matrix, dist_coeffs

def read_image(image_path: str) -> np.ndarray:
    """Read an image from the given path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    return img

def detect_aruco_corners(image: np.ndarray, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """Detect ArUco markers in the image and return their corners and IDs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(image)
    return corners, ids

def estimate_board_pose(corners: list, ids: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                       marker_size: float = 0.036):  # marker size in mm
    """Estimate board pose using solvePnP with four corners of a marker."""
    if not corners or len(corners) == 0:
        return [], []
    
    rvecs = []
    tvecs = []
    
    # Define 3D points of the marker in marker coordinate system
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ])
    
    # Process first two markers
    for marker_corners in corners[:2]:  # Only take first two markers
        # Reshape image points
        img_points = marker_corners.reshape(-1, 2)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)
    
    return rvecs, tvecs

def draw_pose_axes(image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                   camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                   axis_length: float = 0.1) -> np.ndarray:
    """Draw coordinate axes on the image to visualize the estimated pose."""
    img_copy = image.copy()
    cv2.drawFrameAxes(img_copy, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    return img_copy

def draw_aruco_borders(image: np.ndarray, corners: list, ids: np.ndarray) -> np.ndarray:
    """Draw borders around detected ArUco markers."""
    img_copy = image.copy()
    if corners and ids is not None:
        cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)
    return img_copy

def estimate_pose_from_both_markers(corners: list, ids: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                              marker_size: float = 0.036):  # marker size in meters
    """Estimate pose using all 8 points from both markers, considering their relative positions."""
    if not corners or len(corners) < 2 or ids is None:
        return None, None

    # Sort corners by marker ID to ensure base marker (lower ID) is first
    sorted_indices = np.argsort(ids.flatten())
    corners = [corners[i] for i in sorted_indices]
    ids = ids[sorted_indices]

    # Define 3D points for both markers in the base coordinate system
    # First marker (base) at origin
    base_points = np.array([
        [-marker_size/2, marker_size/2, 0],  # top-left
        [marker_size/2, marker_size/2, 0],   # top-right
        [marker_size/2, -marker_size/2, 0],  # bottom-right
        [-marker_size/2, -marker_size/2, 0]  # bottom-left
    ])

    # Second marker offset by [0.180, 0.140] meters
    second_marker_points = base_points + np.array([0.180, 0.140, 0])

    # Combine all object points
    obj_points = np.vstack([base_points, second_marker_points])

    # Combine all image points
    img_points = np.vstack([
        corners[0].reshape(-1, 2),  # First marker's corners
        corners[1].reshape(-1, 2)   # Second marker's corners
    ])

    # Solve PnP with all 8 points
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

    if success:
        return rvec, tvec
    return None, None

def rodrigues_to_se3(rvec: np.ndarray, tvec: np.ndarray) -> SE3:
    """Convert rotation vector and translation vector to SE3 transform."""
    R, _ = cv2.Rodrigues(rvec)
    # Convert tvec from meters to mm for Scene3D
    tvec_mm = tvec * 1000
    return SE3(translation=tvec_mm.flatten(), rotation=SO3(R))

def calculate_orthogonality(transform_base: SE3, transform_second: SE3) -> float:
    """Calculate angle between base Z axis and vector from base to second transform."""
    # Get Z axis of base transform
    z_base = transform_base.rotation.rot[:, 2]
    
    # Calculate delta vector (from base to second aruco)
    delta = transform_second.translation - transform_base.translation
    delta_normalized = delta / np.linalg.norm(delta)
    
    # Calculate dot product
    dot_product = np.abs(np.dot(z_base, delta_normalized))
    # Return angle in degrees
    return np.arccos(dot_product) * 180 / np.pi

if __name__ == "__main__":
    # Load camera parameters once
    camera_matrix, dist_coeffs = load_camera_params()
    
    # Get all PNG images from boards_dataset
    dataset_path = Path("boards_dataset")
    image_files = sorted(dataset_path.glob("*.png"))
    
    for image_path in image_files:
        print(f"\nProcessing {image_path.name}")
        
        # Create new 3D scene for each image
        scene = Scene3D().invert_z_axis().z_from_zero()
        
        # Read and process image
        image = read_image(str(image_path))
        
        # Detect markers
        corners, ids = detect_aruco_corners(image)
        
        if corners and ids is not None and len(corners) >= 2:
            # Get individual ArUco poses
            rvecs, tvecs = estimate_board_pose(corners, ids, camera_matrix, dist_coeffs)
            
            # Get combined pose from all 8 points
            rvec_combined, tvec_combined = estimate_pose_from_both_markers(corners, ids, camera_matrix, dist_coeffs)
            
            # Add camera frame
            scene.add_transform("Camera", SE3())
            
            # Add individual ArUco transforms and store them with their IDs
            aruco_transforms = {}
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                transform = rodrigues_to_se3(rvec, tvec)
                marker_id = ids[i][0]
                scene.add_transform(f"ArUco_{marker_id}", transform)
                aruco_transforms[marker_id] = transform
            
            # Add combined transform
            if rvec_combined is not None and tvec_combined is not None:
                transform_combined = rodrigues_to_se3(rvec_combined, tvec_combined)
                scene.add_transform("Combined_Base", transform_combined)
                
                # Get the ArUco with higher ID (second ArUco)
                second_aruco_id = max(aruco_transforms.keys())
                first_aruco_id = min(aruco_transforms.keys())
                
                # Calculate and visualize delta vector from base to second ArUco
                delta_base = aruco_transforms[second_aruco_id].translation - transform_combined.translation
                delta_point_base = transform_combined.translation + delta_base
                scene.add_point("Delta_Base_End", delta_point_base, color='r')
                scene.add_connection("Combined_Base", "Delta_Base_End")
                
                # Calculate and visualize delta vector from first to second ArUco
                delta_aruco = aruco_transforms[second_aruco_id].translation - aruco_transforms[first_aruco_id].translation
                delta_point_aruco = aruco_transforms[first_aruco_id].translation + delta_aruco
                scene.add_point("Delta_Aruco_End", delta_point_aruco, color='b')
                scene.add_connection(f"ArUco_{first_aruco_id}", "Delta_Aruco_End")
                
                # Calculate and print orthogonality between base Z and base-to-second delta
                angle_base = calculate_orthogonality(transform_combined, aruco_transforms[second_aruco_id])
                print(f"\nDelta vector base to ArUco_{second_aruco_id} (mm):", delta_base)
                print(f"Angle between Combined_Base Z axis and delta vector: {angle_base:.2f} degrees")
                print(f"Deviation from orthogonality: {abs(90 - angle_base):.2f} degrees")
                
                # Calculate and print orthogonality between first ArUco Z and first-to-second delta
                angle_aruco = calculate_orthogonality(aruco_transforms[first_aruco_id], aruco_transforms[second_aruco_id])
                print(f"\nDelta vector ArUco_{first_aruco_id} to ArUco_{second_aruco_id} (mm):", delta_aruco)
                print(f"Angle between ArUco_{first_aruco_id} Z axis and delta vector: {angle_aruco:.2f} degrees")
                print(f"Deviation from orthogonality: {abs(90 - angle_aruco):.2f} degrees")
            
            # Display 3D scene
            scene.set_axis_length(50)  # Set axis length to 50mm
            scene.display()

            # Display 2D visualization
            result_image = image.copy()
            
            # Draw individual ArUco poses in blue (smaller axes)
            for rvec, tvec in zip(rvecs, tvecs):
                result_image = draw_pose_axes(result_image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.05)
            
            # Draw combined pose in default colors (larger axes)
            if rvec_combined is not None and tvec_combined is not None:
                result_image = draw_pose_axes(result_image, rvec_combined, tvec_combined, camera_matrix, dist_coeffs, axis_length=0.1)
            
            # Draw ArUco borders
            result_image = draw_aruco_borders(result_image, corners, ids)
            
            # Create window with fixed size
            window_name = f"Board Detection - {image_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 900)
            cv2.imshow(window_name, result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Need at least 2 ArUco markers in the image")