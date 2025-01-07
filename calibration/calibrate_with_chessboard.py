import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from src.scene3d import Scene3D
from src.se3 import SE3
from src.so3 import SO3


def calibrate_camera(
    images_path: str,
    board_height: int,
    board_width: int,
    square_size: float,
    marker_size: float,
    output_dir: str = ".",
    show_images: bool = False
):
    """
    Calibrate camera using ChArUco board images.

    Args:
        images_path (str): Path to directory containing calibration images
        board_height (int): Number of squares in height
        board_width (int): Number of squares in width
        square_size (float): Size of each square in mm
        marker_size (float): Size of ArUco marker in mm
        output_dir (str): Directory to save calibration results
        show_images (bool): Whether to display processed images
    """
    print(f"Calibrating camera with ChArUco board of size {board_width}x{board_height} with square size {square_size}mm and marker size {marker_size}mm")
    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((board_width, board_height), square_size, marker_size, aruco_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Get calibration images
    images = sorted(Path(images_path).glob('*.png'))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {images_path}")

    # Lists to store points
    all_corners = []
    all_ids = []
    img_size = None

    # Process each image
    for i, fname in enumerate(images):
        print(f"Processing image {i + 1}: {fname}")

        img = cv2.imread(str(fname))
        if img is None:
            continue
        
        if img_size is None:
            img_size = img.shape[:2]
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco markers
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        print(f"Detected {len(marker_corners)} markers")
        if marker_ids is not None:
            # Refine and interpolate corners
            response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board)
            # print(f"Interpolated {len(charuco_corners)} corners")
            print(f"Response: {response}")
            if response > 20:  # Minimum number of corners
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

                if show_images:
                    img_display = img.copy()
                    # Draw detected markers
                    cv2.aruco.drawDetectedMarkers(img_display, marker_corners, marker_ids)
                    # Draw ChArUco corners
                    cv2.aruco.drawDetectedCornersCharuco(img_display, charuco_corners, charuco_ids)

                    plt.figure(figsize=(15, 10))
                    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
                    plt.title(f'Calibration Image {i} with ChArUco Corners')
                    plt.axis('off')
                    plt.show()
        else:
            print(f"No ChArUco markers found in image {i}")

    if not all_corners:
        print("No ChArUco patterns found!")
        return None

    print(f"\nCalibrating with {len(all_corners)} images...")
    
    flags = cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_FIX_PRINCIPAL_POINT

    # Calibrate camera using ChArUco
    err, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None, flags=flags
    )

    print(f"\nRMS re-projection error: {err} pixels")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients:\n{dist}")

    # Save calibration results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    np.save(output_path / 'camera_matrix.npy', K)
    np.save(output_path / 'dist_coeffs.npy', dist)
    print(f"\nCalibration results saved to {output_path}")

    # Create and display calibration visualization
    scene = Scene3D().invert_z_axis()
    scene.add_transform("Camera", SE3())

    # Add each board
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Transform board corners to camera frame
        R, _ = cv2.Rodrigues(rvec)
        
        # Get board corners in 3D
        board_corners = board.getChessboardCorners()
        pts_transformed = (R @ board_corners.T + tvec).T

        # Add board and its frame
        scene.add_calibration_board(pts_transformed)
        scene.add_transform(f"Board_{i}", SE3(
            translation=pts_transformed.mean(axis=0),
            rotation=SO3(R)
        ))

    scene.display()

    return err, K, dist, rvecs, tvecs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera calibration from ChArUco board images')
    parser.add_argument('--images', type=str, default='calibration/calibration_images',
                        help='Path to directory containing calibration images')
    parser.add_argument('--board-height', type=int, required=True,
                        help='Number of squares in height')
    parser.add_argument('--board-width', type=int, required=True,
                        help='Number of squares in width')
    parser.add_argument('--square-size', type=float, required=True,
                        help='Size of each square in mm')
    parser.add_argument('--marker-size', type=float, required=True,
                        help='Size of ArUco marker in mm')
    parser.add_argument('--output-dir', type=str, default='calibration/calibration_data',
                        help='Directory to save calibration results')
    parser.add_argument('--show-images', action='store_true',
                        help='Display processed images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    calibrate_camera(
        images_path=args.images,
        board_height=args.board_height,
        board_width=args.board_width,
        square_size=args.square_size,
        marker_size=args.marker_size,
        output_dir=args.output_dir,
        show_images=args.show_images
    )
