import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from perception_utils import viz_calibration, viz_calibration_boards


def calibrate_camera(
    images_path: str,
    board_height: int,
    board_width: int,
    square_size: float,
    output_dir: str = ".",
    show_images: bool = False
):
    """
    Calibrate camera using chessboard images.

    Args:
        images_path (str): Path to directory containing calibration images
        board_height (int): Number of internal corners in height
        board_width (int): Number of internal corners in width
        square_size (float): Size of each square in mm
        output_dir (str): Directory to save calibration results
        show_images (bool): Whether to display processed images
    """
    # Configuration
    CHECKERBOARD = (board_height, board_width)
    SQUARE_SIZE = square_size

    # Get calibration images
    images = sorted(Path(images_path).glob('*.png'))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {images_path}")

    # Create 3D points pattern
    corners3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    idx = 0
    for y in range(CHECKERBOARD[1]):
        for x in range(CHECKERBOARD[0]):
            corners3d[idx] = [x * SQUARE_SIZE,
                              (CHECKERBOARD[1] - y - 1) * SQUARE_SIZE, 0]
            idx += 1

    # Lists to store points
    pts2d = []
    pts3d = []

    # Process each image
    for i, fname in enumerate(images):
        print(f"\nProcessing image {i}: {fname}")

        img = cv2.imread(str(fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        success, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if success:
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            pts2d.append(corners)
            pts3d.append(corners3d)

            if show_images:
                img_display = img.copy()
                cv2.drawChessboardCorners(
                    img_display, CHECKERBOARD, corners, success)

                for j, (corner, point3d) in enumerate(zip(corners, corners3d)):
                    x, y = corner.ravel()
                    cv2.circle(img_display, (int(x), int(y)),
                               3, (0, 255, 0), -1)

                    text_2d = f"2D:({int(x)},{int(y)})"
                    text_3d = f"3D:({int(point3d[0])},{int(point3d[1])})"
                    text_id = f"ID: {j}"

                    cv2.putText(img_display, text_2d, (int(x)-40, int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    cv2.putText(img_display, text_3d, (int(x)-40, int(y)+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    cv2.putText(img_display, text_id, (int(x)-40, int(y)+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                plt.figure(figsize=(15, 10))
                plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
                plt.title(f'Calibration Image {i} with Corner Coordinates')
                plt.axis('off')
                plt.show()

            pattern_width_px = np.linalg.norm(
                corners[CHECKERBOARD[1]-1] - corners[0])
            pattern_height_px = np.linalg.norm(corners[-1] - corners[0])
            perspective_ratio = (
                pattern_height_px/CHECKERBOARD[0]) / (pattern_width_px/CHECKERBOARD[1])

            print(f"Perspective ratio (height/width): {perspective_ratio:.2f}")
            if perspective_ratio > 1.2 or perspective_ratio < 0.8:
                print("WARNING: Large perspective distortion detected!")
        else:
            print(f"No chessboard found in image {i}")

    if not pts2d:
        print("No chessboard patterns found!")
        return None

    print(f"\nCalibrating with {len(pts2d)} images...")
    h, w = gray.shape
    flags = cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_FIX_PRINCIPAL_POINT

    err, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        pts3d, pts2d, (w, h), None, None, flags=flags
    )

    print(f"\nRMS re-projection error: {err} pixels")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients:\n{dist}")

    for i in range(len(pts2d)):
        imgpoints2, _ = cv2.projectPoints(
            pts3d[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(pts2d[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        print(f"Image {i} error: {error} pixels")

    # Save calibration results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    np.save(output_path / 'camera_matrix.npy', K)
    np.save(output_path / 'dist_coeffs.npy', dist)
    print(f"\nCalibration results saved to {output_path}")

    # Visualize calibration results - both views
    viz_calibration(
        corners3d,  # Original chessboard pattern
        (board_height, board_width),
        rvecs,  # Camera rotations
        tvecs,  # Camera translations
        axis_length=2.2 * square_size
    )

    viz_calibration_boards(
        corners3d,  # Original chessboard pattern
        (board_height, board_width),
        rvecs,  # Board rotations
        tvecs,  # Board translations
        axis_length=2.2 * square_size
    )

    return err, K, dist, rvecs, tvecs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera calibration from chessboard images')
    parser.add_argument('--images', type=str, default='calibration_images',
                        help='Path to directory containing calibration images')
    parser.add_argument('--board-height', type=int, required=True,
                        help='Number of internal corners in height')
    parser.add_argument('--board-width', type=int, required=True,
                        help='Number of internal corners in width')
    parser.add_argument('--square-size', type=float, required=True,
                        help='Size of each square in mm')
    parser.add_argument('--output-dir', type=str, default='calibration',
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
        output_dir=args.output_dir,
        show_images=args.show_images
    )
