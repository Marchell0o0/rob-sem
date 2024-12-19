import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_display_chessboard(image_path, board_height, board_width):
    """
    Detect and display chessboard corners in an image.

    Args:
        image_path (str): Path to the image file.
        board_height (int): Number of internal corners in height.
        board_width (int): Number of internal corners in width.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the chessboard size
    CHECKERBOARD = (board_height, board_width)

    # Find the chessboard corners
    success, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if success:
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, success)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Chessboard Corners')
        plt.axis('off')
        plt.show()
    else:
        print("No chessboard found in the image.")

# Example usage
detect_and_display_chessboard('calibration_images/calib_01_20241219_173258.png', board_height=6, board_width=10)