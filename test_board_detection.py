import cv2
import numpy as np
from board_detection import Board
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Load and process image
# img = cv2.imread("images/capture_20241215_151731.png")
# img = cv2.imread("images/capture_20241215_173518.png")
img = cv2.imread("images/capture_20241215_173530.png")

if img is None:
    raise FileNotFoundError("Test image not found")

# Load camera calibration
camera_matrix = np.load("calibration/camera_matrix.npy")
dist_coeffs = np.load("calibration/dist_coeffs.npy")

# Detect markers
corners, ids, rejected = detector.detectMarkers(img)

if ids is None:
    print("No markers detected!")
    exit()

print(f"Detected markers with IDs: {ids.flatten()}")

# Create boards from detected markers
boards = Board.create_boards_from_markers(ids.flatten())
print(f"\nFound {len(boards)} boards:")
for board in boards:
    print(f"Board with markers {board.pair}")

# Create corners dictionary for easier lookup
corners_dict = {ids[i][0]: corners[i] for i in range(len(ids))}

# Update board poses and load slot positions
for board in boards:
    # Update poses
    if board.update_poses_from_corners(corners_dict, camera_matrix, dist_coeffs):
        # Load slot positions if poses were updated successfully
        success = board.load_slot_positions(
            f"boards/positions_plate_{board.ref_marker_id:02d}-{board.second_marker_id:02d}.csv")
        if success:
            print(f"Loaded slot positions for board {board.pair}")

# Draw detected markers
cv2.aruco.drawDetectedMarkers(img, corners, ids)

# Draw camera coordinate system origin projected at z=1200mm
axis_length = 50  # mm
axis_points = np.float32([[0, 0, 1200],  # origin
                         [axis_length, 0, 1200],  # x-axis (pointing right)
                         [0, axis_length, 1200]])  # y-axis (pointing down)

img_points, _ = cv2.projectPoints(
    axis_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)

# Draw camera axes
origin = tuple(map(int, img_points[0][0]))
x_end = tuple(map(int, img_points[1][0]))
y_end = tuple(map(int, img_points[2][0]))
cv2.line(img, origin, x_end, (0, 0, 255), 2)  # X-axis in red (right)
cv2.line(img, origin, y_end, (0, 255, 0), 2)  # Y-axis in green (down)
cv2.putText(img, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(img, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Draw boards in 2D
for board in boards:
    board.draw_2d(img, camera_matrix, dist_coeffs)

# Show OpenCV visualization
cv2.imshow('Detected Boards', img)

# Create 3D visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw boards in 3D
for i, board in enumerate(boards):
    board.draw_3d(ax, marker_colors=[f'C{i*2}', f'C{i*2+1}'])

# Set axis limits to make everything visible
all_points = []
for board in boards:
    all_points.extend([
        board.board_transform.translation,
        board.second_marker_transform.translation,
        *[t.translation for _, t in board.slot_transforms]
    ])
all_points = np.array(all_points)

center = np.mean(all_points, axis=0)
max_dist = np.max(np.linalg.norm(all_points - center, axis=1))

ax.set_xlim(center[0] - max_dist, center[0] + max_dist)
ax.set_ylim(center[1] - max_dist, center[1] + max_dist)
ax.set_zlim(center[2] - max_dist/4, center[2] +
            max_dist/4)  # Make Z range smaller

# Set labels and appearance
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_box_aspect([1, 1, 0.3])  # Make Z dimension smaller

ax.legend()

# Show both visualizations
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
