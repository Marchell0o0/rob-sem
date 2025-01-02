import numpy as np
import cv2
from src.camera_image import CameraImage
from src.scene3d import Scene3D
from src.robot_box import RobotBox, RobotType
from src.se3 import SE3

# Load camera calibration
camera_matrix = np.load("calibration/calibration_data/camera_matrix.npy")
dist_coeffs = np.load("calibration/calibration_data/dist_coeffs.npy")

# Create image scene
image = CameraImage(camera_matrix, dist_coeffs)
image.set_image(cv2.imread("images/capture_20241219_174515.png"))

# Detect ArUco markers and boards
arucos = image.get_arucos(36, cv2.aruco.DICT_4X4_50)
boards = image.detect_boards()

# Draw markers and board slots
image.draw_arucos(arucos)
for board in boards:
    image.draw_board_slots(board)
image.display()

# Create 3D scene for camera view
scene_camera = Scene3D().invert_z_axis().z_from_zero()
scene_camera.add_transform("Camera", SE3())

# Add ArUco markers
for marker_id, pose in arucos.items():
    scene_camera.add_transform(f"ID: {marker_id}", pose)

# Add boards and their slots
for board in boards:
    scene_camera.add_board(board)

scene_camera.display()

# Create robot scene
scene_robot = Scene3D().z_from_zero()
box = RobotBox(RobotType.CRS97, robot_active=False, camera_active=False)

# Show robot in home position
print("Visualizing home position...")
scene_robot.add_robot(box, box.robot.q_home)
scene_robot.display()
