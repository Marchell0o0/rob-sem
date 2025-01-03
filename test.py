import numpy as np
import cv2
from src.camera_image import CameraImage
from src.scene3d import Scene3D
from src.robot_box import RobotBox, RobotType
from src.se3 import SE3


box = RobotBox(RobotType.RV6S)
scene = Scene3D().z_from_zero()

# box.robot.move_to_q(np.deg2rad([0, 90, 0, 0, 0, 0]))
# box.robot.move_to_q(np.deg2rad([0, 0, 90, 0, 0, 0]))
# box.robot.move_to_q(np.deg2rad([0, 30, 130, 0, -70, -90]))
# q_soft_home = np.deg2rad([90, 0, 90, 0, 90, 0])
q_soft_home = np.deg2rad([0, 60, 60, 0, 60, 0])
box.robot.move_to_q(q_soft_home)
box.robot.wait_for_motion_stop()
print(box.robot.fk(q_soft_home))
# box.robot.soft_home()
# box.robot.wait_for_motion_stop()
# q = box.robot.get_q()
# scene.add_robot(box, q_soft_home)
# scene.display()
# scene.add_robot(box, np.deg2rad([0, 30, 130, 0, -70, -90]))
# scene.display()


# # Load camera calibration
# camera_matrix = np.load("calibration/calibration_data/camera_matrix.npy")
# dist_coeffs = np.load("calibration/calibration_data/dist_coeffs.npy")

# # Create image scene
# image = CameraImage(camera_matrix, dist_coeffs)
# image.set_image(cv2.imread("images/capture_20241219_174515.png"))

# # Detect ArUco markers and boards
# arucos = image.get_arucos(36, cv2.aruco.DICT_4X4_50)
# boards = image.detect_boards()
# image.mark_boards_empty(boards)

# # Draw markers and board slots
# image.draw_arucos(arucos)
# for board in boards:
#     image.draw_board_slots(board)


# image.display()

# # Create 3D scene for camera view
# scene_camera = Scene3D().invert_z_axis().z_from_zero()
# scene_camera.add_transform("Camera", SE3())

# # Add ArUco markers
# for marker_id, pose in arucos.items():
#     scene_camera.add_transform(f"ID: {marker_id}", pose)

# # Add boards and their slots
# for board in boards:
#     scene_camera.add_board(board)

# scene_camera.display()

# # Create robot scene
# scene_robot = Scene3D().z_from_zero()
# box = RobotBox(RobotType.CRS97, robot_active=False, camera_active=False)

# # Show robot in home position
# print("Visualizing home position...")
# calibration_aruco_configurations = [
#     np.deg2rad([0, 30, 130, 0, -70, -90]),
#     np.deg2rad([20, 30, 130, 0, -70, -90]),
#     np.deg2rad([40, 30, 130, 0, -70, -90]),
#     np.deg2rad([-20, 30, 130, 0, -70, -90]),
#     np.deg2rad([-40, 30, 130, 0, -70, -90])
# ]
# for config in calibration_aruco_configurations:
#     scene_robot.add_robot(box, q=config)
#     scene_robot.display()
