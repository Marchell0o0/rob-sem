import cv2
import numpy as np
from board_detection import Board
from se3 import SE3
from ctu_mitsubishi import Rv6s
from so3 import SO3
from time import sleep
from camera import BaslerCamera

def is_slot_empty(slot_transform, camera_matrix, dist_coeffs, img):
    """
    Check if a slot is empty by projecting its transform onto the image
    and detecting a circle nearby using Hough Circle Transform.
[
    Args:
        slot_transform: SE3 transform of the slot.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        img: Image from the camera.

    Returns:
        bool: True if the slot is empty, False otherwise.
    """
    # Project the slot center onto the image
    slot_rvec, _ = cv2.Rodrigues(slot_transform.rotation.rot)
    slot_tvec = slot_transform.translation.reshape(3, 1)
    point_3d = np.float32([[0, 0, 0]])  # Origin in slot's frame
    point_2d, _ = cv2.projectPoints(
        point_3d,
        slot_rvec, slot_tvec,
        camera_matrix, dist_coeffs
    )

    # Convert to integer pixel coordinates
    center = (int(round(point_2d[0][0][0])), int(round(point_2d[0][0][1])))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 30, 100)

    # Find contours in the edge map
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw contours
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        contour_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    # Check if any circle contains the slot center
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if np.linalg.norm(np.array([x, y]) - np.array(center)) < r:
                return True

    return False

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
robot_type = "rv6s"


# Initialize camera
camera = BaslerCamera()
if robot_type == "crs93":
    camera.connect_by_ip("192.168.137.107")
    camera.connect_by_name("camera-crs93")
elif robot_type == "crs97":
    camera.connect_by_ip("192.168.137.106")
    camera.connect_by_name("camera-crs97")
elif robot_type == "rv6s":
    camera.connect_by_ip("192.168.137.109")
    camera.connect_by_name("camera-rv6s")
else:
    raise ValueError(f"Invalid robot type: {robot_type}")
camera.open()
camera.set_parameters()
img = camera.grab_image()
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

empty_slot_count = [0, 0]
# Check each slot for emptiness
for idx, board in enumerate(boards):
    for slot_idx, slot_transform in board.slot_transforms:
        is_empty = is_slot_empty(slot_transform, camera_matrix, dist_coeffs, img)
        print(f"Slot {slot_idx} is {'empty' if is_empty else 'not empty'}.")
        if is_empty:
            empty_slot_count[idx] += 1

print(empty_slot_count)
if empty_slot_count[0] > empty_slot_count[1]:
    goal_board = boards[0]
    srouce_board = boards[1]
else:
    goal_board = boards[1]
    srouce_board = boards[0]


camera_to_robot = SE3(
    rotation=SO3.from_euler_angles(np.deg2rad([180, 0, 90]), ["x", "y", "z"]),
    translation=np.array([500, 140, 1370])
)

robot_gripper_offset = SE3(
    rotation=SO3.from_euler_angles(np.deg2rad([0, 0, 0]), ["x", "y", "z"]),
    translation=np.array([0, 0, 200])
)

robot = Rv6s()
robot.initialize()

robot_fk = robot.fk(robot.get_q())
print("robot_fk: ", robot_fk)

robot_pos = SE3(rotation = SO3(robot_fk[:3, :3]), translation = robot_fk[:3, 3] * 1000)
print("robot: ", robot_pos)


print("first slot in goal board: ", goal_board.slot_transforms[0][1])
print("camera to robot: ", camera_to_robot)

slot_1_in_robot = camera_to_robot * goal_board.slot_transforms[0][1]
print("slot 1 in robot: ", slot_1_in_robot)

slot_1_in_robot_with_gripper = slot_1_in_robot * robot_gripper_offset.inverse()
print("slot 1 in robot with gripper: ", slot_1_in_robot_with_gripper)

full_matrix = np.eye(4)
full_matrix[:3, :3] = slot_1_in_robot_with_gripper.rotation.rot
full_matrix[:3, 3] = slot_1_in_robot_with_gripper.translation / 1000

print("full_matrix: ", full_matrix)

robot_to_slot = robot.ik(full_matrix)
print("robot to slot: ", robot_to_slot)



robot.move_to_q(robot_to_slot[0])

sleep(1)

robot.move_to_q(np.deg2rad([90, 0, 90, 0, 90, 0]))
robot.stop_robot()
robot.close_connection()