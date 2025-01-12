from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.enums import RobotType
import cv2
from src.board import Board
from src.camera_image import CameraImage
from src.scene3d import Scene3D
from pathlib import Path
class GripperWrapper():
    def __init__(self, gripper, robot_type) -> None:
            self.gripper = gripper
            self.robot_type = robot_type
    
    def open(self):
        if self.robot_type == RobotType.CRS97 or self.robot_type == RobotType.CRS93:
            self.gripper.control_position(1000)
            self.gripper.wait_for_motion_stop()
            self.gripper.control_position(1000)
            self.gripper.wait_for_motion_stop()
        elif self.robot_type == RobotType.RV6S:
            self.gripper.open()

    def close(self):
        if self.robot_type == RobotType.CRS97 or self.robot_type == RobotType.CRS93:
            self.gripper.control_position(-1000)
            self.gripper.wait_for_motion_stop()
            self.gripper.control_position(-1000)
            self.gripper.wait_for_motion_stop()
        elif self.robot_type == RobotType.RV6S:
            self.gripper.close()

class RobotBox():
    def __init__(self, robot_type: RobotType, robot_active: bool = True, camera_active: bool = True):
        self.robot_type = robot_type
        if robot_type == RobotType.CRS93:
            from ctu_crs import CRS93
            self.robot = CRS93(
                tty_dev=None if not robot_active else "/dev/mars")
            self.gripper = GripperWrapper(self.robot.gripper, robot_type)
        elif robot_type == RobotType.CRS97:
            from ctu_crs import CRS97
            self.robot = CRS97(
                tty_dev=None if not robot_active else "/dev/mars")
            self.gripper = GripperWrapper(self.robot.gripper, robot_type)
        elif robot_type == RobotType.RV6S:
            from ctu_mitsubishi import Rv6s, Rv6sGripper
            self.robot = Rv6s()
            self.gripper = GripperWrapper(Rv6sGripper, robot_type)

        if robot_active:
            self.robot.initialize(home=False)

        

        if camera_active:
            from src.camera import Camera
            self.camera = Camera(robot_type)
        else:
            self.camera = None

        self.BOARD_ARUCO_SIZE = 36
        self.BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50

        # Calibruco cube
        self.CALIBRATION_ARUCO_ID = 2
        # self.CALIBRATION_ARUCO_SIZE = 29
        self.CALIBRATION_ARUCO_SIZE = 38
        self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50
        self.calibration_aruco_configurations = []

        configurations_path = Path("only_10_both_robots")
        for file in configurations_path.glob("*.npy"):
            self.calibration_aruco_configurations.append(np.load(file))

        # CRS93
        # levels = [np.array([0, -10, -110, 0, -60, 0]), np.array([0, -30, -110, 0, -40, 0]),
        #           np.array([0, -45, -102, 0, -33, 0])]
        
        # CRS97
        # levels = [np.array([0, -10, -115, 0, -55, 0]), np.array([0, -30, -110, 0, -40, 0]),
        #           np.array([0, -45, -102, 0, -33, 0])]

        # offsets = [-20, -15, -10, -5, 0 , 5, 10, 15]
        # angles = [0, 15]
        # for level in levels:
        #     for offset in offsets:
        #         config = level
        #         config[0] = offset
        #         for angle in angles:
        #             config[4] += angle
        #             self.calibration_aruco_configurations.append(np.deg2rad(config))
        #             config[4] -= angle

        print(np.rad2deg(self.calibration_aruco_configurations).round()) 


        

        # self.calibration_aruco_configurations = [
        #     np.deg2rad([-10, 30, 130, 0, -70, 0]),
        #     np.deg2rad([-5, 30, 130, 0, -70, 0]),
        #     np.deg2rad([0, 30, 130, 0, -70, 0]),
        #     np.deg2rad([5, 30, 130, 0, -70, 0]),
        #     np.deg2rad([5, 30, 130, 0, -70, 30]),
        #     np.deg2rad([10, 30, 130, 0, -70, 0]),
        #     np.deg2rad([15, 30, 130, 0, -70, 0]),
        #     np.deg2rad([20, 30, 130, 0, -70, 0]),
        #     np.deg2rad([25, 30, 130, 0, -70, 0]),
        #     np.deg2rad([25, 15, 145, 0, -70, 0]),
        #     np.deg2rad([20, 15, 145, 0, -70, 0]),
        #     np.deg2rad([15, 15, 145, 0, -70, 0]),
        #     np.deg2rad([10, 15, 145, 0, -70, 0]),
        #     np.deg2rad([5, 15, 145, 0, -70, 0]),
        #     np.deg2rad([5, 15, 145, 0, -70, 30]),
        #     np.deg2rad([0, 15, 145, 0, -70, 0]),
        #     np.deg2rad([-5, 15, 145, 0, -70, 0]),
        #     np.deg2rad([-10, 15, 145, 0, -70, 0]),
        #     np.deg2rad([-10, 45, 115, 0, -70, 0]),
        #     np.deg2rad([-5, 45, 115, 0, -70, 0]),
        #     np.deg2rad([0, 45, 115, 0, -70, 0]),
        #     np.deg2rad([5, 45, 115, 0, -70, 0]),
        #     np.deg2rad([5, 45, 115, 0, -70, 30]),
        #     np.deg2rad([10, 45, 115, 0, -70, 0]),
        #     np.deg2rad([15, 45, 115, 0, -70, 0]),
        #     np.deg2rad([20, 45, 115, 0, -70, 0]),
        #     np.deg2rad([25, 45, 115, 0, -70, 0]),
        #     np.deg2rad([25, 45, 145, 0, -100, 0]),
        #     np.deg2rad([20, 45, 145, 0, -100, 0]),
        #     np.deg2rad([15, 45, 145, 0, -100, 0]),
        #     np.deg2rad([10, 45, 145, 0, -100, 0]),
        #     np.deg2rad([5, 45, 145, 0, -100, 0]),
        #     np.deg2rad([5, 45, 145, 0, -100, -30]),
        #     np.deg2rad([0, 45, 145, 0, -100, 0]),
        #     np.deg2rad([-5, 45, 145, 0, -100, 0]),
        #     np.deg2rad([-10, 45, 145, 0, -100, 0]),
        #     np.deg2rad([-10, 60, 100, 0, -70, 0]),
        #     np.deg2rad([-5, 60, 100, 0, -70, 0]),
        #     np.deg2rad([0, 60, 100, 0, -70, 0]),
        #     np.deg2rad([5, 60, 100, 0, -70, 0]),
        #     np.deg2rad([5, 60, 100, 0, -70, -30]),
        #     np.deg2rad([10, 60, 100, 0, -70, 0]),
        #     np.deg2rad([15, 60, 100, 0, -70, 0]),
        #     np.deg2rad([20, 60, 100, 0, -70, 0]),
        #     np.deg2rad([25, 60, 100, 0, -70, 0]),
        #     np.deg2rad([-10, 60, 100, 0, -90, 0]),
        #     np.deg2rad([-5, 60, 100, 0, -90, 0]),
        #     np.deg2rad([0, 60, 100, 0, -90, 0]),
        #     np.deg2rad([5, 60, 100, 0, -90, 0]),
        #     np.deg2rad([5, 60, 100, 0, -90, 0]),
        #     np.deg2rad([10, 60, 100, 0, -90, 0]),
        #     np.deg2rad([15, 60, 100, 0, -90, 0]),
        #     np.deg2rad([20, 60, 100, 0, -90, 0]),
        #     np.deg2rad([25, 60, 100, 0, -90, 0]),
        # ]


    def solve_AX_YB(self, gripper_poses, robot_poses):
        """
        Solve the AX=YB calibration problem to find the camera-to-base (X) and gripper-to-flange (Y) transforms.
        Uses the robot-world-hand-eye calibration method from OpenCV.

        Args:
            gripper_poses (list): List of gripper poses in camera frame
            robot_poses (list): List of end-effector poses in base frame

        Returns:
            tuple: (camera_to_base, gripper_to_flange) transforms
        """
        # Convert poses to rotation matrices and translation vectors
        R_gripper = []  # Rotation matrices of gripper in camera frame
        t_gripper = []  # Translation vectors of gripper in camera frame
        R_robot = []    # Rotation matrices of end-effector in base frame
        t_robot = []    # Translation vectors of end-effector in base frame

        # Validate and convert poses
        for gripper, robot in zip(gripper_poses, robot_poses):

            print("gripper: ", gripper)
            print("robot: ", robot)

            # Get rotation and translation from gripper pose
            R_g = gripper.rotation.rot
            t_g = gripper.translation.reshape(3, 1)
            
            # Get rotation and translation from robot pose
            R_r = robot.rotation.rot
            t_r = robot.translation.reshape(3, 1)

            # Validate rotation matrices
            det_g = np.linalg.det(R_g)
            det_r = np.linalg.det(R_r)
            
            if abs(det_g - 1.0) > 1e-6 or abs(det_r - 1.0) > 1e-6:
                print(f"Warning: Invalid rotation matrix detected!")
                print(f"Determinant of gripper rotation: {det_g}")
                print(f"Determinant of robot rotation: {det_r}")
                continue

            # Check for NaN or Inf values
            if np.any(np.isnan(R_g)) or np.any(np.isnan(R_r)) or \
               np.any(np.isnan(t_g)) or np.any(np.isnan(t_r)) or \
               np.any(np.isinf(R_g)) or np.any(np.isinf(R_r)) or \
               np.any(np.isinf(t_g)) or np.any(np.isinf(t_r)):
                print("Warning: NaN or Inf values detected in poses!")
                continue

            R_gripper.append(R_g)
            t_gripper.append(t_g)
            R_robot.append(R_r)
            t_robot.append(t_r)

        if len(R_gripper) < 3:
            raise ValueError(f"Not enough valid poses for calibration. Need at least 3, got {len(R_gripper)}")

        print(f"Using {len(R_gripper)} valid poses for calibration")

        # Convert lists to numpy arrays
        R_gripper = np.array(R_gripper)
        t_gripper = np.array(t_gripper)
        R_robot = np.array(R_robot)
        t_robot = np.array(t_robot)

        # Print some statistics about the poses
        print("\nPose Statistics:")
        print("Translation ranges (min, max) in mm:")
        print(f"Gripper X: ({t_gripper[:, 0].min():.1f}, {t_gripper[:, 0].max():.1f})")
        print(f"Gripper Y: ({t_gripper[:, 1].min():.1f}, {t_gripper[:, 1].max():.1f})")
        print(f"Gripper Z: ({t_gripper[:, 2].min():.1f}, {t_gripper[:, 2].max():.1f})")
        print(f"Robot X: ({t_robot[:, 0].min():.1f}, {t_robot[:, 0].max():.1f})")
        print(f"Robot Y: ({t_robot[:, 1].min():.1f}, {t_robot[:, 1].max():.1f})")
        print(f"Robot Z: ({t_robot[:, 2].min():.1f}, {t_robot[:, 2].max():.1f})")

        # Perform the calibration
        try:
            # R_gf, t_gf, R_cb, t_cb = cv2.calibrateRobotWorldHandEye(
            #     R_gripper, t_gripper, R_robot, t_robot,
            #     cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
            # )
            R_gf, t_gf, R_cb, t_cb = cv2.calibrateRobotWorldHandEye(
                R_robot, t_robot, R_gripper, t_gripper,
                cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
            )
        except cv2.error as e:
            print("\nCalibration failed! Try collecting new calibration data with:")
            print("1. More diverse robot poses (different angles and positions)")
            print("2. Ensure the ArUco marker is clearly visible in all images")
            print("3. Make sure the robot poses are accurate")
            raise e

        # Create SE3 transforms from results
        camera_to_base = SE3(
            translation=t_cb.flatten(),
            rotation=SO3(rotation_matrix=R_cb)
        )
        gripper_to_flange = SE3(
            translation=t_gf.flatten(),
            rotation=SO3(rotation_matrix=R_gf)
        )

        return camera_to_base, gripper_to_flange
    
    def get_camera_to_base_transform(self) -> SE3 | None:
        """
        Get the transform from the camera to the base of the robot

        returns:
            SE3: the transform from the camera to the base of the robot
        """

        gripper_poses = []  # Gripper poses in camera frame
        robot_poses = []    # End-effector poses in base frame

        scene_camera = Scene3D().invert_z_axis().z_from_zero()
        scene_camera.add_transform("Camera", SE3())

        scene_robot = Scene3D().z_from_zero()
        scene_robot.add_transform("Base", SE3())

        for idx, config in enumerate(self.calibration_aruco_configurations):
            if self.robot._initialized:
                self.robot.move_to_q(config)
                self.robot.wait_for_motion_stop()

            img = None
            img = self.camera.grab_image()
            while img.image is None or img.image.size == 0:
                img = self.camera.grab_image()
                print("waiting for image")


            arucos = img.get_arucos(
                self.CALIBRATION_ARUCO_SIZE, self.CALIBRATION_ARUCO_DICT)
            print(arucos)
            img.draw_arucos(arucos)
            img.display()

            if self.CALIBRATION_ARUCO_ID not in arucos:
                print(f"Calibration ArUco not found for config {config}")
                continue

            gripper = arucos[self.CALIBRATION_ARUCO_ID]
            gripper_poses.append(gripper)
            
            img.add_transform(f"Gripper in pose {idx + 1}", gripper)
            
            scene_camera.add_transform(f"Gripper in pose {idx + 1}", gripper)

            # Get robot end-effector pose in base frame
            if not self.robot or not self.robot._initialized:
                print("Can't continue without robot because of fk")
                continue

            q = self.robot.get_q()
            scene_robot.add_robot(self, q)

            # Get flange pose from FK
            fk = self.robot.fk(q)
            end_effector = SE3().from_matrix(fk, "meters")

            # Apply gripper offset to get actual gripper pose
            # end_effector = flange * self.gripper_offset
            robot_poses.append(end_effector)

            scene_robot.add_transform(f"End effector in pose {idx + 1}", end_effector)

            print(f"Configuration {idx + 1}:")
            print("Joint angles:", np.rad2deg(q).round())
            print("Gripper in camera:", gripper)
            print("End effector in base:", end_effector)
            print()

        np.save("calibration/calibration_data/gripper_poses.npy", gripper_poses)
        np.save("calibration/calibration_data/robot_poses.npy", robot_poses)
        # gripper_poses = np.load("calibration/calibration_data/gripper_poses.npy", allow_pickle=True)
        # robot_poses = np.load("calibration/calibration_data/robot_poses.npy", allow_pickle=True)
        print("gripper_poses: ", gripper_poses)
        print("robot_poses: ", robot_poses)
        print("len(gripper_poses), len(robot_poses):", len(gripper_poses), len(robot_poses))
        # scene_camera.display()
        # scene_robot.display()


        if len(gripper_poses) < 3 or len(robot_poses) < 3:
            print("Not enough valid poses for calibration")
            return None

        # Solve AX=YB to get camera-to-base (X) and gripper-to-flange (Y) transforms
        camera_to_base, gripper_to_flange = self.solve_AX_YB(
            gripper_poses, robot_poses)

        # my_gripper_to_flange, my_camera_to_base = self.solve_robot_hand_eye_with_fixed_camera(robot_poses, gripper_poses)
        print("Camera to base transform:", camera_to_base)
        print("Gripper to flange transform:", gripper_to_flange)

        # print("my_camera_to_base: ", my_camera_to_base)
        # print("my_gripper_to_flange: ", my_gripper_to_flange)

        np.save("calibration/calibration_data/camera_to_base.npy", camera_to_base.to_matrix())
        np.save("calibration/calibration_data/gripper_to_flange.npy", gripper_to_flange.to_matrix())

        print("saved matrices camera_to_base.npy and gripper_to_flange.npy")

        return camera_to_base

    def find_boards(self):
        if self.robot._initialized:
            self.robot.move_to_q(np.deg2rad([90, 0, 90, 0, 90, 0]))
            self.robot.wait_for_motion_stop()
        else:
            print("No robot found")

        img = self.camera.grab_image()
        while img.image is None or img.image.size == 0:
            img = self.camera.grab_image()
            print("waiting for image")
        # img = CameraImage(self.camera.camera_matrix, self.camera.dist_coeffs).set_image(np.array(cv2.imread("boards_dataset/board_01_20250105_122115.png")))

        # Get ArUco corners
        aruco_corners = img.get_aruco_corners(self.BOARD_ARUCO_DICT)
        # Create boards for each valid pair
        boards = []
        for pair in Board.VALID_PAIRS:
            if pair[0] in aruco_corners and pair[1] in aruco_corners:
                board = Board(pair[0], pair[1])
                board.calculate_board_transform(aruco_corners, img.camera_matrix, img.dist_coeffs)
                board.calculate_slot_transforms()
                if board.board_transform is not None:
                    boards.append(board)

        # boards = img.detect_boards()
        img.mark_boards_empty(boards)

        for board in boards:
            img.draw_board_slots(board)
            img.add_transform(f"Board {board.pair}", board.board_transform, axis_length=300.0)
        #     # img.add_transform(f"Board {board.pair}, aruco {board.second_marker_id}", board.second_marker_transform)
        #     img.add_transform(f"Board transform {board.pair}", board.board_transform)
        img.display()

        return boards

    def close(self):
        if self.robot is not None:
            self.robot.wait_for_motion_stop()
            if self.robot_type == RobotType.RV6S:
                self.robot.close_connection()
            else:
                self.robot.close()
        if self.camera is not None:
            self.camera.close()
