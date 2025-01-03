from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.enums import RobotType
import cv2

from src.scene3d import Scene3D


class RobotBox():
    def __init__(self, robot_type: RobotType, robot_active: bool = True, camera_active: bool = True):
        self.robot_type = robot_type
        if robot_type == RobotType.CRS93:
            from ctu_crs import CRS93
            self.robot = CRS93(
                tty_dev=None if not robot_active else "/dev/null")
        elif robot_type == RobotType.CRS97:
            from ctu_crs import CRS97
            self.robot = CRS97(
                tty_dev=None if not robot_active else "/dev/null")
        elif robot_type == RobotType.RV6S:
            from ctu_mitsubishi import Rv6s
            self.robot = Rv6s()

        if robot_active:
            self.robot.initialize()

        if camera_active:
            from src.camera import Camera
            self.camera = Camera(robot_type)
        else:
            self.camera = None

        # CRS93
        # self.calibration_aruco_offset = np.deg2rad([0, 10, -70, 0, 60, 176])
        # self.gripper_to_aruco =  \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 0, 5]), ["x", "y", "z"])) * \
        #       SE3(translation=[0, -70, -20]) * \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles([np.pi/2, 0, np.pi/2], ["x", "y", "z"])) * \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([-5, 0, 0]), ["x", "y", "z"]))

        # CRS97
        # self.gripper_to_aruco = SE3(translation=[70, 33, 0])

        self.BOARD_ARUCO_SIZE = 36
        self.BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50

        # CRS93
        # self.CALIBRATION_ARUCO_ID = 2
        # self.CALIBRATION_ARUCO_SIZE = 38
        # self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50

        # RV6S
        self.CALIBRATION_ARUCO_ID = 2
        self.CALIBRATION_ARUCO_SIZE = 38
        self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50

        self.calibration_aruco_configurations = [
            np.deg2rad([-10, 30, 130, 0, -70, -90]),
            np.deg2rad([-10, 30, 130, 0, -70, -45]),
            np.deg2rad([0, 30, 130, 0, -70, -90]),
            np.deg2rad([0, 30, 130, 0, -70, -45]),
            np.deg2rad([20, 30, 130, 0, -70, -90]),
            np.deg2rad([20, 30, 130, 0, -70, -45]),


            np.deg2rad([-10, 45, 115, 0, -70, -90]),
            np.deg2rad([-10, 45, 115, 0, -70, -45]),
            np.deg2rad([0, 45, 115, 0, -70, -90]),
            np.deg2rad([0, 45, 115, 0, -70, -45]),
            np.deg2rad([20, 45, 115, 0, -70, -90]),
            np.deg2rad([20, 45, 115, 0, -70, -45]),


            np.deg2rad([-10, 60, 100, 0, -70, -90]),
            np.deg2rad([-10, 60, 100, 0, -70, -45]),
            np.deg2rad([0, 60, 100, 0, -70, -90]),
            np.deg2rad([0, 60, 100, 0, -70, -45]),
            np.deg2rad([20, 60, 100, 0, -70, -90]),
            np.deg2rad([20, 60, 100, 0, -70, -45]),


            np.deg2rad([-10, 60, 100, 0, -90, -90]),
            np.deg2rad([-10, 60, 100, 0, -90, -45]),
            np.deg2rad([0, 60, 100, 0, -90, -90]),
            np.deg2rad([0, 60, 100, 0, -90, -45]),
            np.deg2rad([20, 60, 100, 0, -90, -90]),
            np.deg2rad([20, 60, 100, 0, -90, -45]),
        ]

        # self.gripper_offset = SE3(translation=[-20, 0, 160])

        self.aruco_to_gripper = SE3(translation=[70, 0, -30]) * \
            SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(
                np.deg2rad([0, 90, -90]), ["x", "y", "z"]))

        # # DH parameters for RV-6S robot
        # self.robot.dh_theta_off = np.deg2rad(
        #     [0, -90, -90, 0, 0, 180])  # Joint angle offset
        # self.robot.dh_a = np.array(
        #     [85, 280, 100, 0, 0, 0]) / 1000.0    # Link length
        # self.robot.dh_d = np.array(
        #     [350, 0, 0, 315, 0, 85]) / 1000.0    # Link offset
        # self.robot.dh_alpha = np.deg2rad(
        #     [-90, 0, -90, 90, -90, 0])     # Link twist

        self.robot.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])
        self.robot.dh_a = np.array([80, 280 - 5, 100, 0, 0, 0]) / 1000.0
        self.robot.dh_d = np.array([350-20, 0, 0, 315, 0, 85 + 165 - 5]) / 1000.0
        self.robot.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])

    def solve_AX_YB(self, a: list[SE3], b: list[SE3]) -> tuple[SE3, SE3]:
        """Solve A^iX=YB^i, return X, Y
        
        Args:
            a: List of gripper poses in camera frame
            b: List of end-effector poses in base frame
        
        Returns:
            tuple[SE3, SE3]: (camera-to-base, gripper-to-flange)
        """
        rvec_a = [T.rotation.log() for T in a]
        tvec_a = [T.translation for T in a]
        rvec_b = [T.rotation.log() for T in b]
        tvec_b = [T.translation for T in b]
        
        # cv2.calibrateRobotWorldHandEye returns (R1, t1, R2, t2)
        # where R1,t1 is gripper-to-flange and R2,t2 is camera-to-base
        R_gf, t_gf, R_cb, t_cb = cv2.calibrateRobotWorldHandEye(
            rvec_a, tvec_a, rvec_b, tvec_b)
        
        # Return in correct order: camera-to-base, gripper-to-flange
        return SE3(t_cb[:, 0], SO3(R_cb)), SE3(t_gf[:, 0], SO3(R_gf))

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

            img = self.camera.grab_image()
            arucos = img.get_arucos(
                self.CALIBRATION_ARUCO_SIZE, self.CALIBRATION_ARUCO_DICT)

            img.draw_arucos(arucos)

            if self.CALIBRATION_ARUCO_ID not in arucos:
                print(f"Calibration ArUco not found for config {config}")
                continue

            # Get gripper pose in camera frame
            gripper = arucos[self.CALIBRATION_ARUCO_ID] * self.aruco_to_gripper
            gripper_poses.append(gripper)
            
            img.add_transform(f"Gripper in pose {idx + 1}", gripper)
            img.display()
            
            scene_camera.add_transform(f"Gripper in pose {idx + 1}", gripper)

            # Get robot end-effector pose in base frame
            if not self.robot:
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

        scene_camera.display()
        scene_robot.display()

        if len(gripper_poses) < 3 or len(robot_poses) < 3:
            print("Not enough valid poses for calibration")
            return None

        # Solve AX=YB to get camera-to-base (X) and gripper-to-flange (Y) transforms
        camera_to_base, gripper_to_flange = self.solve_AX_YB(
            gripper_poses, robot_poses)
        print("Camera to base transform:", camera_to_base)
        print("Gripper to flange transform:", gripper_to_flange)
        print(
            "Gripper to flange should be close to identity since we pre-applied the offset")

        np.save("calibration/calibration_data/camera_to_base.npy", camera_to_base.to_matrix())
        np.save("calibration/calibration_data/gripper_to_flange.npy", gripper_to_flange.to_matrix())

        return camera_to_base

    def find_boards(self):
        if self.robot is not None:
            self.robot.move_to_q(np.deg2rad([90, 0, 90, 0, 90, 0]))
            self.robot.wait_for_motion_stop()
        else:
            print("No robot found")

        img = self.camera.grab_image()
        boards = img.detect_boards()
        img.mark_boards_empty(boards)

        for board in boards:
            img.draw_board_slots(board)
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
