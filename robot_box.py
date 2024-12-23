from se3 import SE3
from so3 import SO3
from ctu_crs import CRS97, CRS93
from ctu_mitsubishi import Rv6s
import numpy as np
from camera import Camera
from enums import RobotType
import cv2

from board import Board

class RobotBox():
    def __init__(self, robot_type: RobotType, robot_active: bool = True):
        self.robot_type = robot_type
        if robot_active:
            if robot_type == RobotType.CRS93:
                self.robot = CRS93()
            elif robot_type == RobotType.CRS97:
                self.robot = CRS97()
            elif robot_type == RobotType.RV6S:
                self.robot = Rv6s()
            self.robot.initialize()
        else:
            self.robot = None
        self.camera = Camera(robot_type)

        # CRS93
        # self.calibration_aruco_offset = np.deg2rad([0, 10, -70, 0, 60, 176])
        # self.gripper_to_aruco =  \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 0, 5]), ["x", "y", "z"])) * \
        #       SE3(translation=[0, -70, -20]) * \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles([np.pi/2, 0, np.pi/2], ["x", "y", "z"])) * \
        #       SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([-5, 0, 0]), ["x", "y", "z"]))
    

        # CRS97
        # self.gripper_to_aruco = SE3(translation=[70, 33, 0])


        # RV6S
        self.calibration_aruco_configuration = np.deg2rad([0, 30, 130, 0, -70, -90])
        self.gripper_offset = SE3(translation=[-20, 0, 160])

        # T_AG
        self.aruco_to_gripper = SE3(translation=[70, 0, -30]) * \
                                 SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 90, -90]), ["x", "y", "z"]))
                                 
        self.BOARD_ARUCO_SIZE = 36
        self.BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50


        # CRS93
        # self.CALIBRATION_ARUCO_ID = 2
        # self.CALIBRATION_ARUCO_SIZE = 38
        # self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50

        # RV6S
        self.CALIBRATION_ARUCO_ID = 2
        # self.CALIBRATION_ARUCO_SIZE = 45
        self.CALIBRATION_ARUCO_SIZE = 38
        self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50

        # DH parameters for RV-6S robot
        self.robot.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])  # Joint angle offset
        self.robot.dh_a = np.array([85, 280, 100, 0, 0, 0]) / 1000.0    # Link length
        self.robot.dh_d = np.array([350, 0, 0, 315, 0, 85]) / 1000.0    # Link offset
        self.robot.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])     # Link twist

    def get_camera_to_base_transform(self):
        """
        Get the transform from the camera to the base of the robot

        returns:
            SE3: the transform from the camera to the base of the robot
        """
        if self.robot is not None:
            # self.robot.soft_home()

            # q = self.robot.get_q()
            # print("Home q: ", np.rad2deg(q).round())
            self.robot.move_to_q(self.calibration_aruco_configuration)
            
            self.robot.wait_for_motion_stop()

        img = self.camera.grab_image()

        arucos = self.camera.get_arucos(img, self.CALIBRATION_ARUCO_SIZE, self.CALIBRATION_ARUCO_DICT)
        print("Calibration ArUco: ", arucos)

        
        transforms = arucos

        gripper = None
        if self.CALIBRATION_ARUCO_ID in arucos:
            # v_G = T_AG^-1 * v_A
            gripper = arucos[self.CALIBRATION_ARUCO_ID] * self.aruco_to_gripper

        transforms["Gripper"] = gripper
        print("Gripper in camera: ", gripper)


        base = None
        if self.robot:
            q = self.robot.get_q()
            print("q for fk: ", np.rad2deg(q).round())
            fk = self.robot.fk(q)
            end_effector = SE3().from_matrix(fk, "meters")

            gripper_in_base =  end_effector * SE3(rotation=SO3.from_euler_angles([0, 0, -q[5]], ["x", "y", "z"])) * self.gripper_offset * SE3(rotation=SO3.from_euler_angles([0, 0, q[5]], ["x", "y", "z"]))

            print("End effector: ", end_effector)
            print("Gripper in base: ", gripper_in_base)
            # T_CB = T_CG * T_BG^-1
            # T_CB = T_CG * T_GB
            base = gripper * gripper_in_base.inverse()
            transforms["Base"] = base
            print("camera to base: ", base.inverse())


        # our_camera_in_base = SE3(translation=[450, 0, 1170]) * \
        #                      SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 180, 90]), ["x", "y", "z"]))
        
        
        img = self.camera.draw_arucos(img, arucos)
        img = self.camera.draw_transform(img, transforms["Gripper"], "Gripper")
        
        # gripper_to_our_base = our_camera_in_base * gripper_in_base
        # print("gripper to our base: ", gripper_to_our_base)
        # transforms["Gripper to our base"] = gripper_to_our_base
        # img = self.camera.draw_transform(img, gripper_to_our_base, "gripper to our base")

        self.camera.display_image(img)

        self.camera.display_transforms_3d(transforms)


        return base
        
    def find_boards(self):
        if self.robot is not None:
            self.robot.soft_home()
            q = self.robot.get_q()
            self.robot.move_to_q(q + np.deg2rad([90, 0, 0, 0, 0, 0]))
            self.robot.wait_for_motion_stop()
        else:   
            print("No robot found")

        img = self.camera.grab_image()
        aruco_transforms = self.camera.get_arucos(img, self.BOARD_ARUCO_SIZE, self.BOARD_ARUCO_DICT)
        print(f"Detected ArUco transforms: {list(aruco_transforms.keys())}")
        
        # Create boards from detected ArUcos
        boards = Board.create_boards_from_transforms(aruco_transforms)
        print(f"Found {len(boards)} boards")
        
        # Display the image with detected markers
        img = self.camera.draw_arucos(img, aruco_transforms)
        for board in boards:
            print(f"Board {board.pair} has {len(board.slots)} slots")
            for slot in board.slots:
                print(f"Drawing slot {slot[0]} for board {board.pair}")
                img = self.camera.draw_transform(img, slot[1], f"Slot {slot[0]}")

        self.camera.display_image(img)


        
        return boards
    def close(self):
        if self.robot is not None:
            self.robot.wait_for_motion_stop()
            if self.robot_type == RobotType.RV6S:
                self.robot.close_connection()
            else:
                self.robot.close()
