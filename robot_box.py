from se3 import SE3
from so3 import SO3
from ctu_crs import CRS97
import numpy as np
from camera import Camera
from enums import RobotType
import cv2

from board import Board

class RobotBox():
    def __init__(self, robot_type: RobotType, robot_active: bool = True):
        if robot_active:
            self.robot = CRS97()
            self.robot.initialize()
        else:
            self.robot = None
        self.camera = Camera(robot_type)

        self.calibration_aruco_offset = np.deg2rad([0, 10, -70, 0, 60, 176])
        # self.calibration_aruco_offset = np.deg2rad([0, 0, -70, 0, 70, 180])

        # CRS97
        # self.gripper_to_aruco = SE3(translation=[70, 33, 0])
        # CRS93
        self.gripper_to_aruco =  \
              SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 0, 5]), ["x", "y", "z"])) * \
              SE3(translation=[0, -70, -20]) * \
              SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles([np.pi/2, 0, np.pi/2], ["x", "y", "z"])) * \
              SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([-5, 0, 0]), ["x", "y", "z"]))
    
        self.BOARD_ARUCO_SIZE = 36
        self.BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50

        self.CALIBRATION_ARUCO_ID = 2
        self.CALIBRATION_ARUCO_SIZE = 38
        self.CALIBRATION_ARUCO_DICT = cv2.aruco.DICT_6X6_50

    def get_camera_to_base_transform(self):
        """
        Get the transform from the camera to the base of the robot

        returns:
            SE3: the transform from the camera to the base of the robot
        """
        if self.robot is not None:
            self.robot.soft_home()

            q = self.robot.get_q()
            print("Home q: ", q)
            self.robot.move_to_q(q + self.calibration_aruco_offset)
            
            self.robot.wait_for_motion_stop()

        img = self.camera.grab_image()
        arucos = self.camera.get_arucos(img, self.CALIBRATION_ARUCO_SIZE, self.CALIBRATION_ARUCO_DICT)
        print("Calibration ArUco: ", arucos)

        transforms = arucos

        gripper = None
        if self.CALIBRATION_ARUCO_ID in arucos:
            gripper = arucos[self.CALIBRATION_ARUCO_ID] * self.gripper_to_aruco
        transforms["Gripper"] = gripper
        print("Gripper in camera: ", gripper)


        base = None
        if self.robot:
            q = self.robot.get_q()
            fk = self.robot.fk(q)
            gripper_in_base = SE3().from_matrix(fk, "meters")
            print("Gripper in base: ", gripper_in_base)
            # T_CB = T_CG * T_BG^-1
            # T_CB = T_CG * T_GB
            base = gripper * gripper_in_base.inverse()
            # base.translation[1] = 0
            transforms["Base"] = base

            print("camera to base: ", base.inverse())


        our_camera_in_base = SE3(translation=[450, 0, 1170]) * \
                             SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 180, 90]), ["x", "y", "z"]))
        
        

        # gripper_to_our_base = our_camera_in_base * gripper_in_base

        # print("gripper to our base: ", gripper_to_our_base)
        # transforms["Gripper to our base"] = gripper_to_our_base
        # img = self.camera.draw_arucos(img, arucos)

        # img = self.camera.draw_transform(img, transforms["Gripper"], "Gripper")
        # if base is not None:
            # img = self.camera.draw_transform(img, base * gripper_in_base, "gripper from base")
        # img = self.camera.draw_transform(img, gripper_to_our_base, "gripper to our base")

        # self.camera.display_image(img)

        # self.camera.display_transforms_3d(transforms)


        return base, our_camera_in_base.inverse()
        
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
            self.robot.close()
