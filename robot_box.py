from se3 import SE3
from so3 import SO3
from ctu_crs import CRS97
import numpy as np
from camera import Camera
from enums import RobotType
import cv2

class RobotBox():
    def __init__(self, robot_type: RobotType, robot_active: bool = True):
        if robot_active:
            self.robot = CRS97()
            self.robot.initialize()
        else:
            self.robot = None
        self.camera = Camera(robot_type)
        self.gripper_to_aruco = SE3(translation=[70, 33, 0])
    def get_camera_to_base_transform(self):
        if self.robot is not None:
            self.robot.soft_home()

            q = self.robot.get_q()

            self.robot.move_to_q(q + np.deg2rad([0, 0, -70, 0, -20, 0]))

            self.robot.wait_for_motion_stop()

        img = self.camera.grab_image()

        arucos = self.camera.get_arucos(img, 45, cv2.aruco.DICT_4X4_50)

        img = self.camera.draw_arucos(img, arucos)
        self.camera.display_image(img)

        gripper = arucos[1] * self.gripper_to_aruco.inverse()
        transforms = arucos
        transforms["Gripper"] = gripper

        if self.robot:
            q = self.robot.get_q()

            gripper_in_base = SE3().from_matrix(self.robot.fk(q))
            
            base = gripper * gripper_in_base.inverse()

            transforms["Base"] = base

        self.camera.display_transforms_3d(transforms)

        print(arucos)
        



    def close(self):
        if self.robot:
            self.robot.wait_for_motion_stop()
            self.robot.close()
