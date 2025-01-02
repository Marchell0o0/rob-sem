import numpy as np
import os
from src.basler_camera import BaslerCamera
from src.enums import RobotType
from src.camera_image import CameraImage


class Camera():
    def __init__(self, robot_type: RobotType):
        self.camera = BaslerCamera()

        if robot_type == RobotType.CRS93:
            self.camera.connect_by_ip("192.168.137.107")
            self.camera.connect_by_name("camera-crs93")
        elif robot_type == RobotType.CRS97:
            self.camera.connect_by_ip("192.168.137.106")
            self.camera.connect_by_name("camera-crs97")
        elif robot_type == RobotType.RV6S:
            self.camera.connect_by_ip("192.168.137.109")
            self.camera.connect_by_name("camera-rv6s")

        self.camera.open()
        self.camera.set_parameters()

        self.camera_matrix = None
        if os.path.exists("calibration/calibration_data/camera_matrix.npy"):
            self.camera_matrix = np.load(
                "calibration/calibration_data/camera_matrix.npy")
        self.dist_coeffs = None
        if os.path.exists("calibration/calibration_data/dist_coeffs.npy"):
            self.dist_coeffs = np.load(
                "calibration/calibration_data/dist_coeffs.npy")

    def grab_image(self):
        return CameraImage(self.camera_matrix, self.dist_coeffs).set_image(self.camera.grab_image())

    def close(self):
        self.camera.close()
