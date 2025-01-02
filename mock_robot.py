import numpy as np
from enum import Enum


class RobotType(Enum):
    RV6S = "RV6S"


class MockRv6s:
    """Mock version of Rv6s class for visualization testing."""

    def __init__(self):
        self.q_home = np.deg2rad([0, 0, 90, 0, 90, 0])

        # DH parameters for RV-6S robot
        self.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])
        self.dh_a = np.array([85, 280, 100, 0, 0, 0]) / 1000.0
        self.dh_d = np.array([350, 0, 0, 315, 0, 85]) / 1000.0
        self.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])


class MockRobotBox:
    """Mock version of RobotBox class for visualization testing."""

    def __init__(self, robot_type: RobotType):
        if robot_type != RobotType.RV6S:
            raise ValueError(
                "Only RV6S robot type is supported in mock version")
        self.robot = MockRv6s()
