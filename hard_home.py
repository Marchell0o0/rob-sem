from src.robot_box import RobotBox
from src.enums import RobotType

box = RobotBox(RobotType.CRS93)
box.robot.initialize()
box.gripper.open()