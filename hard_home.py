from src.robot_box import RobotBox
from src.enums import RobotType

box = RobotBox(RobotType.CRS97)
box.robot.initialize()