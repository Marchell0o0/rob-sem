from robot_box import RobotBox, RobotType

box = RobotBox(RobotType.CRS97, True)

box.get_camera_to_base_transform()

box.close()
