from robot_box import RobotBox, RobotType
import numpy as np

box = RobotBox(RobotType.CRS93, True)

box.robot.soft_home()
q = box.robot.get_q()
q += np.deg2rad([90, 0, 0, 0, 0, 0])
box.robot.move_to_q(q)
box.robot.wait_for_motion_stop()

np.save("fk.npy", box.robot.fk(q))
# box.robot.close()
box.robot.close()