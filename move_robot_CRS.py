from ctu_crs import CRS97 # or CRS97
import numpy as np

robot = CRS97()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot, perform hard and soft home
# q = robot.get_q()
# print("fk: ", robot.fk(q))

q = robot.get_q()  # get current joint configuration
robot.move_to_q(q + np.deg2rad([90, 0, 0, 0, 0, 0]))  # move robot all values in radians
robot.wait_for_motion_stop() # wait until the robot stops
robot.close()  # close the connection
