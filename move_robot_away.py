import numpy as np
from ctu_mitsubishi import Rv6s, Rv6sGripper
from se3 import SE3
from so3 import SO3
from time import sleep
robot = Rv6s(debug=False)
robot.initialize()

q = robot.get_q()
# print(np.rad2deg(q).round())
robot.soft_home()
# q[0] += np.deg2rad(90)
# print(np.rad2deg(q).round())

# robot.move_to_q(q)
# print(f"Moved to {np.rad2deg(q).round()} degrees")

robot_pos = robot.fk(q)
print("robot pos: ", robot_pos)
robot_pos = SE3(rotation = SO3(robot_pos[:3, :3]), translation = robot_pos[:3, 3] * 1000)

robot_gripper_offset = SE3(
    rotation=SO3.from_euler_angles(np.deg2rad([0, 360, 0]), ["x", "y", "z"]),
    translation=np.array([0, 0, 200])
)

# board_pos = SE3(rotation = SO3(robot_pos.rotation.rot), translation = np.array([500, 0, 45]))
final_rotation = SO3.from_euler_angles(np.deg2rad([0, 90, 0]), ["x", "y", "z"]) * SO3(robot_pos.rotation.rot)
board_pos = SE3(rotation = final_rotation, translation = np.array([500, 0, 500]))
print("board pos: ", board_pos)

board_with_gripper = robot_gripper_offset * board_pos
print("board with gripper: ", board_with_gripper)

full_matrix = np.eye(4)
full_matrix[:3, :3] = board_with_gripper.rotation.rot
full_matrix[:3, 3] = board_with_gripper.translation / 1000

print("full matrix: ", full_matrix)
ik = robot.ik(full_matrix)
print("ik: ", ik)
robot.move_to_q(ik[0])

gripper = Rv6sGripper()

robot.stop_robot()


# robot home pos 
# robot pos:  [[-9.99999985e-01  1.74532924e-04 -2.22044605e-16  4.00083757e-01]
#  [ 1.74532924e-04  9.99999985e-01  1.22464680e-16  6.43612918e-18]
#  [ 2.22065976e-16  1.22425924e-16 -1.00000000e+00  6.44890034e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


gripper.open()
sleep(2)
gripper.close()
sleep(2)
robot.stop_robot()
robot.close_connection()


# from time import sleep

# from ctu_mitsubishi import Rv6sGripper

# gripper = Rv6sGripper()

# for _ in range(5):
#     gripper.open()
#     sleep(1)
#     gripper.close()
#     sleep(1)


# gripper.disconnect()