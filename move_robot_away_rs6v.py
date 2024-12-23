import numpy as np
from ctu_mitsubishi import Rv6s, Rv6sGripper
from se3 import SE3
from so3 import SO3
from time import sleep
from robot_box import RobotBox
from enums import RobotType

box = RobotBox(robot_type=RobotType.RV6S)

# DH parameters for RV-6S robot
# box.robot.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])  # Joint angle offset
# box.robot.dh_a = np.array([85, 280, 100, 0, 0, 0]) / 1000.0    # Link length
# box.robot.dh_d = np.array([350, 0, 0, 315 + 20, 0, 85 + 100]) / 1000.0    # Link offset
# box.robot.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])     # Link twist

box.robot.dh_theta_off = np.deg2rad([0, -90, -90, 0, 0, 180])  # Joint angle offset
box.robot.dh_a = np.array([85, 280, 0, 0, 0, 0]) / 1000.0    # Link length
box.robot.dh_d = np.array([330, 0, 100, 80, 230, 230]) / 1000.0    # Link offset
box.robot.dh_alpha = np.deg2rad([-90, 0, -90, 90, -90, 0])     # Link twist


# box.robot.soft_home()
# box.robot.wait_for_motion_stop()

home_q = np.deg2rad([0, 0, 90, 0, 90, 0])

# box.robot.move_to_q(home_q)
# box.robot.wait_for_motion_stop()
print("Home q: ", np.rad2deg(home_q).round())

end_effector = SE3.from_matrix(box.robot.fk(home_q))

print("End effector: ", end_effector.translation.round(4))

# q = box.robot.get_q()
straight_q = np.deg2rad([-90, 0, 0, 0, 0, 0])
box.robot.move_to_q(straight_q)
box.robot.wait_for_motion_stop()
print("Straight q: ", np.rad2deg(straight_q).round())

end_effector = SE3.from_matrix(box.robot.fk(straight_q))

print("End effector: ", end_effector.translation.round(4))


calibration_q = np.deg2rad([0, 30, 130, 0, -70, -90])
# box.robot.move_to_q(calibration_q)
# box.robot.wait_for_motion_stop()

# q = box.robot.get_q()
print("Calibration q: ", np.rad2deg(calibration_q).round())
end_effector = SE3.from_matrix(box.robot.fk(calibration_q).round(4))

print("End effector: ", end_effector.translation)
exit()

# gripper_offset = SE3(translation=[-20, 0, 160])
transforms = {}

transforms["Base"] = SE3()

q = box.robot.get_q()
print("Home q: ", np.rad2deg(q).round())
end_effector = SE3.from_matrix(box.robot.fk(q))
gripper = end_effector * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi) - q[5]], ["x", "y", "z"])) * gripper_offset * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi) + q[5]], ["x", "y", "z"]))

transforms["end effector home"] = end_effector
transforms["gripper home"] = gripper
print("End effector: ", end_effector.translation)
print("Gripper in soft home: ", gripper.translation)

q = box.robot.get_q()
q[5] = np.deg2rad(90)
box.robot.move_to_q(q)
box.robot.wait_for_motion_stop()

q = box.robot.get_q()
print("Turned 90 degrees q: ", np.rad2deg(q).round())
end_effector = SE3.from_matrix(box.robot.fk(q))
gripper = end_effector * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi ) - q[5]], ["x", "y", "z"])) * gripper_offset * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi)  + q[5]], ["x", "y", "z"]))

print("End effector: ", end_effector.translation)
print("Gripper rotated 90 degrees: ", gripper.translation)

transforms["end effector rotated 90 degrees"] = end_effector
transforms["gripper rotated 90 degrees"] = gripper

box.robot.move_to_q(np.deg2rad([0, 30, 130, 0, -70, -90]))
box.robot.wait_for_motion_stop()

q = box.robot.get_q()
print("Calibration q: ", np.rad2deg(q).round())
end_effector = SE3.from_matrix(box.robot.fk(q))
gripper = end_effector * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi ) - q[5]], ["x", "y", "z"])) * gripper_offset * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi)  + q[5]], ["x", "y", "z"]))

print("End effector: ", end_effector.translation)
print("Calibration: ", gripper.translation)


transforms["end effector calibration"] = end_effector
transforms["gripper calibration"] = gripper

box.robot.move_to_q(np.deg2rad([0, 30, 130, 0, -70, 0]))
box.robot.wait_for_motion_stop()

q = box.robot.get_q()
print("Calibration rotated 90 degrees q: ", np.rad2deg(q).round())
end_effector = SE3.from_matrix(box.robot.fk(q))
gripper = end_effector * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi)  - q[5]], ["x", "y", "z"])) * gripper_offset * SE3(rotation=SO3.from_euler_angles([0, 0, (np.pi)  + q[5]], ["x", "y", "z"]))

print("End effector: ", end_effector.translation)
print("Calibration rotated 90 degrees: ", gripper.translation)


transforms["end effector calibration rotated 90 degrees"] = end_effector
transforms["gripper calibration rotated 90 degrees"] = gripper



box.camera.display_transforms_3d(transforms, False)
box.close()

# q = robot.get_q()
# q[5] = np.deg2rad(-90)
# robot.move_to_q(q)
# robot.wait_for_motion_stop()
# # gripper_offset = SE3(translation=[np.cos(q[5]) * 20, np.sin(q[5]) * -20, 160])

# q = robot.get_q()
# end_effector = SE3.from_matrix(robot.fk(q))
# # gripper = end_effector * gripper_offset.inverse()
# gripper = SE3(translation=end_effector.translation - gripper_offset.translation, rotation=end_effector.rotation)


# print("End effector: ", end_effector.translation)
# print("Gripper offset: ", gripper_offset.translation)
# print("Gripper rotated -90 degrees: ", gripper.translation)


# q = robot.get_q()
# q[5] = np.deg2rad(180)
# robot.move_to_q(q)
# robot.wait_for_motion_stop()
# # gripper_offset = SE3(translation=[np.cos(q[5]) * 20, np.sin(q[5]) * -20, 160])

# q = robot.get_q()
# end_effector = SE3.from_matrix(robot.fk(q))
# # gripper = end_effector * gripper_offset.inverse()
# gripper = SE3(translation=end_effector.translation - gripper_offset.translation, rotation=end_effector.rotation)

# print("End effector: ", end_effector.translation)
# print("Gripper offset: ", gripper_offset.translation)
# print("Gripper rotated 180 degrees: ", gripper.translation)



# print(np.rad2deg(q).round())

# robot.move_to_q(q)
# print(f"Moved to {np.rad2deg(q).round()} degrees")

# robot_pos = robot.fk(q)
# print("robot pos: ", robot_pos)
# robot_pos = SE3(rotation = SO3(robot_pos[:3, :3]), translation = robot_pos[:3, 3] * 1000)

# robot_gripper_offset = SE3(
#     rotation=SO3.from_euler_angles(np.deg2rad([0, 360, 0]), ["x", "y", "z"]),
#     translation=np.array([0, 0, 200])
# )

# board_pos = SE3(rotation = SO3(robot_pos.rotation.rot), translation = np.array([500, 0, 45]))
# final_rotation = SO3.from_euler_angles(np.deg2rad([0, 90, 0]), ["x", "y", "z"]) * SO3(robot_pos.rotation.rot)
# board_pos = SE3(rotation = final_rotation, translation = np.array([500, 0, 500]))
# print("board pos: ", board_pos)

# board_with_gripper = robot_gripper_offset * board_pos
# print("board with gripper: ", board_with_gripper)

# full_matrix = np.eye(4)
# full_matrix[:3, :3] = board_with_gripper.rotation.rot
# full_matrix[:3, 3] = board_with_gripper.translation / 1000

# print("full matrix: ", full_matrix)
# ik = robot.ik(full_matrix)
# print("ik: ", ik)
# robot.move_to_q(ik[0])

# gripper = Rv6sGripper()

# robot.stop_robot()


# robot home pos 
# robot pos:  [[-9.99999985e-01  1.74532924e-04 -2.22044605e-16  4.00083757e-01]
#  [ 1.74532924e-04  9.99999985e-01  1.22464680e-16  6.43612918e-18]
#  [ 2.22065976e-16  1.22425924e-16 -1.00000000e+00  6.44890034e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


# gripper.open()
# sleep(2)
# gripper.close()
# sleep(2)

# from time import sleep

# from ctu_mitsubishi import Rv6sGripper

# gripper = Rv6sGripper()

# for _ in range(5):
#     gripper.open()
#     sleep(1)
#     gripper.close()
#     sleep(1)


# gripper.disconnect()