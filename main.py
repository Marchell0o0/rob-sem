from robot_box import RobotBox, RobotType
import argparse
from se3 import SE3
from so3 import SO3
import numpy as np

from ctu_crs import CRS93

parser = argparse.ArgumentParser()
parser.add_argument("--robot-type", type=str, default="RV6S")
parser.add_argument("--robot-active", action="store_true", help="Enable robot activity")
parser.add_argument("--robot-inactive", action="store_false", dest="robot_active", 
                    help="Disable robot activity")
parser.set_defaults(robot_active=True)
args = parser.parse_args()

box = RobotBox(RobotType[args.robot_type], args.robot_active)

camera_to_base = box.get_camera_to_base_transform()
# our_camera_to_base = camera_to_base.inverse()

transforms = {}

# transforms["Camera"] = camera_to_base
# transforms["Our camera"] = our_camera_to_base

# our_camera_in_base = SE3(translation=[450, 0, 1170]) * \
                        # SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 180, 90]), ["x", "y", "z"]))
# our_camera_to_base = our_camera_in_base.inverse()

# boards = box.find_boards()
# transforms = {f"Board {board.pair} Slot {slot[0]}": slot[1] for board in boards for slot in board.slots}

# if box.robot is not None:
#     q = box.robot.get_q()
#     robot = SE3().from_matrix(box.robot.fk(q))
# else: 
#     robot = SE3().from_matrix(np.load("fk.npy"), "meters")

# # transforms[f"Robot"] = our_camera_to_base * robot
# transforms[f"Robot"] = camera_to_base * robot

# transforms[f"Base"] = camera_to_base
# # transforms["Our base"] = our_camera_to_base

# goal_in_camera = boards[0].slots[0][1]

# # goal = our_camera_to_base.inverse() * goal_in_camera
# goal = camera_to_base.inverse() * goal_in_camera

# goal = goal * SE3(translation=[0, 0, -100])

# goal_matrix = goal.to_matrix("meters")
# print("goal matrix: ", goal_matrix)
# goal_q = CRS93().ik(goal_matrix)

# # transforms[f"Goal"] = our_camera_to_base * goal
# transforms[f"Goal"] = camera_to_base * goal

# print("Possible goal configurations: ", goal_q)
# if goal_q is not None:
#     success = False
#     for q in goal_q:
#         try:
#             box.robot.move_to_q(q)
#             box.robot.wait_for_motion_stop()
#             success = True
#         except Exception as e:
#             print(e)
#         if success:
#             break
#     if not success:
#         print("Failed to reach goal")
            

# box.camera.display_transforms_3d(transforms)

# # transforms_in_base = {k: our_camera_to_base.inverse() * v for k, v in transforms.items()}
# # box.camera.display_transforms_3d(transforms_in_base)

box.close()