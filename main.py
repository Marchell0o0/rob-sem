from src.robot_box import RobotBox
from src.enums import RobotType
import argparse
from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.scene3d import Scene3D

parser = argparse.ArgumentParser()
parser.add_argument("--robot-type", type=str, default="RV6S")
parser.add_argument("--robot-active", action="store_true",
                    help="Enable robot activity")
parser.add_argument("--robot-inactive", action="store_false", dest="robot_active",
                    help="Disable robot activity")
parser.add_argument("--camera-active", action="store_true",
                    help="Enable camera activity")
parser.add_argument("--camera-inactive", action="store_false", dest="camera_active",
                    help="Disable camera activity")
parser.set_defaults(robot_active=True, camera_active=True)
args = parser.parse_args()

box = RobotBox(RobotType[args.robot_type],
               args.robot_active, args.camera_active)

scene_base = Scene3D().z_from_zero()
scene_camera = Scene3D().z_from_zero().invert_z_axis()

# scene.add_transform("Camera", SE3())
# camera_to_base = box.get_camera_to_base_transform()
# Load and fix the camera-to-base transform
camera_to_base_matrix = np.load("calibration/calibration_data/camera_to_base.npy")
# Invert Y axis by negating the Y column and row
print("Matrix original: ", camera_to_base_matrix)
# camera_to_base_matrix[0:3, 1] *= -1  # Y column
# camera_to_base_matrix[1, 0:3] *= -1  # Y row
# temp = camera_to_base_matrix[0, 3] 
# camera_to_base_matrix[0,3] = camera_to_base_matrix[1, 3]
# camera_to_base_matrix[1,3] = -temp
# print("Matrix fixed: ", camera_to_base_matrix)
camera_to_base = SE3().from_matrix(camera_to_base_matrix, "meters")

if not camera_to_base:
    print("Camera to base transform not found")
    exit()

scene_base.add_transform("Base", SE3())
scene_base.add_transform("Camera", camera_to_base.inverse())
scene_base.add_robot(box, box.robot.get_q())

scene_camera.add_transform("Base", camera_to_base)
scene_camera.add_transform("Camera", SE3())

boards = box.find_boards()
if len(boards) != 2:
    print("Didn't find 2 boards")
    exit()

# scene.add_robot(box, box.robot.get_q())

scene_configurations = Scene3D().z_from_zero()
for board in boards:
    # scene.add_board(board)
    for idx, slot in enumerate(board.slots):
        scene_base.add_transform(f"Slot {idx}, board {board.pair}", camera_to_base.inverse() * slot[1])
        scene_camera.add_transform(f"Slot {idx}, board {board.pair}", slot[1])

    if board.empty:
        destination_in_camera = board.slots[0][1]

        destination_in_base = camera_to_base.inverse() * destination_in_camera
        destination_in_base = destination_in_base * \
            SE3(translation=[0, 0, -300]) * \
            SE3(translation=[0, 0, 0], rotation=SO3.from_euler_angles(np.deg2rad([0, 0, -90]), ["x", "y", "z"]))
        scene_base.add_transform("Destination", destination_in_base)
        scene_camera.add_transform("Destination", destination_in_camera)

        print("Destination in base: ", destination_in_base)
        print("Destination matrix: ", destination_in_base.to_matrix())
        configurations = box.robot.ik(destination_in_base.to_matrix())
        print("Configurations: ", configurations)
        success = False
        for configuration in configurations:
            scene_configurations.add_robot(box, configuration)

            scene_configurations.display()
            print("Configuration: ", configuration)
            try:
                box.robot.move_to_q(configuration)
                box.robot.wait_for_motion_stop()
                scene_base.add_robot(box, configuration)
                success = True
            except Exception as e:
                print(e)
            if success:
                break

scene_base.display()
scene_camera.display()


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
