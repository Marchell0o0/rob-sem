from src.robot_box import RobotBox
from src.enums import RobotType
import argparse
from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.scene3d import Scene3D
import time
parser = argparse.ArgumentParser()
parser.add_argument("--robot-type", type=str, default="CRS97")
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


# for cfg in box.calibration_aruco_configurations:
#     box.robot.move_to_q()

# exit()
scene_base = Scene3D().z_from_zero()
scene_camera = Scene3D().z_from_zero()


# camera_to_base = box.get_camera_to_base_transform()

# exit()

# # Load and fix the camera-to-base transform
camera_to_base_matrix = np.load("calibration/calibration_data/camera_to_base.npy")
gripper_to_flange_matrix = np.load("calibration/calibration_data/gripper_to_flange.npy")

camera_to_base = SE3().from_matrix(camera_to_base_matrix, "meters")
gripper_to_flange = SE3().from_matrix(gripper_to_flange_matrix, "meters")


if not camera_to_base or not gripper_to_flange:
    print("Camera to base or gripper to flange transform not found")
    exit()
print("Camera to base: ", camera_to_base)
print("Gripper to flange: ", gripper_to_flange)

scene_camera.add_transform("Camera", SE3())
scene_camera.add_transform("Base", camera_to_base)
scene_base.add_transform("Base", SE3())
rotate_by_z = SE3(rotation=SO3().from_euler_angles(np.deg2rad([0, 0, 90]), ["x", "y", "z"]))
print("rotation by z 90 degrees", rotate_by_z)
base_to_camera = rotate_by_z * camera_to_base
scene_base.add_transform("Camera", base_to_camera )
# scene_base.add_robot(box, box.robot.get_q())


flange = SE3().from_matrix(box.robot.fk(box.robot.get_q()), "meters")
print("Flange: ", flange)
scene_base.add_transform("Flange", flange)


# gripper = flange * gripper_to_flange.inverse()
# print("Gripper: ", gripper)
# scene_base.add_transform("Gripper", gripper)

# flange_to_gripper = flange.inverse() * gripper
# print("Flange to gripper: ", flange_to_gripper)
# # scene_base.add_transform("Flange to gripper", flange_to_gripper)



boards = box.find_boards()
for board in boards:
    print(board.board_transform)
    scene_camera.add_transform(f"Board {board.pair}", board.board_transform)
    scene_base.add_transform(f"Board {board.pair}", base_to_camera * board.board_transform)
    for slot_idx, slot_transform in board.slot_transforms:
        scene_camera.add_transform(f"Slot {slot_idx}, board {board.pair}", slot_transform)
        scene_base.add_transform(f"Slot {slot_idx}, board {board.pair}", base_to_camera * slot_transform)

if len(boards) != 2:
    print("Didn't find 2 boards")
    exit()


source_board = None
destination_board = None
for board in boards:
    if board.empty:
        destination_board = board
    else:
        source_board = board

path = []

cube = source_board.slot_transforms[0][1]
cube_gripper_destination = base_to_camera * cube
cube_gripper_destination *= SE3(translation=[0, 0, -100])
cube_flange_destination = gripper_to_flange * cube_gripper_destination

scene_base.add_transform("Gripper dest", cube_gripper_destination)
scene_base.add_transform("Flange dest", cube_flange_destination)

exit()

configurations = box.robot.ik(cube_gripper_destination.to_matrix())
print(configurations)
print(box.robot.fk(configurations[0]))
for config in configurations:
    try: 
        box.robot.move_to_q(config)
        box.robot.wait_for_motion_stop()
        break
    except:
        pass
# scene_camera.display()
scene_base.display()
exit()

path.append(("move", cube_flange_destination * SE3(translation=[0, 0, -100])))
path.append(("move", cube_flange_destination * SE3(translation=[0, 0, -10])))
path.append(("close", None))
path.append(("move", cube_flange_destination * SE3(translation=[0, 0, -100])))

slot = destination_board.slots[1][1]
slot_gripper_destination = camera_to_base.inverse() * slot
slot_flange_destination = flange_to_gripper * slot_gripper_destination

path.append(("move", slot_flange_destination * SE3(translation=[0, 0, -100])))
path.append(("move", slot_flange_destination * SE3(translation=[0, 0, -80])))
path.append(("open", None))
path.append(("move", slot_flange_destination * SE3(translation=[0, 0, -100])))

box.robot.soft_home()

previous_configuration = box.robot.get_q()
for action, data in path:
    if action == "move":
        configurations = box.robot.ik(data.to_matrix())
        # print("Configurations: ", configurations)
        closest_configuration = None
        closest_distance = float('inf')
        for configuration in configurations:
            distance = abs(configuration[5] - previous_configuration[5])
            if distance < closest_distance:
                closest_distance = distance
                closest_configuration = configuration
        
        print("\nConfiguration: ", np.rad2deg(closest_configuration).round())
        user_input = input("\nPress 'y' to move robot to this configuration (any other key to skip): ")
        if user_input.lower() != 'y':
            exit()        

        try:
            box.robot.move_to_q(closest_configuration)
            box.robot.wait_for_motion_stop()
            previous_configuration = closest_configuration
        except Exception as e:
            print(f"Movement failed: {e}")
            exit()

    elif action == "close":
        user_input = input("\nPress 'y' to close gripper: ")
        if user_input.lower() != 'y':
            exit()
        box.gripper.close()
        time.sleep(2)
    elif action == "open":
        user_input = input("\nPress 'y' to open gripper: ")
        if user_input.lower() != 'y':
            exit()
        box.gripper.open()
        time.sleep(2)

box.robot.soft_home()
box.robot.wait_for_motion_stop()

box.close()