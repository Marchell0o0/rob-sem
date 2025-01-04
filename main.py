from src.robot_box import RobotBox
from src.enums import RobotType
import argparse
from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.scene3d import Scene3D
import time
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

# # Load and fix the camera-to-base transform
camera_to_base_matrix = np.load("calibration/calibration_data/camera_to_base.npy")
aruco_to_flange_matrix = np.load("calibration/calibration_data/gripper_to_flange.npy")

camera_to_base = SE3().from_matrix(camera_to_base_matrix, "meters")
aruco_to_flange = SE3().from_matrix(aruco_to_flange_matrix, "meters")

if not camera_to_base or not aruco_to_flange:
    print("Camera to base or gripper to flange transform not found")
    exit()
print("Camera to base: ", camera_to_base)
print("Gripper to flange: ", aruco_to_flange)


scene_base.add_transform("Base", SE3())
scene_base.add_transform("Camera", camera_to_base.inverse())
scene_base.add_robot(box, box.robot.get_q())

flange = SE3().from_matrix(box.robot.fk(box.robot.get_q()), "meters")
print("Flange: ", flange)
aruco = flange * aruco_to_flange.inverse()
print("Aruco: ", aruco)
scene_base.add_transform("Aruco", aruco)

aruco_to_gripper = SE3(translation=[90, -20, 10], rotation=SO3.from_euler_angles(np.deg2rad([0, 90, -90]), ["x", "y", "z"]))
gripper = aruco * aruco_to_gripper
print("Gripper: ", gripper)
scene_base.add_transform("Gripper", gripper)

flange_to_gripper = flange.inverse() * gripper
print("Flange to gripper: ", flange_to_gripper)
# scene_base.add_transform("Flange to gripper", flange_to_gripper)

boards = box.find_boards()
if len(boards) != 2:
    print("Didn't find 2 boards")
    exit()

# # scene.add_robot(box, box.robot.get_q())
box.robot.soft_home()
scene_configurations = Scene3D().z_from_zero()
source_board = None
destination_board = None
for board in boards:
    # scene.add_board(board)
    if board.empty:
        destination_board = board
    else:
        source_board = board

    for idx, slot in enumerate(board.slots):
        scene_base.add_transform(f"Slot {idx}, board {board.pair}", camera_to_base.inverse() * slot[1])
#         scene_camera.add_transform(f"Slot {idx}, board {board.pair}", slot[1])
    scene_base.add_transform(f"Board {board.pair}, aruco {board.ref_marker_id}", camera_to_base.inverse() * board.ref_marker_transform)
    scene_base.add_transform(f"Board {board.pair}, aruco {board.second_marker_id}", camera_to_base.inverse() * board.second_marker_transform)
    
    # if not board.empty:
    #     path = []

    #     destination_in_camera = board.slots[2][1]

    #     destination_in_base = camera_to_base.inverse() * destination_in_camera
    #     scene_base.add_transform("Destination", destination_in_base)

    #     flange_destination = flange_to_gripper * destination_in_base
    #     scene_base.add_transform("Flange destination", flange_destination)

    #     path.append(flange_destination * SE3(translation=[0, 0, -100]))
    #     path.append(flange_destination * SE3(translation=[0, 0, -10]))
    #     path.append(flange_destination * SE3(translation=[0, 0, -100]))

    #     previous_configuration = None

    #     for checkpoint in path:
    #         configurations = box.robot.ik(checkpoint.to_matrix())
    #         print("Configurations: ", configurations)
    #         for configuration in configurations:
    #             if previous_configuration is not None:
    #                 if np.allclose(previous_configuration[5], configuration[5] + 2 * np.pi) or np.allclose(previous_configuration[5], configuration[5] - 2 * np.pi):
    #                     configuration[5] = previous_configuration[5]
    #                     print("Using last joint rotation from previous configuration")
    #                     continue

    #             print("\nConfiguration: ", np.rad2deg(configuration).round())
                
    #             user_input = input("\nPress 'y' to move robot to this configuration (any other key to skip): ")
    #             if user_input.lower() != 'y':
    #                 print("Skipping this configuration")
    #                 continue
                
    #             try:
    #                 box.robot.move_to_q(configuration)
    #                 box.robot.wait_for_motion_stop()
    #                 previous_configuration = configuration
    #                 break
    #             except Exception as e:
    #                 print(f"Movement failed: {e}")
    #                 continue
scene_base.display()
exit()
path = []

cube = source_board.slots[0][1]
cube_gripper_destination = camera_to_base.inverse() * cube
cube_flange_destination = flange_to_gripper * cube_gripper_destination

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