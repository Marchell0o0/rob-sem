from src.robot_box import RobotBox
from src.enums import RobotType
import argparse
from src.se3 import SE3
from src.so3 import SO3
import numpy as np
from src.scene3d import Scene3D
import time
parser = argparse.ArgumentParser()
parser.add_argument("--robot-type", type=str, default="CRS93")
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
scene_camera = Scene3D().z_from_zero()

# camera_to_base = box.get_camera_to_base_transform()
# exit()


# # Load and fix the camera-to-base transform
camera_to_base_matrix = np.load("calibration/calibration_data/camera_to_base.npy")
gripper_to_flange_matrix = np.load("calibration/calibration_data/gripper_to_flange.npy")

print("Camera to base new: ", camera_to_base_matrix)
print("Gripper to flange new: ", gripper_to_flange_matrix)

camera_to_base_matrix += np.load("calibration/calibration_data/working_camera_to_base.npy")
gripper_to_flange_matrix += np.load("calibration/calibration_data/working_gripper_to_flange.npy")


camera_to_base_matrix = camera_to_base_matrix / 2
gripper_to_flange_matrix = gripper_to_flange_matrix / 2

offset = SE3(rotation=SO3.from_euler_angles(np.deg2rad(np.array([0, 0, 3])), ["x", "y", "z"]), translation=[0, 0, 0])

camera_to_base = SE3().from_matrix(camera_to_base_matrix, "meters")
camera_to_base = camera_to_base * offset
gripper_to_flange = SE3().from_matrix(gripper_to_flange_matrix, "meters")


if not camera_to_base or not gripper_to_flange:
    print("Camera to base or gripper to flange transform not found")
    exit()
print("Camera to base: ", camera_to_base)
print("Gripper to flange: ", gripper_to_flange)

# scene_camera.add_transform("Camera", SE3())
# scene_camera.add_transform("Base", camera_to_base)

scene_base.add_transform("Base", SE3())
scene_base.add_transform("Camera", camera_to_base)

# scene_base.add_robot(box, box.robot.get_q())


flange = SE3().from_matrix(box.robot.fk(box.robot.get_q()), "meters")
print("Flange: ", flange)
scene_base.add_transform("Flange", flange)

aruco = flange * gripper_to_flange
print("Gripper: ", aruco)
scene_base.add_transform("Gripper", aruco)



boards = box.find_boards()
for board in boards:
    print(board.board_transform)
    scene_base.add_transform(f"Board {board.pair}", camera_to_base * board.board_transform)
    for slot_idx, slot_transform in board.slot_transforms:
        scene_base.add_transform(f"Slot {slot_idx}, board {board.pair}", camera_to_base * slot_transform)

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


box.robot.soft_home()
box.gripper.open()

camera_to_base=camera_to_base * SE3(translation=[0, -7, 0]) 

for ((slot_idx, slot), (cube_idx, cube)) in zip(destination_board.slot_transforms, source_board.slot_transforms):
    cube_gripper_destination = camera_to_base * cube 

    path = []
    path.append(("move", cube_gripper_destination * SE3(translation=[0, 0, -140])))
    path.append(("move", cube_gripper_destination * SE3(translation=[0, 0, -40])))
    path.append(("close", None))
    path.append(("move", cube_gripper_destination * SE3(translation=[0, 0, -140])))

    slot_gripper_destination = camera_to_base * slot

    path.append(("move", slot_gripper_destination * SE3(translation=[0, 0, -140])))
    path.append(("move", slot_gripper_destination * SE3(translation=[0, 0, -40])))
    path.append(("open", None))
    path.append(("move", slot_gripper_destination * SE3(translation=[0, 0, -140])))


    previous_configuration = box.robot.get_q()

    def sort_configurations_by_distance(configurations, previous_config):
        """Sort configurations by their distance to the previous configuration."""
        return sorted(configurations, key=lambda config: np.linalg.norm(config - previous_config))

    for action, data in path:
        if action == "move":
            configurations = box.robot.ik(data.to_matrix())
            # Sort configurations by distance to previous configuration
            configurations = sort_configurations_by_distance(configurations, previous_configuration)

            if not configurations:
                print("No configurations found")
                exit()

            print("Most likely configuration: ", np.rad2deg(configurations[0]).round())

            success = False
            for configuration in configurations:
                configuration_transformation = SE3().from_matrix(box.robot.fk(configuration), "meters")

                if configuration_transformation.translation[2] < 0:
                    print("JEDU V STOL")
                    exit()

                try:
                    box.robot.move_to_q(configuration)
                    box.robot.wait_for_motion_stop()
                    previous_configuration = configuration  # Update previous configuration after successful move
                    success = True
                    break
                except Exception as e:
                    print(f"Movement failed: {e}")
            
            if not success:
                print("Failed to move to any configuration")

        elif action == "close":
            box.gripper.close()
            time.sleep(2)
        elif action == "open":
            box.gripper.open()
            time.sleep(2)

box.robot.soft_home()
box.robot.wait_for_motion_stop()

box.close()