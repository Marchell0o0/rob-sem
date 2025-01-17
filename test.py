import numpy as np
import cv2
from src.camera_image import CameraImage
from src.scene3d import Scene3D
from src.robot_box import RobotBox, RobotType
from src.se3 import SE3
from src.so3 import SO3
from pathlib import Path

# def create_path_points(start, end, interval=30):
#     """Create points along a line from start to end with given interval."""
#     # Calculate direction vector
#     direction = end - start
#     # Number of points needed
#     distance = np.linalg.norm(direction)
#     num_points = int(np.ceil(distance / interval))
#     # Create evenly spaced points
#     points = np.linspace(start, end, num_points)
#     return points

# def find_next_target(current_pos, path_points, lookahead_distance=70):
#     """Find the furthest point within lookahead distance on the path."""
#     # Calculate distances to all points
#     distances = np.linalg.norm(path_points - current_pos, axis=1)
    
#     # Find the closest point index
#     closest_idx = np.argmin(distances)
    
#     # Look ahead from the closest point
#     candidates = path_points[closest_idx:]
#     if len(candidates) == 0:
#         return None
        
#     # Among points ahead, find the furthest one within lookahead distance
#     distances_ahead = np.linalg.norm(candidates - current_pos, axis=1)
#     valid_indices = np.where(distances_ahead <= lookahead_distance)[0]
    
#     if len(valid_indices) == 0:
#         # If no points within lookahead, take the closest point ahead
#         return candidates[0]
    
#     # Return the furthest valid point
#     furthest_idx = valid_indices[-1]
#     return candidates[furthest_idx]

box = RobotBox(RobotType.CRS93)
scene = Scene3D().z_from_zero()

camera_to_base = box.get_camera_to_base_transform()

# img = box.camera.grab_image()
# img.display()
# img.save_image("images/boards2.png")
# box.robot.move_to_q(np.deg2rad([90, 0, -45, 0, -45, 0]))

# # Get initial configuration
# current_config = box.robot.get_q()

# # Define start and end points
# start_point = np.array([600, -150, 200])
# end_point = np.array([600, 150, 200])

# # Create path points
# path_points = create_path_points(start_point, end_point)
# print(path_points)
# # Create rotation matrix where:
# # x points down (negative z in world frame)
# # z points in x direction (positive x in world frame)
# # y will be automatically determined to maintain right-hand rule
# down_x = np.array([0, 0, -1])  # x axis points down
# forward_z = np.array([1, 0, 0])  # z axis points forward (in x direction)
# right_y = -np.cross(down_x, forward_z)  # y axis is determined by cross product

# # Create rotation matrix from these axes
# up_rotation = SO3(rotation_matrix=np.column_stack([down_x, right_y, forward_z]))

# # Create transforms for visualization
# for i, point in enumerate(path_points):
#     pose = SE3(
#         translation=point,
#         rotation=up_rotation
#     )
#     scene.add_transform(f"Path_{i}", pose)

# # First move to the starting point
# start_pose = SE3(
#     translation=start_point,
#     rotation=up_rotation
# )

# print("\nMoving to start point...")
# try:
#     ik_solutions = box.robot.ik(start_pose.to_matrix())
#     # Sort by distance to current config
#     distances = [np.linalg.norm(sol - current_config) for sol in ik_solutions]
#     sorted_indices = np.argsort(distances)
    
#     # Try all solutions for start point
#     solution_found = False
#     for idx in sorted_indices:
#         solution = ik_solutions[idx]
#         print(f"Start solution {idx}: {np.rad2deg(solution).round(1)}")
#         if box.robot.in_limits(solution):
#             print(f"Moving to start point... (using solution {idx})")
#             box.robot.move_to_q(solution)
#             current_config = solution
#             configs = [current_config]  # Reset configs to start with this one
#             solution_found = True
#             break
    
#     if not solution_found:
#         print("Could not reach start point - no valid solution")
#         exit()
        
# except Exception as e:
#     print(f"IK failed for start point: {e}")
#     exit()

# # Continue with pure pursuit
# reached_end = False

# while not reached_end:
#     # Get current position
#     current_transform = SE3().from_matrix(box.robot.fk(current_config))
#     current_pos = current_transform.translation
#     print(f"\nCurrent position: {current_pos.round(2)}")
    
#     # Find next target point
#     next_point = find_next_target(current_pos, path_points)
    
#     if next_point is None:
#         print("Reached end of path")
#         break
    
#     print(f"Next target: {next_point.round(2)}")
    
#     # Create target pose
#     target_pose = SE3(
#         translation=next_point,
#         rotation=up_rotation
#     )
    
#     # Get IK solution
#     try:
#         ik_solutions = box.robot.ik(target_pose.to_matrix())
#         # Sort by distance to current config
#         distances = [np.linalg.norm(sol - current_config) for sol in ik_solutions]
#         sorted_indices = np.argsort(distances)
        
#         # Try all solutions, starting with the closest one
#         solution_found = False
#         for idx in sorted_indices:
#             solution = ik_solutions[idx]
#             print(f"Solution {idx}: {np.rad2deg(solution).round(1)}")
#             if box.robot.in_limits(solution):
#                 print(f"Moving to next point... (using solution {idx})")
#                 box.robot.move_to_q(solution)
#                 current_config = solution
#                 configs.append(current_config)
#                 solution_found = True
#                 break
        
#         if not solution_found:
#             print("No valid solution found - all solutions out of limits")
#             break
            
#     except Exception as e:
#         print(f"IK failed: {e}")
#         break
    
#     # Check if we're close to the end point
#     if np.linalg.norm(current_pos - end_point) < 30:
#         print("Reached end point")
#         reached_end = True

# # Print the configurations
# print("\nGenerated Configurations:")
# print("np.deg2rad(np.array([")
# for config in configs:
#     config_deg = np.rad2deg(config).round(1)
#     print(f"    [{config_deg[0]:6.1f}, {config_deg[1]:6.1f}, {config_deg[2]:6.1f}, {config_deg[3]:6.1f}, {config_deg[4]:6.1f}, {config_deg[5]:6.1f}],")
# print("]))")

scene.display()
box.close()
