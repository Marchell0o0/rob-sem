import numpy as np
from src.se3 import SE3
from src.so3 import SO3
from pathlib import Path
from src.robot_box import RobotBox, RobotType
from src.scene3d import Scene3D

box = RobotBox(RobotType.CRS93, robot_active=False, camera_active=False)
scene = Scene3D().z_from_zero()
camera_to_base = SE3().from_matrix(np.load("calibration/calibration_data/working_camera_to_base.npy"))
print("Camera position:", camera_to_base.translation)

configurations_path = Path("only_10_both_robots")
configurations = []
poses = []
for idx, file in enumerate(configurations_path.glob("*.npy")):
    configuration = np.load(file)
    configurations.append(configuration)

    pose = SE3().from_matrix(box.robot.fk(configuration))
    poses.append(pose)
    scene.add_transform(f"{file.stem}", pose)

# Get last configuration for distance sorting
last_config = configurations[-1]

# Calculate bounding box
translations = np.array([pose.translation for pose in poses])
min_bounds = np.min(translations, axis=0)
max_bounds = np.max(translations, axis=0)

print("\nBounding Box:")
print(f"Min bounds: {min_bounds}")
print(f"Max bounds: {max_bounds}")
min_bounds = np.array([420, -166, 170])
max_bounds = np.array([599, 217, 367])
def sort_configurations_by_distance(configurations, reference_config):
    """Sort configurations by their distance to the reference configuration."""
    distances = [np.linalg.norm(config - reference_config) for config in configurations]
    sorted_indices = np.argsort(distances)
    return [configurations[i] for i in sorted_indices], [distances[i] for i in sorted_indices]

def sort_configurations_by_chain(configs):
    """Sort configurations to minimize distance between consecutive configurations."""
    if not configs:
        return []
    
    # Start with the first configuration
    sorted_configs = [configs[0]]
    remaining = configs[1:]
    
    # Keep adding the closest configuration to the last one
    while remaining:
        last = sorted_configs[-1]
        # Find closest remaining configuration
        distances = [np.linalg.norm(config - last) for config in remaining]
        closest_idx = np.argmin(distances)
        sorted_configs.append(remaining[closest_idx])
        remaining.pop(closest_idx)
    
    return sorted_configs

# Generate random points until we get 20 valid configurations
random_poses = []
new_configurations = []  # Array to store the best IK solution for each pose
attempts = 0
max_attempts = 100  # Prevent infinite loop

while len(new_configurations) < 20 and attempts < max_attempts:
    attempts += 1
    # Random position within bounds, ensuring Z >= 100mm
    random_pos = np.random.uniform(min_bounds, max_bounds)
    
    # Calculate direction to camera
    to_camera = camera_to_base.translation - random_pos
    to_camera = to_camera / np.linalg.norm(to_camera)
    
    # Add random deviation to the camera direction 
    random_angle_x = np.random.uniform(-np.pi/6, np.pi/6)
    random_angle_y = np.random.uniform(-np.pi/6, np.pi/6)
    
    # Create rotation matrices for random deviation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(random_angle_x), -np.sin(random_angle_x)],
        [0, np.sin(random_angle_x), np.cos(random_angle_x)]
    ])
    Ry = np.array([
        [np.cos(random_angle_y), 0, np.sin(random_angle_y)],
        [0, 1, 0],
        [-np.sin(random_angle_y), 0, np.cos(random_angle_y)]
    ])
    
    # Apply random rotation to the camera direction
    x_axis = Ry @ Rx @ to_camera
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Choose any perpendicular vector for y (using cross product with up vector)
    up = np.array([0, 0, 1])
    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Get z using cross product to ensure orthogonality
    z_axis = np.cross(x_axis, y_axis)
    
    # Create rotation matrix
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    
    # Create SE3 transform
    pose = SE3(
        translation=random_pos,
        rotation=SO3(rotation_matrix=rotation)
    )
    
    # Calculate IK solutions
    try:
        ik_solutions = box.robot.ik(pose.to_matrix())
        # Sort solutions by distance from last configuration
        sorted_solutions, distances = sort_configurations_by_distance(ik_solutions, last_config)
        
        # Check if the best solution is within robot limits
        best_solution = sorted_solutions[0]
        if box.robot.in_limits(best_solution):
            random_poses.append(pose)
            new_configurations.append(best_solution)
            scene.add_transform(f"Random_{len(new_configurations)}", pose)
            print(f"Found valid configuration {len(new_configurations)}/20")
        else:
            print(f"Configuration out of limits, attempt {attempts}")
            
    except Exception as e:
        print(f"IK failed: {e}, attempt {attempts}")

if len(new_configurations) < 20:
    print(f"\nWarning: Only found {len(new_configurations)} valid configurations in {max_attempts} attempts")

# Sort configurations to minimize movement between poses
sorted_configurations = sort_configurations_by_chain(new_configurations)

print("\nSorted Configurations Array (minimizing distances between consecutive poses):")
print("np.deg2rad(np.array([")

for i, config in enumerate(sorted_configurations):
    config_deg = np.rad2deg(config).round(1)
    print(f"    [{config_deg[0]:6.1f}, {config_deg[1]:6.1f}, {config_deg[2]:6.1f}, {config_deg[3]:6.1f}, {config_deg[4]:6.1f}, {config_deg[5]:6.1f}],")

print("]))")

scene.display()
box.close()

