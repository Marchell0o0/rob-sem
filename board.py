import numpy as np
import cv2
from se3 import SE3
from so3 import SO3
import csv
import matplotlib.pyplot as plt


class Board:
    # Valid marker pairs that define boards
    VALID_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]
    MARKER_SIZE = 36  # mm

    def __init__(self, marker1_id: int, marker2_id: int):
        """Initialize a board with two marker IDs.

        Args:
            marker1_id: ID of first marker (should be odd)
            marker2_id: ID of second marker (should be even)
        """
        self.pair = tuple(sorted([marker1_id, marker2_id]))
        if self.pair not in self.VALID_PAIRS:
            raise ValueError(
                f"Invalid marker pair {self.pair}. Must be one of {self.VALID_PAIRS}")

        # Reference marker is always the one with lower ID
        self.ref_marker_id = self.pair[0]
        self.second_marker_id = self.pair[1]

        # Will be set when markers are detected
        self.ref_marker_transform = None
        self.second_marker_transform = None
        self.board_transform = None

        # Raw positions from CSV (x,y coordinates)
        self.slot_positions = []
        # Transformed slots in camera frame (index, SE3 transform)
        self.slots = []

    def _load_slot_positions(self) -> bool:
        """Load slot positions from CSV file.
        CSV file should be named positions_plate_XX-YY.csv where XX and YY are marker IDs.

        Returns:
            True if positions were loaded successfully
        """
        csv_path = f"boards/positions_plate_{self.pair[0]:02d}-{self.pair[1]:02d}.csv"
        print(f"Loading slots from {csv_path}")
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                # Check if first row matches this board's markers
                header = next(reader)
                print(f"CSV header: {header}")
                if len(header) != 2 or int(header[0]) != self.ref_marker_id or int(header[1]) != self.second_marker_id:
                    print(f"Header mismatch: expected {self.ref_marker_id}, {self.second_marker_id}")
                    return False

                # Load slot positions
                self.slot_positions = []
                for row in reader:
                    # Convert string values to float
                    x = float(row[0])
                    y = float(row[1])
                    self.slot_positions.append((x, y))
                
                print(f"Loaded {len(self.slot_positions)} slot positions: {self.slot_positions}")
                assert len(self.slot_positions) == 4, f"Expected 4 slots, got {len(self.slot_positions)}"
                
                # Calculate transforms if we have marker poses
                if self.board_transform is not None:
                    print("Board transform exists, calculating slot transforms")
                    self._calculate_slot_transforms()
                else:
                    print("No board transform yet, skipping slot transform calculation")
                return True
        except (FileNotFoundError, ValueError, IndexError) as e:
            print(f"Error loading slots: {e}")
            return False

    @staticmethod
    def estimate_marker_pose(corners, camera_matrix, dist_coeffs):
        """Estimate pose of a single marker.

        Args:
            corners: Marker corners from ArUco detection
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            SE3 transform of marker in camera frame
        """
        # Define marker corners in marker's coordinate system
        # Corners arranged to get:
        # - X axis pointing right (red)
        # - Y axis pointing down (green)
        # - Z axis out of plane (blue)
        half_size = Board.MARKER_SIZE / 2
        objPoints = np.array([
            [-half_size, half_size, 0],     # bottom-left
            [-half_size, -half_size, 0],    # top-left
            [half_size, -half_size, 0],     # top-right
            [half_size, half_size, 0]],     # bottom-right
            dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objPoints, corners, camera_matrix, dist_coeffs)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return SE3(translation=tvec.flatten(), rotation=SO3(rotation_matrix=R))
        return None

    def update_poses(self, aruco_transforms):
        """Update board pose from detected ArUco transforms.

        Args:
            aruco_transforms: Dict mapping marker IDs to SE3 transforms

        Returns:
            True if poses were updated successfully
        """
        if self.ref_marker_id not in aruco_transforms or self.second_marker_id not in aruco_transforms:
            return False

        self.ref_marker_transform = aruco_transforms[self.ref_marker_id]
        self.second_marker_transform = aruco_transforms[self.second_marker_id]

        if self.ref_marker_transform is not None and self.second_marker_transform is not None:
            # Calculate board transform once
            self._calculate_board_transform()
            return True
        return False

    def _calculate_board_transform(self):
        """Calculate board transform in camera frame."""
        # Get Z axes from both markers
        z1 = self.ref_marker_transform.rotation.rot[:, 2]
        z2 = self.second_marker_transform.rotation.rot[:, 2]
        z = (z1 + z2) / 2

        # Get vector from ref marker to second marker
        delta = self.second_marker_transform.translation - \
            self.ref_marker_transform.translation

        # Calculate board normal (Z axis)
        # First get a vector perpendicular to both delta and z1
        n1 = np.cross(delta, z)
        # Then get a vector perpendicular to both delta and n1
        z_axis = -np.cross(delta, n1)  # Remove negative to match camera frame
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Project delta onto the board plane (perpendicular to z_axis)
        delta = delta / np.linalg.norm(delta)

        # Calculate angle from 140x180 right triangle
        angle = np.arctan2(140, 180)  # angle between X axis and diagonal

        # For X axis: rotate by angle
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_axis = np.array([
            cos_a * delta[0] - sin_a * delta[1],
            cos_a * delta[1] + sin_a * delta[0],
            delta[2]
        ])

        # For Y axis: rotate by -(90° - angle)
        angle_y = np.pi/2 - angle
        cos_a = np.cos(-angle_y)  # negative to rotate in opposite direction
        sin_a = np.sin(-angle_y)
        y_axis = np.array([
            cos_a * delta[0] - sin_a * delta[1],
            cos_a * delta[1] + sin_a * delta[0],
            delta[2]
        ])

        # Normalize all axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Create rotation matrix with proper axes order
        rotation = np.column_stack([x_axis, y_axis, z_axis])

        # Create SE3 transform with ref marker position but new rotation
        self.board_transform = SE3(
            translation=self.ref_marker_transform.translation,
            rotation=SO3(rotation_matrix=rotation)
        )

    def _calculate_slot_transforms(self):
        """Calculate transforms for all slots in camera frame.
        After transforming to camera frame, aligns slot coordinate systems 
        with reference ArUco marker convention:
        - X axis pointing right (red)
        - Y axis pointing down (green)
        - Z axis pointing out of plane (blue)
        """
        print("Calculating slot transforms")
        self.slots = []
        if not self.slot_positions:
            print("No slot positions loaded!")
            return

        # Get reference marker axes in camera frame
        ref_x = self.ref_marker_transform.rotation.rot[:, 0]  # Right
        ref_y = self.ref_marker_transform.rotation.rot[:, 1]  # Down
        ref_z = self.ref_marker_transform.rotation.rot[:, 2]  # Out

        for i, (x, y) in enumerate(self.slot_positions):
            print(f"Processing slot {i}: ({x}, {y})")
            # First create slot transform in board coordinates
            slot_transform = SE3(translation=np.array([x, y, 0]))

            # Transform slot to camera frame using board transform
            slot_camera_transform = self.board_transform * slot_transform

            # Get current axes
            current_rot = slot_camera_transform.rotation.rot
            x_axis = current_rot[:, 0]
            y_axis = current_rot[:, 1]
            z_axis = current_rot[:, 2]

            # Check if x and y axes need to be swapped (90-degree rotation)
            # Compare dot products to see which axis aligns better
            x_dot_ref_x = abs(np.dot(x_axis, ref_x))
            x_dot_ref_y = abs(np.dot(x_axis, ref_y))
            y_dot_ref_x = abs(np.dot(y_axis, ref_x))
            y_dot_ref_y = abs(np.dot(y_axis, ref_y))

            # If x axis is more aligned with ref_y and y axis with ref_x, swap them
            if x_dot_ref_y > x_dot_ref_x and y_dot_ref_x > y_dot_ref_y:
                x_axis, y_axis = y_axis, -x_axis  # Negative to maintain right-hand rule

            # Now align axes with reference marker
            if np.dot(x_axis, ref_x) < 0:
                x_axis = -x_axis
            if np.dot(y_axis, ref_y) < 0:
                y_axis = -y_axis

            # Ensure axes are orthogonal
            # First normalize x_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Make y_axis perpendicular to x_axis
            y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # Calculate z_axis as cross product to ensure right-hand rule
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Verify orthogonality
            xy_dot = abs(np.dot(x_axis, y_axis))
            yz_dot = abs(np.dot(y_axis, z_axis))
            xz_dot = abs(np.dot(x_axis, z_axis))
            
            orthogonality_threshold = 1e-10
            if xy_dot > orthogonality_threshold or yz_dot > orthogonality_threshold or xz_dot > orthogonality_threshold:
                print(f"Warning: Axes not orthogonal for slot {i}")
                print(f"xy_dot: {xy_dot}, yz_dot: {yz_dot}, xz_dot: {xz_dot}")
                continue

            # Create new rotation matrix with aligned and orthogonal axes
            aligned_rot = np.column_stack([x_axis, y_axis, z_axis])

            # Verify rotation matrix is proper (det = 1)
            det = np.linalg.det(aligned_rot)
            if not np.isclose(abs(det), 1.0, rtol=1e-5):
                print(f"Warning: Invalid rotation matrix for slot {i}, determinant = {det}")
                continue

            # Create new transform with aligned rotation
            aligned_transform = SE3(
                translation=slot_camera_transform.translation,
                rotation=SO3(rotation_matrix=aligned_rot)
            )
            aligned_transform = aligned_transform * SE3(translation=np.array([0, 0, 0]),\
                                                         rotation=SO3.from_euler_angles(np.deg2rad([180, 0, 0]), ["x", "y", "z"]))

            self.slots.append((i, aligned_transform))
        
        print(f"Calculated {len(self.slots)} slot transforms")
        assert len(self.slots) == 4, f"Expected 4 slot transforms, got {len(self.slots)}"

    @classmethod
    def create_boards_from_transforms(cls, aruco_transforms: dict) -> list:
        """Create board instances from detected ArUco transforms.

        Args:
            aruco_transforms: Dict mapping marker IDs to SE3 transforms

        Returns:
            List of Board instances
        """
        marker_ids = set(aruco_transforms.keys())
        print(f"Detected markers: {marker_ids}")
        boards = []

        for pair in cls.VALID_PAIRS:
            print(f"Checking pair {pair}")
            if pair[0] in marker_ids and pair[1] in marker_ids:
                print(f"Creating board for pair {pair}")
                board = cls(pair[0], pair[1])
                if board.update_poses(aruco_transforms):
                    print(f"Updated poses for board {pair}")
                    # Load slot positions after board transform is calculated
                    if board._load_slot_positions():
                        print(f"Successfully loaded slots for board {pair}")
                    else:
                        print(f"Failed to load slots for board {pair}")
                    boards.append(board)

        return boards
