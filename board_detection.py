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

        self.slots = []
        self.slot_transforms = []

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

    def update_poses_from_corners(self, corners_dict, camera_matrix, dist_coeffs):
        """Update marker poses from detected corners.

        Args:
            corners_dict: Dict mapping marker IDs to corner arrays
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            True if poses were updated successfully
        """
        if self.ref_marker_id not in corners_dict or self.second_marker_id not in corners_dict:
            return False

        self.ref_marker_transform = self.estimate_marker_pose(
            corners_dict[self.ref_marker_id], camera_matrix, dist_coeffs)
        self.second_marker_transform = self.estimate_marker_pose(
            corners_dict[self.second_marker_id], camera_matrix, dist_coeffs)

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

    def draw_2d(self, img, camera_matrix, dist_coeffs, axis_length=30):
        """Draw board visualization in 2D image."""
        if not self.board_transform:
            return

        # Draw reference marker coordinate system
        ref_rvec, _ = cv2.Rodrigues(self.ref_marker_transform.rotation.rot)
        ref_tvec = self.ref_marker_transform.translation.reshape(3, 1)
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,
                          ref_rvec, ref_tvec, axis_length)

        # Draw second marker coordinate system
        second_rvec, _ = cv2.Rodrigues(
            self.second_marker_transform.rotation.rot)
        second_tvec = self.second_marker_transform.translation.reshape(3, 1)
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,
                          second_rvec, second_tvec, axis_length)

        # Draw slots
        for slot_idx, slot_transform in self.slot_transforms:
            # Draw slot coordinate system using its aligned rotation
            slot_rvec, _ = cv2.Rodrigues(slot_transform.rotation.rot)
            slot_tvec = slot_transform.translation.reshape(3, 1)
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,
                              slot_rvec, slot_tvec, axis_length//2)

            # Project slot center
            point_3d = np.float32([[0, 0, 0]])  # Origin in slot's frame
            point_2d, _ = cv2.projectPoints(
                point_3d,
                slot_rvec, slot_tvec,
                camera_matrix, dist_coeffs
            )

            # Convert to integer pixel coordinates
            center = (int(round(point_2d[0][0][0])),
                      int(round(point_2d[0][0][1])))
            text_pos = (center[0] + 5, center[1] - 5)

            # Draw circle and text
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            cv2.putText(img, f"S{slot_idx}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def draw_3d(self, ax, marker_colors=['r', 'g']):
        """Draw board visualization in 3D plot."""
        if not self.board_transform:
            return

        # # Draw reference marker
        # ref_pos = self.ref_marker_transform.translation
        # ref_rot = self.ref_marker_transform.rotation.rot
        # ax.scatter([ref_pos[0]], [ref_pos[1]], [ref_pos[2]],
        #            color='r', s=100, label='Ref Marker')
        # self.draw_3d_axes(ax, ref_pos, ref_rot, axis_length=50)

        # Draw second marker
        second_pos = self.second_marker_transform.translation
        second_rot = self.second_marker_transform.rotation.rot
        ax.scatter([second_pos[0]], [second_pos[1]], [second_pos[2]],
                   color='g', s=100, label='Second Marker')
        self.draw_3d_axes(ax, second_pos, second_rot, axis_length=50)

        # Draw board
        board_pos = self.board_transform.translation
        board_rot = self.board_transform.rotation.rot
        ax.scatter([board_pos[0]], [board_pos[1]], [board_pos[2]],
                   color='b', s=100, label='Board')
        self.draw_3d_axes(ax, board_pos, board_rot, axis_length=50)

        # Draw slots with their aligned rotations
        for slot_idx, slot_transform in self.slot_transforms:
            slot_pos = slot_transform.translation
            slot_rot = slot_transform.rotation.rot  # Use slot's aligned rotation
            ax.scatter([slot_pos[0]], [slot_pos[1]], [slot_pos[2]],
                       color='r', s=100, label=f'Slot {slot_idx}' if slot_idx == 0 else None)
            self.draw_3d_axes(ax, slot_pos, slot_rot, axis_length=25)
            # Add slot label
            ax.text(slot_pos[0], slot_pos[1], slot_pos[2],
                    f'S{slot_idx}', color='r')

    @staticmethod
    def draw_3d_axes(ax, pos, rot, axis_length=50):
        """Draw coordinate axes in 3D plot."""
        # Draw axes
        for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
            ax.quiver(pos[0], pos[1], pos[2],
                      axis_length * rot[0, i],
                      axis_length * rot[1, i],
                      axis_length * rot[2, i],
                      color=color, label=label if pos[2] < 0 else None)

    def load_slot_positions(self, csv_path: str) -> bool:
        """Load slot positions from CSV file and calculate their transforms.
        First row contains ArUco marker IDs and should be skipped.
        CSV coordinates are used directly as they are in board coordinate system.

        Args:
            csv_path: Path to CSV file containing slot positions

        Returns:
            True if positions were loaded successfully for this board
        """
        if not self.ref_marker_transform or not self.second_marker_transform:
            return False

        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                # Check if first row matches this board's markers
                header = next(reader)
                if len(header) != 2 or int(header[0]) != self.ref_marker_id or int(header[1]) != self.second_marker_id:
                    return False

                # Load slot positions
                self.slots = []
                for row in reader:
                    # Convert string values to float
                    x = float(row[0])
                    y = float(row[1])
                    self.slots.append((x, y))
                # Calculate transforms once
                self._calculate_slot_transforms()
                return True
        except (FileNotFoundError, ValueError, IndexError):
            return False

    def _calculate_slot_transforms(self):
        """Calculate transforms for all slots in camera frame.
        After transforming to camera frame, aligns slot coordinate systems 
        with reference ArUco marker convention:
        - X axis pointing right (red)
        - Y axis pointing down (green)
        - Z axis pointing out of plane (blue)
        """
        self.slot_transforms = []
        if not self.slots:
            return

        # Get reference marker axes in camera frame
        ref_x = self.ref_marker_transform.rotation.rot[:, 0]  # Right
        ref_y = self.ref_marker_transform.rotation.rot[:, 1]  # Down
        ref_z = self.ref_marker_transform.rotation.rot[:, 2]  # Out

        for i, (x, y) in enumerate(self.slots):
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

            # Create new rotation matrix with aligned axes
            aligned_rot = np.column_stack([x_axis, y_axis, z_axis])

            # Create new transform with aligned rotation
            aligned_transform = SE3(
                translation=slot_camera_transform.translation,
                rotation=SO3(rotation_matrix=aligned_rot)
            )

            self.slot_transforms.append((i, aligned_transform))

    @classmethod
    def create_boards_from_markers(cls, marker_ids: list) -> list:
        """Create board instances from detected marker IDs.

        Args:
            marker_ids: List of detected marker IDs

        Returns:
            List of Board instances that can be created from the detected markers
        """
        marker_ids = set(marker_ids)
        boards = []

        for pair in cls.VALID_PAIRS:
            if pair[0] in marker_ids and pair[1] in marker_ids:
                boards.append(cls(pair[0], pair[1]))

        return boards
