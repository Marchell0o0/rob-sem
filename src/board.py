import numpy as np
import csv
from src.se3 import SE3
from src.so3 import SO3
import cv2

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

        self.ref_marker_id = self.pair[0]  # Reference marker is always the one with lower ID
        self.second_marker_id = self.pair[1]
        self.board_transform = None
        self.slot_transforms = []

    def calculate_slot_transforms(self) -> list:
        """Calculate transforms for all slots in camera frame.
        Requires board transform to be set first.
        
        Returns:
            List of (slot_index, SE3 transform) tuples
        """
        if self.board_transform is None:
            return []

        # Load slot positions from CSV
        csv_path = f"boards/positions_plate_{self.pair[0]:02d}-{self.pair[1]:02d}.csv"
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader)
                # Load slot positions
                slot_positions = []
                for row in reader:
                    x = float(row[0])
                    y = float(row[1])
                    slot_positions.append((x, y))
        except Exception as e:
            print(f"Error loading slots from {csv_path}: {e}")
            return []

        # Calculate transforms for each slot
        self.slot_transforms = []
        for i, (x, y) in enumerate(slot_positions):
            # Create slot transform in board coordinates
            slot_transform = SE3(translation=np.array([x, y, 0]))
            # Transform to camera frame
            slot_transform = self.board_transform * slot_transform

            rotation_down = SE3(rotation=SO3().from_euler_angles(np.deg2rad([0, 180, 0]), "xyz"))
            slot_transform = slot_transform * rotation_down

            self.slot_transforms.append((i, slot_transform))

        return self.slot_transforms

    def calculate_board_transform(self, aruco_corners: dict, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> SE3:
        """Calculate board transform from ArUco marker transforms using solvePnP.
        
        Args:
            aruco_corners: Dict mapping marker IDs to their corners and IDs
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            SE3 transform of the board in camera frame
        """
        if self.ref_marker_id not in aruco_corners or self.second_marker_id not in aruco_corners:
            return None
            
        # Get corners for both markers
        ref_corners = aruco_corners[self.ref_marker_id]
        second_corners = aruco_corners[self.second_marker_id]
        
        # Define 3D points for both markers in board coordinate system
        marker_size = self.MARKER_SIZE
        
        # First marker at origin
        ref_points = np.array([
            [-marker_size/2, marker_size/2, 0],  # top-left
            [marker_size/2, marker_size/2, 0],   # top-right
            [marker_size/2, -marker_size/2, 0],  # bottom-right
            [-marker_size/2, -marker_size/2, 0]  # bottom-left
        ])
        
        # Second marker offset by [180, 140] mm
        second_points = ref_points + np.array([180, 140, 0])
        
        # Combine all points
        obj_points = np.vstack([ref_points, second_points])
        img_points = np.vstack([ref_corners, second_corners])
        
        # Solve PnP with all 8 points
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Convert to SE3 transform
        R, _ = cv2.Rodrigues(rvec)
        self.board_transform = SE3(
            translation=tvec.flatten(),
            rotation=SO3(rotation_matrix=R)
        )
        
        return self.board_transform
