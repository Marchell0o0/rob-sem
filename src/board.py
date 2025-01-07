import numpy as np
import cv2
from src.se3 import SE3
from src.so3 import SO3
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

        self.empty = None

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
                if len(header) != 2 or int(header[0]) != self.ref_marker_id or int(header[1]) != self.second_marker_id:
                    print(
                        f"Header mismatch: expected {self.ref_marker_id}, {self.second_marker_id}")
                    return False

                # Load slot positions
                self.slot_positions = []
                for row in reader:
                    # Convert string values to float
                    x = float(row[0])
                    y = float(row[1])
                    self.slot_positions.append((x, y))

                print(
                    f"Loaded {len(self.slot_positions)} slot positions: {self.slot_positions}")
                assert len(
                    self.slot_positions) == 4, f"Expected 4 slots, got {len(self.slot_positions)}"

                # Calculate transforms if we have marker poses
                if self.board_transform is not None:
                    self._calculate_slot_transforms()
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

    def _calculate_board_transform(self, x_axis: np.ndarray):
        """Calculate board transform in camera frame."""

        # x1 = self.ref_marker_transform.rotation.rot[:, 0]
        # x2 = self.second_marker_transform.rotation.rot[:, 0]
        # x = (x1 + x2) / 2
        delta = self.second_marker_transform.translation - self.ref_marker_transform.translation
        delta = delta / np.linalg.norm(delta)
        # z_axis = np.cross(x_axis, delta)
        z_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis[2] = 0
        x_axis = np.cross(y_axis, z_axis)

        # Normalize all axes
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Create rotation matrix with proper axes order
        rotation = np.column_stack([x_axis, y_axis, z_axis])

        # Create SE3 transform with ref marker position but new rotation
        height = (self.ref_marker_transform.translation[2] + self.second_marker_transform.translation[2]) / 2
        board_translation = self.ref_marker_transform.translation
        board_translation[2] = height

        self.board_transform = SE3(
            translation=board_translation,
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
        self.slots = []
        if not self.slot_positions:
            print("No slot positions loaded!")
            return

        for i, (x, y) in enumerate(self.slot_positions):
            # First create slot transform in board coordinates
            slot_transform = SE3(translation=np.array([x, y, 0]))

            # Transform slot to camera frame using board transform
            slot_camera_transform = self.board_transform * slot_transform
            slot_camera_transform = slot_camera_transform * SE3(translation=np.array([0, 0, 0]),
                                                               rotation=SO3.from_euler_angles(np.deg2rad([180, 0, 0]), ["x", "y", "z"]))
          
            self.slots.append((i, slot_camera_transform))

        assert len(
            self.slots) == 4, f"Expected 4 slot transforms, got {len(self.slots)}"

    @classmethod
    def create_boards_from_transforms(cls, aruco_transforms: dict, image=None, camera_image=None) -> list:
        """Create board instances from detected ArUco transforms.
        
        Args:
            aruco_transforms: Dict mapping marker IDs to SE3 transforms
            image: Optional camera image for board contour detection
            camera_image: CameraImage instance for projection
        """
        marker_ids = set(aruco_transforms.keys())
        print(f"Detected markers: {marker_ids}")
        boards = []

        for pair in cls.VALID_PAIRS:
            if pair[0] in marker_ids and pair[1] in marker_ids:
                print(f"Creating board for pair {pair}")
                board = cls(pair[0], pair[1])
                if image is not None and camera_image is not None:
                    board.ref_marker_transform = aruco_transforms[board.ref_marker_id]
                    board.second_marker_transform = aruco_transforms[board.second_marker_id]
                    x_axis = board.detect_board_contour(image, camera_image)
                    if x_axis is not None:
                        board._calculate_board_transform(x_axis)
                    else:
                        print(f"Failed to detect contour for board {pair}")
                        continue
                board._load_slot_positions()
                boards.append(board)

        return boards
    # @classmethod
    # def create_boards_from_transforms(cls, aruco_transforms: dict, image=None, camera_image=None) -> list:
    #     """Create board instances from detected ArUco transforms.
        
    #     Args:
    #         aruco_transforms: Dict mapping marker IDs to SE3 transforms
    #         image: Optional camera image for board contour detection
    #         camera_image: CameraImage instance for projection
    #     """
    #     marker_ids = set(aruco_transforms.keys())
    #     print(f"Detected markers: {marker_ids}")
    #     boards = []

    #     for pair in cls.VALID_PAIRS:
    #         if pair[0] in marker_ids and pair[1] in marker_ids:
    #             print(f"Creating board for pair {pair}")
    #             board = cls(pair[0], pair[1])
    #             if image is not None and camera_image is not None:
    #                 board.ref_marker_transform = aruco_transforms[board.ref_marker_id]
    #                 board.second_marker_transform = aruco_transforms[board.second_marker_id]
    #                 x_axis = board.detect_board_contour(image, camera_image)
    #                 if x_axis is not None:
    #                     board._calculate_board_transform(x_axis)
    #                 else:
    #                     print(f"Failed to detect contour for board {pair}")
    #                     continue
    #             board._load_slot_positions()
    #             boards.append(board)

    #     return boards

    def detect_board_contour(self, image, camera_image) -> np.ndarray:
        """Detect board contour and return its X axis direction."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Project ArUco marker corners to image coordinates
        ref_corners = self.project_marker_corners(self.ref_marker_transform, camera_image)
        second_corners = self.project_marker_corners(self.second_marker_transform, camera_image)
        
        # Get bounding box of both markers with padding
        all_corners = np.vstack([ref_corners, second_corners])
        min_xy = np.min(all_corners, axis=0).astype(int) - 150
        max_xy = np.max(all_corners, axis=0).astype(int) + 150
        
        # Ensure bounds are within image
        min_xy = np.maximum(min_xy, [0, 0])
        max_xy = np.minimum(max_xy, [gray.shape[1], gray.shape[0]])
        
        # Extract ROI
        roi = gray[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]
        
        # Preprocessing
        denoised = cv2.fastNlMeansDenoising(roi, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 70)
        
        # Connect edges
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        small_kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, small_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]
        if not valid_contours:
            return None
        
        board_contour = max(valid_contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.03 * cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, epsilon, True)
        
        if len(approx) != 4:
            return None
        
        # Get corners and adjust for ROI offset
        corners = approx.reshape(-1, 2) + min_xy
        
        # Sort corners by x coordinate
        sorted_x = corners[corners[:, 0].argsort()]
        left_points = sorted_x[:2]
        right_points = sorted_x[2:]
        
        # Sort each pair by y coordinate
        left_points = left_points[left_points[:, 1].argsort()]
        right_points = right_points[right_points[:, 1].argsort()]
        
        # Get side lengths
        left_side = left_points[1] - left_points[0]
        right_side = right_points[1] - right_points[0]
        left_length = np.linalg.norm(left_side)
        right_length = np.linalg.norm(right_side)
        
        top_side = right_points[0] - left_points[0]
        bottom_side = right_points[1] - left_points[1]
        top_length = np.linalg.norm(top_side)
        bottom_length = np.linalg.norm(bottom_side)
        
        # Get x_axis from the longer side
        if max(left_length, right_length) > max(top_length, bottom_length):
            x_axis_1 = right_side / np.linalg.norm(right_side)
            x_axis_2 = left_side / np.linalg.norm(left_side)
            x_axis = (x_axis_1 + x_axis_2) / 2
        else:
            x_axis_1 = top_side / np.linalg.norm(top_side)
            x_axis_2 = bottom_side / np.linalg.norm(bottom_side)
            x_axis = (x_axis_1 + x_axis_2) / 2

        
        # Add Z component
        x_axis = np.array([x_axis[0], x_axis[1], 0])
        
        # Align with marker direction
        marker_direction = self.second_marker_transform.translation - self.ref_marker_transform.translation
        marker_direction = marker_direction / np.linalg.norm(marker_direction)
        if np.dot(x_axis, marker_direction) < 0:
            x_axis = -x_axis
        
        return x_axis

    def project_marker_corners(self, marker_transform: SE3, camera_image) -> np.ndarray:
        """Project ArUco marker corners to image coordinates."""
        # Define marker corners in marker's coordinate system
        half_size = self.MARKER_SIZE / 2
        marker_corners = np.array([
            [-half_size, half_size, 0],     # bottom-left
            [-half_size, -half_size, 0],    # top-left
            [half_size, -half_size, 0],     # top-right
            [half_size, half_size, 0]       # bottom-right
        ])
        
        # Transform corners to camera frame and project
        camera_corners = []
        for corner in marker_corners:
            # Transform corner to camera frame
            corner_camera = marker_transform * SE3(translation=corner)
            # Project using CameraImage's method
            point_2d = camera_image.project_point(corner_camera.translation)
            camera_corners.append(point_2d)
        
        return np.array(camera_corners)
