import numpy as np
import cv2
from src.se3 import SE3
from src.so3 import SO3
import csv
import matplotlib.pyplot as plt
from src.scene3d import Scene3D

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
        BOARD_ARUCO_SIZE = self.MARKER_SIZE
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
        board_corners_2d = approx.reshape(-1, 2) + min_xy
        print("corners 2d", board_corners_2d)
        
        # Sort corners based on distance to reference ArUco marker center
        ref_center = np.mean(ref_corners, axis=0)
        distances_to_ref = [np.linalg.norm(corner - ref_center) for corner in board_corners_2d]
        print("distances to ref", distances_to_ref)
        
        # Sort corners based on distances
        corner_info = list(enumerate(zip(board_corners_2d, distances_to_ref)))
        sorted_corners = sorted(corner_info, key=lambda x: x[1][1])
        print("sorted corners", sorted_corners)
        
        # Extract just the corner coordinates in sorted order
        board_corners_2d = np.array([corner[1][0] for corner in sorted_corners])
        
        # 3D points are already in the desired order (closest to farthest from ref marker)
        board_corners_3d = np.array([
            [-30, -30, 0],    # closest to ref marker
            [-30, 170, 0],    # second closest
            [210, -30, 0],    # third closest
            [210, 170, 0],    # farthest from ref marker
        ], dtype=np.float32)
        
        # Draw board visualization
        vis_img = image.copy()
        
        # Draw ArUco markers
        cv2.aruco.drawDetectedMarkers(vis_img, [ref_corners.reshape(1, 4, 2)], np.array([self.ref_marker_id]))
        cv2.aruco.drawDetectedMarkers(vis_img, [second_corners.reshape(1, 4, 2)], np.array([self.second_marker_id]))
        
        # Add coordinate labels for first marker corners
        first_marker_coords = [
            [-BOARD_ARUCO_SIZE/2, BOARD_ARUCO_SIZE/2],  # top-left
            [BOARD_ARUCO_SIZE/2, BOARD_ARUCO_SIZE/2],   # top-right
            [BOARD_ARUCO_SIZE/2, -BOARD_ARUCO_SIZE/2],  # bottom-right
            [-BOARD_ARUCO_SIZE/2, -BOARD_ARUCO_SIZE/2], # bottom-left
        ]
        for corner_2d, corner_3d in zip(ref_corners, first_marker_coords):
            coord_text = f"({corner_3d[0]:.0f}, {corner_3d[1]:.0f})"
            text_pos = (int(corner_2d[0]) + 5, int(corner_2d[1]) + 5)
            cv2.putText(vis_img, coord_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add coordinate labels for second marker corners (offset by [140, 180])
        second_marker_coords = [[x + 140, y + 180] for x, y in first_marker_coords]
        for corner_2d, corner_3d in zip(second_corners, second_marker_coords):
            coord_text = f"({corner_3d[0]:.0f}, {corner_3d[1]:.0f})"
            text_pos = (int(corner_2d[0]) + 5, int(corner_2d[1]) + 5)
            cv2.putText(vis_img, coord_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # Draw board edges in order: closest to ref -> second closest -> third closest -> farthest -> back to closest
        cv2.line(vis_img, tuple(board_corners_2d[0].astype(int)), tuple(board_corners_2d[1].astype(int)), (0, 0, 255), 2)  # closest to second closest
        cv2.line(vis_img, tuple(board_corners_2d[1].astype(int)), tuple(board_corners_2d[3].astype(int)), (0, 0, 255), 2)  # second closest to farthest
        cv2.line(vis_img, tuple(board_corners_2d[3].astype(int)), tuple(board_corners_2d[2].astype(int)), (0, 0, 255), 2)  # farthest to third closest
        cv2.line(vis_img, tuple(board_corners_2d[2].astype(int)), tuple(board_corners_2d[0].astype(int)), (0, 0, 255), 2)  # third closest to closest
        
        # Draw and label corners
        for i, (edge_point_3d, edge_point_2d) in enumerate(zip(board_corners_3d, board_corners_2d)):
            # Draw corner point
            cv2.circle(vis_img, tuple(edge_point_2d.astype(int)), 5, (255, 0, 0), -1)  # Blue dots
            
            # Add coordinate labels
            coord_text = f"({edge_point_3d[0]:.0f}, {edge_point_3d[1]:.0f})"
            text_pos = (int(edge_point_2d[0]) + 10, int(edge_point_2d[1]) + 10)
            
            # Draw text with background
            text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_img, 
                        (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                        (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                        (255, 255, 255), -1)
            cv2.putText(vis_img, coord_text, text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show visualization
        cv2.imshow("Board Detection", vis_img)
        cv2.waitKey(1)
        
        marker_positions = [
            [0, 0, 0],           # First marker at origin
            [140, 180, 0]        # Second marker offset
        ]
        
        marker_size = BOARD_ARUCO_SIZE
        marker_corners_3d = []
        for mx, my, mz in marker_positions:
            corners = np.array([
                [mx - marker_size/2, my + marker_size/2, mz],  # top-left
                [mx + marker_size/2, my + marker_size/2, mz],  # top-right
                [mx + marker_size/2, my - marker_size/2, mz],  # bottom-right
                [mx - marker_size/2, my - marker_size/2, mz],  # bottom-left
            ], dtype=np.float32)
            marker_corners_3d.extend(corners)
        
        # Combine all 3D points and 2D projections
        obj_points = np.vstack([board_corners_3d, marker_corners_3d])
        img_points = np.vstack([board_corners_2d, ref_corners, second_corners])
        
        # Estimate board pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, img.camera_matrix, img.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print(f"Failed to estimate pose for marker pair ({self.ref_marker_id}, {self.second_marker_id})")
            return None
        
        # Convert to SE3 transform
        R, _ = cv2.Rodrigues(rvec)
        board_transform = SE3(
            translation=tvec.flatten(),
            rotation=SO3(rotation_matrix=R)
        )

        img.add_transform(f"Board {(self.ref_marker_id, self.second_marker_id)}", board_transform)
        img.display()
        
        # Create Board object
        board = Board(self.ref_marker_id, self.second_marker_id)
        board.board_transform = board_transform
        # board = Board(
        #     pair=(self.ref_marker_id, self.second_marker_id),
        #     ref_marker_id=self.ref_marker_id,
        #     second_marker_id=self.second_marker_id,
        #     ref_marker_transform=None,  # Not needed
        #     second_marker_transform=None,  # Not needed
        #     board_transform=board_transform
        # )
        
        # Load slot positions from CSV
        board_csv = f"boards/positions_plate_{self.ref_marker_id:02d}_{self.second_marker_id:02d}.csv"
        try:
            slot_data = np.loadtxt(board_csv, delimiter=',')
            board.slot_positions = slot_data
            
            # Calculate slot transforms
            board.slots = []
            for i, (x, y) in enumerate(board.slot_positions):
                # Create slot transform in board coordinates
                slot_transform = SE3(translation=np.array([x, y, 0]))
                
                # Transform slot to camera frame using board transform
                slot_camera_transform = board.board_transform * slot_transform
                # Align slot coordinate system with reference ArUco marker convention
                slot_camera_transform = slot_camera_transform * SE3(
                    translation=np.array([0, 0, 0]),
                    rotation=SO3.from_euler_angles(np.deg2rad([180, 0, 0]), ["x", "y", "z"])
                )
                
                board.slots.append((i, slot_camera_transform))
            
            print(f"Loaded {len(board.slots)} slots for board {board.pair}")
            
        except Exception as e:
            print(f"Failed to load slot positions for board {board.pair}: {e}")
            return None
        
        return board

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

    @staticmethod
    def detect_boards_from_image(img):
        """
        Detect boards from an image using ArUco markers and contour detection.
        
        Args:
            img: CameraImage object containing the image and camera parameters
            
        Returns:
            list: List of Board objects
        """
        # ArUco detection parameters
        BOARD_ARUCO_SIZE = 36
        BOARD_ARUCO_DICT = cv2.aruco.DICT_4X4_50
        
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(BOARD_ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(img.image)
        scene = Scene3D().invert_z_axis().z_from_zero()
        
        if ids is None:
            print("No ArUco markers detected")
            return []
            
        # Convert ids to 1D array
        ids = ids.flatten()
        
        # Create dictionary of marker corners
        marker_dict = {}  # Store corners for each marker ID
        for marker_corners, marker_id in zip(corners, ids):
            marker_dict[marker_id] = marker_corners[0]  # corners[0] because OpenCV returns a nested array

        # Find valid marker pairs using predefined VALID_PAIRS
        valid_pairs = []
        for pair in Board.VALID_PAIRS:
            if pair[0] in marker_dict and pair[1] in marker_dict:
                valid_pairs.append(pair)

        if not valid_pairs:
            print("No valid marker pairs found")
            return []

        boards = []
        for ref_id, second_id in valid_pairs:
            print(f"Processing marker pair: {ref_id} - {second_id}")
            # Get marker corners
            ref_corners = marker_dict[ref_id]
            second_corners = marker_dict[second_id]
            
            # # Get ROI around the markers
            # all_corners = np.vstack([ref_corners, second_corners])
            # min_xy = np.min(all_corners, axis=0).astype(int) - 500
            # max_xy = np.max(all_corners, axis=0).astype(int) + 500
            
            # # Ensure bounds are within image
            # min_xy = np.maximum(min_xy, [0, 0])
            # max_xy = np.minimum(max_xy, [img.image.shape[1], img.image.shape[0]])
            
            # # Extract and process ROI
            # roi = img.image[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]
            # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # # Preprocessing for edge detection
            # denoised = cv2.fastNlMeansDenoising(gray_roi, None, 10, 7, 21)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # enhanced = clahe.apply(denoised)
            # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            # edges = cv2.Canny(blurred, 20, 70)
            
            # # Connect edges
            # kernel = np.ones((5,5), np.uint8)
            # dilated = cv2.dilate(edges, kernel, iterations=1)
            # small_kernel = np.ones((3,3), np.uint8)
            # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, small_kernel, iterations=1)
            # eroded = cv2.erode(closed, small_kernel, iterations=2)
            
            # # Find contours
            # contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # # Debug visualization
            # debug_img = roi.copy()  # Use color ROI directly
            
            # # Draw all contours in red
            # cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
            
            # # Filter contours by area
            # valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]
            
            # # Draw largest contour in green if found
            # if valid_contours:
            #     board_contour = max(valid_contours, key=cv2.contourArea)
            #     cv2.drawContours(debug_img, [board_contour], -1, (0, 255, 0), 3)
            
            # # Show debug images
            # cv2.imshow("Edges", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            # cv2.imshow("Dilated", cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))
            # cv2.imshow("Closed", cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR))
            # cv2.imshow("Eroded", cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR))
            # cv2.imshow("Contours", debug_img)
            # cv2.waitKey(1)

            # if not valid_contours:
            #     print(f"No valid contours found for marker pair ({ref_id}, {second_id})")
            #     continue
                
            # # Get largest contour
            # board_contour = max(valid_contours, key=cv2.contourArea)
            
            # # Approximate polygon
            # epsilon = 0.03 * cv2.arcLength(board_contour, True)
            # approx = cv2.approxPolyDP(board_contour, epsilon, True)
            
            # if len(approx) != 4:
            #     print(f"Invalid board contour for marker pair ({ref_id}, {second_id}): {len(approx)} corners")
            #     continue
            
            # # Adjust corners for ROI offset
            # board_corners_2d = approx.reshape(-1, 2) + min_xy
            # print("corners 2d", board_corners_2d)
            
            # # Sort corners based on distance to reference ArUco marker center
            # ref_center = np.mean(ref_corners, axis=0)
            # distances_to_ref = [np.linalg.norm(corner - ref_center) for corner in board_corners_2d]
            # print("distances to ref", distances_to_ref)
            
            # # Sort corners based on distances
            # corner_info = list(enumerate(zip(board_corners_2d, distances_to_ref)))
            # sorted_corners = sorted(corner_info, key=lambda x: x[1][1])
            # print("sorted corners", sorted_corners)
            
            # # Extract just the corner coordinates in sorted order
            # board_corners_2d = np.array([corner[1][0] for corner in sorted_corners])
            
            # # 3D points are already in the desired order (closest to farthest from ref marker)
            # board_corners_3d = np.array([
            #     [-30, -30, 0],    # closest to ref marker
            #     [-30, 170, 0],    # second closest
            #     [210, -30, 0],    # third closest
            #     [210, 170, 0],    # farthest from ref marker
            # ], dtype=np.float32)
            
            # Draw board visualization
            vis_img = img.image.copy()
            
            # Draw ArUco markers and their corners
            cv2.aruco.drawDetectedMarkers(vis_img, [ref_corners.reshape(1, 4, 2)], np.array([ref_id]))
            cv2.aruco.drawDetectedMarkers(vis_img, [second_corners.reshape(1, 4, 2)], np.array([second_id]))
            
            # Draw corners and add relative coordinates for reference marker
            for i, corner_2d in enumerate(ref_corners):
                cv2.circle(vis_img, tuple(corner_2d.astype(int)), 3, (0, 255, 0), -1)  # Green dots
                # Get relative coordinates in board frame
                if i == 0:  # top-left
                    coord = [-BOARD_ARUCO_SIZE/2, BOARD_ARUCO_SIZE/2, 0]
                elif i == 1:  # top-right
                    coord = [BOARD_ARUCO_SIZE/2, BOARD_ARUCO_SIZE/2, 0]
                elif i == 2:  # bottom-right
                    coord = [BOARD_ARUCO_SIZE/2, -BOARD_ARUCO_SIZE/2, 0]
                else:  # bottom-left
                    coord = [-BOARD_ARUCO_SIZE/2, -BOARD_ARUCO_SIZE/2, 0]
                coord_text = f"({coord[0]:.0f}, {coord[1]:.0f})"
                text_pos = (int(corner_2d[0]) + 5, int(corner_2d[1]) + 5)
                cv2.putText(vis_img, coord_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw corners and add relative coordinates for second marker
            for i, corner_2d in enumerate(second_corners):
                cv2.circle(vis_img, tuple(corner_2d.astype(int)), 3, (255, 165, 0), -1)  # Orange dots
                # Get relative coordinates in board frame (offset by [140, 180])
                if i == 0:  # top-left
                    coord = [140 - BOARD_ARUCO_SIZE/2, 180 + BOARD_ARUCO_SIZE/2, 0]
                elif i == 1:  # top-right
                    coord = [140 + BOARD_ARUCO_SIZE/2, 180 + BOARD_ARUCO_SIZE/2, 0]
                elif i == 2:  # bottom-right
                    coord = [140 + BOARD_ARUCO_SIZE/2, 180 - BOARD_ARUCO_SIZE/2, 0]
                else:  # bottom-left
                    coord = [140 - BOARD_ARUCO_SIZE/2, 180 - BOARD_ARUCO_SIZE/2, 0]
                coord_text = f"({coord[0]:.0f}, {coord[1]:.0f})"
                text_pos = (int(corner_2d[0]) + 5, int(corner_2d[1]) + 5)
                cv2.putText(vis_img, coord_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
            
            # Show visualization
            cv2.imshow("Board Detection", vis_img)
            cv2.waitKey(1)
            
            # Create Board object
            board = Board(ref_id, second_id)
            
            # Compute board transform using solvePnP with ArUco corners
            marker_positions = [
                [0, 0, 0],           # First marker at origin
                [140, 180, 0]        # Second marker offset
            ]
            
            marker_size = BOARD_ARUCO_SIZE
            marker_corners_3d = []
            for mx, my, mz in marker_positions:
                corners = np.array([
                    [mx - marker_size/2, my + marker_size/2, mz],  # top-left
                    [mx + marker_size/2, my + marker_size/2, mz],  # top-right
                    [mx + marker_size/2, my - marker_size/2, mz],  # bottom-right
                    [mx - marker_size/2, my - marker_size/2, mz],  # bottom-left
                ], dtype=np.float32)
                marker_corners_3d.extend(corners)
            
            # Combine ArUco corners
            obj_points = np.array(marker_corners_3d, dtype=np.float32)
            img_points = np.vstack([ref_corners, second_corners])
            
            # Print points for debugging
            print("\nObject points (3D):")
            for i, pt in enumerate(obj_points):
                print(f"Point {i}: {pt}")
            print("\nImage points (2D):")
            for i, pt in enumerate(img_points):
                print(f"Point {i}: {pt}")
            
            # Estimate board pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, img.camera_matrix, img.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print(f"Failed to estimate pose for marker pair ({ref_id}, {second_id})")
                continue
            
            # Convert to SE3 transform
            R, _ = cv2.Rodrigues(rvec)
            board_transform = SE3(
                translation=tvec.flatten(),
                rotation=SO3(rotation_matrix=R)
            )
            
            # Print transform for debugging
            print("\nBoard transform:")
            print(f"Translation: {board_transform.translation}")
            print(f"Rotation matrix:\n{board_transform.rotation.rot}")

            img.add_transform(f"Board {(ref_id, second_id)}", board_transform)
            board.board_transform = board_transform
            scene.add_transform(f"Board {(ref_id, second_id)}", board_transform)
            
            # Add ArUco corner points to scene
            # Project points using camera matrix
            projected_points, _ = cv2.projectPoints(
                obj_points, rvec, tvec, img.camera_matrix, img.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Reference marker corners
            for i, (corner_3d, proj_2d) in enumerate(zip(marker_corners_3d[:4], projected_points[:4])):
                point_name = f"ArUco {ref_id} Corner {i}"
                scene.add_point(point_name, proj_2d, color=(0, 1, 0))  # Green for reference marker
            
            # Second marker corners
            for i, (corner_3d, proj_2d) in enumerate(zip(marker_corners_3d[4:], projected_points[4:])):
                point_name = f"ArUco {second_id} Corner {i}"
                scene.add_point(point_name, proj_2d, color=(1, 0.65, 0))  # Orange for second marker
            
            # Load slot positions from CSV
            board_csv = f"boards/positions_plate_{ref_id:02d}-{second_id:02d}.csv"
            try:
                slot_data = np.loadtxt(board_csv, delimiter=',')
                board.slot_positions = slot_data
                
                # Calculate slot transforms
                board.slots = []
                for i, (x, y) in enumerate(board.slot_positions):
                    # Create slot transform in board coordinates
                    slot_transform = SE3(translation=np.array([x, y, 0]))
                    scene.add_transform(f"Slot {i}", slot_transform)
                    
                    # # Transform slot to camera frame using board transform
                    # slot_camera_transform = board.board_transform * slot_transform
                    # # Align slot coordinate system with reference ArUco marker convention
                    # slot_camera_transform = slot_camera_transform * SE3(
                    #     translation=np.array([0, 0, 0]),
                    #     rotation=SO3.from_euler_angles(np.deg2rad([180, 0, 0]), ["x", "y", "z"])
                    # )
                    
                    board.slots.append((i, slot_transform))
                
                print(f"Loaded {len(board.slots)} slots for board {board.pair}")
                
            except Exception as e:
                print(f"Failed to load slot positions for board {board.pair}: {e}")
                continue
            
            boards.append(board)
        scene.display()
        
        return boards
