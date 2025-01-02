import cv2
import numpy as np
from src.se3 import SE3
from src.so3 import SO3
from src.board import Board


class CameraImage:
    def __init__(self, camera_matrix=None, dist_coeffs=None, display_width: int = 1920):
        """Initialize empty image scene.

        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            display_width: Width for display window in pixels (height maintains aspect ratio)
        """
        self.image = None
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.display_width = display_width
        self.transforms = {}  # label -> SE3
        self.points = {}      # label -> (point2d, color)
        self.texts = {}       # label -> (position, text, color)

    def set_image(self, image: np.ndarray):
        """Set base image to draw on."""
        self.image = image.copy()
        return self

    def save_image(self, filename: str):
        cv2.imwrite(filename, self.image)

    def add_transform(self, label: str, transform: SE3, axis_length: float = 30.0):
        """Add and draw a coordinate frame transform (must be in camera frame)."""
        # Get rotation and translation
        rvec, _ = cv2.Rodrigues(transform.rotation.rot)
        tvec = transform.translation

        # Draw coordinate axes
        cv2.drawFrameAxes(self.image, self.camera_matrix, self.dist_coeffs,
                          rvec, tvec, axis_length)

        # Project origin for label
        point_2d = self.project_point(transform.translation)
        x = int(round(point_2d[0]))
        y = int(round(point_2d[1]))

        # Draw label
        cv2.putText(self.image, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return self

    def add_point(self, label: str, point2d: np.ndarray, color: tuple = (0, 255, 0)):
        """Add and draw a 2D point."""
        x = int(round(point2d[0]))
        y = int(round(point2d[1]))
        cv2.circle(self.image, (x, y), 3, color, -1)
        return self

    def add_text(self, label: str, position: np.ndarray, text: str,
                 color: tuple = (0, 255, 0)):
        """Add and draw text at specified position."""
        x = int(round(position[0]))
        y = int(round(position[1]))
        cv2.putText(self.image, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return self

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Project a 3D point to 2D image coordinates."""
        points_3d = np.float32([[point_3d]])
        points_2d, _ = cv2.projectPoints(
            points_3d, np.zeros(3), np.zeros(3),
            self.camera_matrix, self.dist_coeffs)
        return points_2d[0][0]

    def set_display_size(self, width: int):
        """Set display window width in pixels (height maintains aspect ratio)."""
        self.display_width = width
        return self

    def display(self, window_name: str = "Image Scene"):
        """Display the scene.

        Controls:
        - 'q': Close window

        Args:
            window_name: Name of the window
        """
        if self.image is None:
            raise ValueError("No image set. Call set_image() first.")

        # Create window and set its position
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, 0)

        # Set window size
        height, img_width = self.image.shape[:2]
        display_height = int(height * (self.display_width / img_width))
        cv2.resizeWindow(window_name, self.display_width, display_height)

        # Display
        cv2.imshow(window_name, self.image)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    def draw_arucos(self, poses: dict[int, SE3], axis_length: float = 30.0) -> 'CameraImage':
        """Draw ArUco markers with their poses.

        Args:
            poses: Dictionary of marker IDs and their SE3 poses
            axis_length: Length of coordinate axes to draw

        Returns:
            self for chaining
        """
        for marker_id, pose in poses.items():
            self.add_transform(f"ID: {marker_id}", pose, axis_length)
        return self

    def get_arucos(self, marker_size_mm: float, dict_size: int) -> dict[int, SE3]:
        """Detect ArUco markers in the image and return their poses.

        Args:
            image: Input image
            marker_size_mm: Size of the ArUco marker in millimeters
            dict_size: ArUco dictionary size (e.g., cv2.aruco.DICT_4X4_50)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            Dictionary mapping marker IDs to their SE3 poses in camera frame
        """
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_size)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(self.image)

        if ids is None:
            return {}

        # Create object points for a single marker
        objPoints = np.array([
            [-marker_size_mm/2, marker_size_mm/2, 0],
            [marker_size_mm/2, marker_size_mm/2, 0],
            [marker_size_mm/2, -marker_size_mm/2, 0],
            [-marker_size_mm/2, -marker_size_mm/2, 0]
        ])

        # Get poses for each marker
        poses = {}
        for i, corner in enumerate(corners):
            marker_id = int(ids[i])

            # Get pose for this marker
            success, rvec, tvec = cv2.solvePnP(
                objPoints,
                corner,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Convert rotation vector to matrix
                rot_mat, _ = cv2.Rodrigues(rvec)

                # Create SE3 transform
                rotation = SO3(rot_mat)
                translation = tvec.flatten()
                poses[marker_id] = SE3(
                    translation=translation, rotation=rotation)

        return poses

    def detect_boards(self) -> list[Board]:
        """Detect boards in the image using ArUco markers.

        Returns:
            List of detected boards
        """
        # Detect ArUco markers
        poses = self.get_arucos(
            Board.MARKER_SIZE,
            cv2.aruco.DICT_4X4_50
        )

        # Create boards from detected markers
        boards = Board.create_boards_from_transforms(poses)
        print(f"Found {len(boards)} boards")
        return boards

    def draw_board_slots(self, board: Board) -> 'CameraImage':
        """Draw board slots on the image.

        Args:
            board: Board instance with calculated slot transforms

        Returns:
            self for chaining
        """
        for slot_idx, slot_transform in board.slots:
            # Draw slot transform with empty label (we'll add our own label)
            self.add_transform("", slot_transform, axis_length=20.0)

            # Project slot center and add label
            point_2d = self.project_point(slot_transform.translation)
            self.add_text(
                f"slot_{slot_idx}",
                point_2d + np.array([-30, -10]),  # Offset text slightly
                f"Slot {slot_idx}",
                color=(0, 0, 255)  # Red for slots
            )
        return self
