from basler_camera import BaslerCamera
import cv2
import numpy as np
import os
from so3 import SO3
from se3 import SE3
from enums import RobotType


class Camera():
    def __init__(self, robot_type: RobotType):
        self.camera = BaslerCamera()

        if robot_type == RobotType.CRS93:
            self.camera.connect_by_ip("192.168.137.107")
            self.camera.connect_by_name("camera-crs93")
        elif robot_type == RobotType.CRS97:
            self.camera.connect_by_ip("192.168.137.106")
            self.camera.connect_by_name("camera-crs97")
        elif robot_type == RobotType.RV6S:
            self.camera.connect_by_ip("192.168.137.109")
            self.camera.connect_by_name("camera-rv6s")
        else:
            raise ValueError(f"Invalid robot type: {robot_type}")

        self.camera.open()
        self.camera.set_parameters()

        self.camera_matrix = None
        if os.path.exists("calibration/camera_matrix.npy"):
            self.camera_matrix = np.load("calibration/camera_matrix.npy")
        self.dist_coeffs = None
        if os.path.exists("calibration/dist_coeffs.npy"):
            self.dist_coeffs = np.load("calibration/dist_coeffs.npy")


    def grab_image(self):
        return self.camera.grab_image()

    @staticmethod
    def save_image(image: np.ndarray, filename: str):
        cv2.imwrite(filename, image)

    def get_arucos(self, image: np.ndarray, marker_size_mm: float, dict_size: int) -> dict[int, SE3]:
        """Detect ArUco markers in the image and return their poses.
        
        Args:
            image: Input image
            marker_size_mm: Size of the ArUco marker in millimeters
            dict_size: ArUco dictionary size (e.g., cv2.aruco.DICT_4X4_50)
            
        Returns:
            Dictionary mapping marker IDs to their SE3 poses in camera frame
        """
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_size)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


        # Detect markers
        corners, ids, rejected = detector.detectMarkers(image)
        
        if ids is None:
            return {}

        # Get poses for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size_mm, self.camera_matrix, self.dist_coeffs)

        # Convert to dictionary of SE3 transforms
        poses = {}
        for i in range(len(ids)):
            marker_id = int(ids[i])
            
            # Convert rotation vector to matrix
            rot_mat, _ = cv2.Rodrigues(rvecs[i])
            
            # Create SE3 transform
            rotation = SO3(rot_mat)
            translation = tvecs[i].flatten()
            poses[marker_id] = SE3(translation=translation, rotation=rotation)

        return poses

    def close(self):
        self.camera.close()

    def display_image(self, img):
        """Display image in a window with controlled size and position.
        Wait for 'q' key to close.
        
        Args:
            img: Image to display
        """
        # Create window and set its position to top-left corner
        window_name = "Camera View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, 0)
        
        # Set window size explicitly
        display_width = 1200  # Adjust this value as needed
        height, width = img.shape[:2]
        display_height = int(height * (display_width / width))
        cv2.resizeWindow(window_name, display_width, display_height)
        
        cv2.imshow(window_name, img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

    def draw_arucos(self, image: np.ndarray, poses: dict[int, SE3], axis_length: float = 30.0) -> np.ndarray:
        """Draw detected ArUco markers with their poses on the image.
        
        Args:
            image: Image to draw on
            poses: Dictionary of marker IDs and their SE3 poses
            axis_length: Length of coordinate axes to draw (in mm)
            
        Returns:
            Image with drawn markers
        """
        img_copy = image.copy()
        
        for marker_id, pose in poses.items():
            img_copy = self.draw_transform(img_copy, pose, f"ID: {marker_id}", axis_length)
            
        return img_copy

    def project_point(self, point_3d):
        """Project a 3D point to 2D image coordinates using camera matrix.
        
        Args:
            point_3d: 3D point in camera coordinates [x, y, z]
            
        Returns:
            2D point in image coordinates [x, y]
        """
        # Reshape point for cv2.projectPoints
        points_3d = np.float32([[point_3d]])
        
        # Get rotation and translation vectors (zero for camera frame)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        
        # Project points
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, 
                                       self.camera_matrix, self.dist_coeffs)
        
        return points_2d[0][0]

    def draw_transform(self, image: np.ndarray, transform: SE3, label: str, 
                      axis_length: float = 30.0) -> np.ndarray:
        """Draw a single SE3 transform on the image (must be in camera reference frame).
        
        Args:
            image: Image to draw on
            transform: SE3 transform in camera reference frame
            label: Text label to display next to the transform
            axis_length: Length of coordinate axes to draw (in mm)
            
        Returns:
            Image with drawn transform
        """
        if transform is None:
            return image
        
        img_copy = image.copy()
        
        # Get rotation and translation
        rvec, _ = cv2.Rodrigues(transform.rotation.rot)
        tvec = transform.translation
        
        # Draw coordinate axes
        cv2.drawFrameAxes(img_copy, self.camera_matrix, self.dist_coeffs, 
                          rvec, tvec, axis_length)
        
        # Project origin point to get text position
        point_2d = self.project_point(transform.translation)
        
        # Ensure coordinates are integers and in the correct format
        x = int(round(float(point_2d[0])))
        y = int(round(float(point_2d[1])))
        
        # Make sure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # Create proper tuple for text position
        text_pos = (x, y)
        
        # Draw label
        cv2.putText(img_copy, label, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy

    @staticmethod
    def display_transforms_3d(transforms: dict[str, SE3], show_camera: bool = True) -> None:
        """Display SE3 transforms in 3D plot. Close with 'q' key.
        
        Args:
            transforms: Dictionary mapping labels to SE3 transforms
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_camera:
            transforms["Camera"] = SE3()

        # Draw each transform
        for label, transform in transforms.items():
            if transform is None:
                continue
            # Plot origin point
            ax.scatter(*transform.translation, marker='o', label=label)
            
            # Draw coordinate axes
            axis_length = 30  # mm
            colors = ['r', 'g', 'b']  # x, y, z
            for i in range(3):
                direction = np.zeros(3)
                direction[i] = axis_length
                end_point = transform.act(direction)
                ax.plot([transform.translation[0], end_point[0]],
                       [transform.translation[1], end_point[1]],
                       [transform.translation[2], end_point[2]],
                       color=colors[i])
        
        # Get axis limits and find the maximum range
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        
        max_range = max([
            x_lim[1] - x_lim[0],
            y_lim[1] - y_lim[0],
            z_lim[1] - z_lim[0]
        ])
        
        # Calculate centers
        x_center = (x_lim[1] + x_lim[0]) / 2
        y_center = (y_lim[1] + y_lim[0]) / 2
        z_center = (z_lim[1] + z_lim[0]) / 2
        
        # Set new limits
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        # Set labels
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        
        # Add legend
        ax.legend()
        
        # Show plot and wait for 'q'
        plt.show(block=False)
        
        def on_key(event):
            if event.key == 'q':
                plt.close('all')
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)