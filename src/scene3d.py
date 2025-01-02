import numpy as np
import matplotlib.pyplot as plt
from src.se3 import SE3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from src.so3 import SO3
from src.board import Board


class Scene3D:
    def __init__(self):
        """Initialize empty 3D scene."""
        self.transforms = {}  # label -> (transform: SE3, axis_length: float)
        self.points = {}      # label -> (point: np.ndarray, color: str)
        self.connections = []  # list of (label1, label2) tuples
        self.boards = []      # list of (corners: np.ndarray, color: str)
        self.axis_length = 50.0
        self.invert_z = False
        self.z_from = False

    def invert_z_axis(self):
        """Invert the Z axis."""
        self.invert_z = True
        return self

    def z_from_zero(self):
        """Set Z axis from zero."""
        self.z_from = True
        return self

    def add_transform(self, label: str, transform: SE3):
        """Add a coordinate frame transform to the scene."""
        self.transforms[label] = transform

    def add_point(self, label: str, point: np.ndarray, color: str = 'b'):
        """Add a point to the scene."""
        self.points[label] = (point, color)

    def add_connection(self, label1: str, label2: str):
        """Add a connection between two points/transforms."""
        self.connections.append((label1, label2))

    def add_calibration_board(self, corners: np.ndarray):
        """Add a calibration board (gray rectangle) to the scene.

        Args:
            corners: Corner points of the board (4x3 array)
        """
        rect = Poly3DCollection([corners], alpha=0.3)
        rect.set_facecolor('gray')
        rect.set_edgecolor('black')
        self.boards.append((corners, 'gray'))
        return self

    def add_board(self, board: Board):
        """Add board and its slots to the scene.

        Args:
            board: Board instance with calculated slot transforms
        """
        # Add board reference marker
        self.add_transform(
            f"Board {board.pair[0]}-{board.pair[1]}",
            board.board_transform
        )

        # Calculate corners using board transform
        base_pos = board.board_transform.translation
        base_rot = board.board_transform.rotation

        # Move in local coordinates to get corners
        corners = np.array([
            base_pos,  # Origin corner
            base_pos + base_rot.act(np.array([0, 140, 0])),  # Move in y
            # Move in x and y
            base_pos + base_rot.act(np.array([180, 140, 0])),
            base_pos + base_rot.act(np.array([180, 0, 0]))  # Move in x
        ])

        # Add board rectangle
        self.boards.append((corners, 'gray'))

        # Add slots with smaller axis length
        for slot_idx, slot_transform in board.slots:
            self.add_transform(
                f"Slot {slot_idx}, board {board.pair[0]}-{board.pair[1]}",
                slot_transform
            )

        return self

    def set_axis_length(self, length: float):
        """Set length for coordinate axes."""
        self.axis_length = length
        return self

    def display(self):
        """Display the 3D scene.

        Controls:
        - Mouse: Default matplotlib 3D rotation and zoom
        - Hover over points to see coordinates
        - 'r': Reset to isometric view
        - '1': View along X axis
        - '2': View along Y axis
        - '3': View along Z axis
        - 'q': Close window
        """
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        # Setup tooltip
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        scatter_points = {}

        # Draw transforms (coordinate frames)
        for label, transform in self.transforms.items():
            # Origin point
            scatter = ax.scatter(*transform.translation,
                                 c='k', marker='o', s=50, picker=True)
            scatter_points[scatter] = (label, transform)

            # Coordinate axes
            for i, color in enumerate(['r', 'g', 'b']):
                direction = np.zeros(3)
                direction[i] = self.axis_length
                end_point = transform.translation + \
                    transform.rotation.act(direction)
                ax.plot([transform.translation[0], end_point[0]],
                        [transform.translation[1], end_point[1]],
                        [transform.translation[2], end_point[2]],
                        color=color, linewidth=2)

        # Draw points
        for label, (point, color) in self.points.items():
            scatter = ax.scatter(
                *point, c=color, marker='o', s=50, picker=True)
            scatter_points[scatter] = (label, point)

        # Draw connections
        for label1, label2 in self.connections:
            point1 = self.transforms[label1].translation if label1 in self.transforms \
                else self.points[label1][0]
            point2 = self.transforms[label2].translation if label2 in self.transforms \
                else self.points[label2][0]
            ax.plot([point1[0], point2[0]],
                    [point1[1], point2[1]],
                    [point1[2], point2[2]], 'k--', alpha=0.5)

        # Draw boards
        for corners, color in self.boards:
            # Draw a single face to ensure it's a flat rectangle
            rect = Poly3DCollection([corners], alpha=0.5)
            rect.set_facecolor(color)
            rect.set_edgecolor('black')
            rect.set_zsort('min')  # Ensure proper depth sorting
            ax.add_collection3d(rect)

        def hover(event):
            if event.inaxes != ax:
                return
            visible = False
            for sc in scatter_points:
                cont, ind = sc.contains(event)
                if cont:
                    pos = sc.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    label, data = scatter_points[sc]

                    if isinstance(data, SE3):
                        angle, axis = data.rotation.to_angle_axis()
                        text = (f"{label}\n"
                                f"Position:\n"
                                f"X: {data.translation[0]:.2f}\n"
                                f"Y: {data.translation[1]:.2f}\n"
                                f"Z: {data.translation[2]:.2f}\n"
                                f"Rotation:\n"
                                f"Angle: {np.rad2deg(angle):.1f}°\n"
                                f"Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")
                    else:
                        text = (f"{label}\n"
                                f"X: {data[0]:.2f}\n"
                                f"Y: {data[1]:.2f}\n"
                                f"Z: {data[2]:.2f}")

                    annot.set_text(text)
                    visible = True
                    break

            if visible != annot.get_visible():
                annot.set_visible(visible)
                fig.canvas.draw_idle()

        def on_key(event):
            if event.key == 'q':
                plt.close('all')
            elif event.key == 'r':
                ax.view_init(elev=30, azim=45)
            elif event.key == '1':
                ax.view_init(elev=0, azim=0)
            elif event.key == '2':
                ax.view_init(elev=0, azim=90)
            elif event.key == '3':
                ax.view_init(elev=90, azim=0)
            fig.canvas.draw()

        # Setup controls
        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Set axes properties
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')

        # Make axes equal
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        radius = np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        if self.z_from:
            ax.set_zlim3d([0, radius])
        else:
            ax.set_zlim3d([-radius, radius])

        # View settings
        ax.view_init(elev=30, azim=45)
        if self.invert_z:
            ax.invert_zaxis()

        plt.show()

    def add_robot(self, box, q=None):
        """Add robot to the scene."""
        if q is None:
            q = box.robot.q_home

        # Add base frame
        self.add_transform("Base", SE3())
        current_transform = np.eye(4)

        # Add each joint
        for i, (d, a, alpha, theta, qi) in enumerate(zip(
            box.robot.dh_d,
            box.robot.dh_a,
            box.robot.dh_alpha,
            box.robot.dh_theta_off if hasattr(
                box.robot, 'dh_theta_off') else box.robot.dh_offset,
            q
        )):
            # Update transform
            ct, st = np.cos(qi + theta), np.sin(qi + theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            current_transform = current_transform @ np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

            # Add joint
            transform = SE3.from_matrix(current_transform)
            if i < len(q) - 1:
                self.add_point(f"Joint_{i+1}", transform.translation, 'b')
            else:
                self.add_transform(f"Joint_{i+1}", transform)

            # Add connection to previous joint
            if i == 0:
                self.add_connection("Base", f"Joint_1")
            else:
                self.add_connection(f"Joint_{i}", f"Joint_{i+1}")

    def add_calibration(self, corners3d, grid, rvecs, tvecs):
        """Add calibration setup to the scene."""
        self.add_transform("Camera", SE3())

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Transform board corners to camera frame
            R, _ = cv2.Rodrigues(rvec)
            pts = corners3d.reshape(-1, 3)
            pts_transformed = (R @ pts.T + tvec).T
            pts_transformed = pts_transformed.reshape(grid[1], grid[0], 3)

            # Add board
            corners = np.array([
                pts_transformed[0, 0],      # Top-left
                pts_transformed[0, -1],     # Top-right
                pts_transformed[-1, -1],    # Bottom-right
                pts_transformed[-1, 0]      # Bottom-left
            ])
            self.add_board(corners)

            # Add board frame
            center = pts_transformed.mean(axis=(0, 1))
            self.add_transform(f"Board_{i}", SE3(
                translation=center, rotation=SO3(R)))
