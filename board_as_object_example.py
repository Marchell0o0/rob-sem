import os
import cv2
import numpy as np
import os


def resize_to_fit_window(image, target_width=1280, target_height=960):
    """
    Масштабирует изображение, чтобы оно вписывалось в заданные размеры (1280x960),
    сохраняя пропорции.
    """
    original_height, original_width = image.shape[:2]

    # Вычисляем коэффициент масштабирования
    scale = min(target_width / original_width, target_height / original_height)

    # Новые размеры изображения
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Масштабируем изображение
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def create_custom_aruco_board():
    """
    Создаём доску 200×240 мм с двумя ArUco-маркерами (36×36 мм):
     - Маркер ID=3 в левом нижнем углу
     - Маркер ID=4 в правом верхнем углу
    Координатная система доски:
     - Начало в левом нижнем углу доски
     - X - вдоль нижнего края доски (слева направо, 240 мм)
     - Y - вдоль левого края доски (снизу вверх, 200 мм)
     - Z - перпендикулярно плоскости доски
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Размеры маркера
    marker_size = 36.0  # mm

    # Маркер ID=3 в левом нижнем углу (начало координат)
    # Начинаем с левого нижнего угла маркера и идем по часовой стрелке
    corners_0 = np.array([
        [0.0, 0.0, 0.0],              # Левый нижний угол
        [marker_size, 0.0, 0.0],       # Правый нижний угол
        [marker_size, marker_size, 0.0],# Правый верхний угол
        [0.0, marker_size, 0.0]        # Левый верхний угол
    ], dtype=np.float32)

    # Маркер ID=4 в правом верхнем углу
    # Позиция: 180 мм вправо и 140 мм вверх от начала координат
    x_offset = 240.0 - marker_size  # Правый край доски минус размер маркера
    y_offset = 200.0 - marker_size  # Верхний край доски минус размер маркера
    corners_1 = np.array([
        [x_offset, y_offset, 0.0],              # Левый нижний угол
        [x_offset + marker_size, y_offset, 0.0], # Правый нижний угол
        [x_offset + marker_size, y_offset + marker_size, 0.0], # Правый верхний угол
        [x_offset, y_offset + marker_size, 0.0]  # Левый верхний угол
    ], dtype=np.float32)

    obj_points = [corners_0, corners_1]
    ids = np.array([3, 4], dtype=np.int32)

    board = cv2.aruco.Board(objPoints=obj_points, dictionary=aruco_dict, ids=ids)
    return board, aruco_dict


def main():
    calib_folder = os.path.join("calibration", "calibration_data")
    camera_matrix_path = os.path.join(calib_folder, "camera_matrix.npy")
    dist_coefs_path = os.path.join(calib_folder, "dist_coeffs.npy")

    camera_matrix = np.load(camera_matrix_path)
    dist_coefs = np.load(dist_coefs_path)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coeffs:\n", dist_coefs)

    board, aruco_dict = create_custom_aruco_board()

    boards_dataset_folder = "boards_dataset"
    test_image_names = os.listdir(boards_dataset_folder)
    # print("test_image_names:", test_image_names)

    for test_image_name in test_image_names:
        test_image_path = os.path.join(boards_dataset_folder, test_image_name)  

        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Не удалось загрузить {test_image_path}. Проверьте путь.")
            return

        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
            # Debug: Estimate and display pose for each marker individually
            for i in range(len(ids)):
                marker_corners = corners[i]
                marker_id = ids[i]
                
                # Get individual marker pose
                marker_length = 36.0  # marker size in mm
                rvec_marker = np.zeros((3, 1), dtype=np.float64)
                tvec_marker = np.zeros((3, 1), dtype=np.float64)
                cv2.aruco.estimatePoseSingleMarkers([marker_corners], marker_length, 
                                                  camera_matrix, dist_coefs, 
                                                  rvec_marker, tvec_marker)
                
                # Draw axes for individual marker (in different color - green)
                cv2.drawFrameAxes(image, camera_matrix, dist_coefs, 
                                rvec_marker, tvec_marker, marker_length, 2)
                
                print(f"Marker {marker_id} pose:")
                print(f"rvec: {rvec_marker.ravel()}")
                print(f"tvec: {tvec_marker.ravel()}")

            # Now estimate board pose
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)

            used_markers, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners, ids, board, camera_matrix, dist_coefs, rvec, tvec
            )

            if used_markers > 0:
                print(f"Обнаружено {used_markers} маркер(ов), совпадающих с доской.")
                print("Board pose:")
                print("rvec:", rvec.ravel())
                print("tvec:", tvec.ravel())

                # Draw board axes in red (thicker)
                cv2.drawFrameAxes(image, camera_matrix, dist_coefs, rvec, tvec, 100, 3)

                # Масштабируем изображение до большего разрешения
                resized_image = resize_to_fit_window(image, target_width=1280, target_height=960)

                # Add text labels for axes
                cv2.putText(resized_image, "Individual markers (green)", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(resized_image, "Board pose (red)", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Показываем результат
                cv2.imshow("Board pose estimation", resized_image)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                cv2.destroyAllWindows()
            else:
                print("Маркер(ы) нашли, но ни один не совпал с ID доски.")
        else:
            print("ArUco маркеры не найдены на изображении.")


if __name__ == "__main__":
    main()
