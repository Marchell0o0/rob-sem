import cv2
import numpy as np

# -------------------------------------------------
# 1) ФУНКЦИЯ ПАРСИНГА CSV
# -------------------------------------------------
def parse_csv(csv_path):
    """
    Считывает CSV, где:
      - первая строка: два идентификатора маркеров (например "1,2"),
      - остальные строки: координаты (x,y) для мест под кубики.
    Возвращает (marker_ids, cube_positions).
    Пример файла .csv:
        1,2
        0,80
        80,60
        100,140
        140,0
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Первая строка => 2 маркерных ID
    marker_ids = [int(x) for x in lines[0].split(',')]

    # Остальные строки => координаты для кубиков
    cube_positions = []
    for line in lines[1:]:
        x_str, y_str = line.split(',')
        x = float(x_str)
        y = float(y_str)
        cube_positions.append((x, y))

    return marker_ids, cube_positions


# -------------------------------------------------
# 2) ФУНКЦИИ РИСОВАНИЯ ДОСКИ, КУБИКОВ, НАКЛОНА И Т.Д.
# -------------------------------------------------

def create_board_sprite(
    board_size=(400, 400), 
    marker_dict=None, 
    marker_coords=None,   # Список (mx, my, msize, marker_id)
    cube_coords=None      # Список (cx, cy, size_гнезда)
):
    """
    Генерирует спрайт доски (BGR-изображение) с:
      - ArUco-маркерами (по заданным координатам marker_coords),
      - 'гнёздами' (прямоугольниками) под кубики (cube_coords).

    board_size = (width, height) - размер доски в пикселях.
    marker_coords = [(mx, my, msize, marker_id), ...].
    cube_coords   = [(cx, cy, csize), ...].
    """
    w, h = board_size
    # Создаём белое поле
    board_img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Сначала рисуем гнёзда (если есть)
    if cube_coords:
        for (cx, cy, sz) in cube_coords:
            pt1 = (int(cx - sz/2), int(cy - sz/2))
            pt2 = (int(cx + sz/2), int(cy + sz/2))
            cv2.rectangle(board_img, pt1, pt2, (200, 200, 200), 2)

    # Рисуем ArUco-маркеры (если coords есть)
    if marker_dict is not None and marker_coords is not None:
        for (mx, my, msize, mid) in marker_coords:
            # Рисуем маркер именно с ID=mid
            aruco_marker = np.zeros((msize, msize), dtype=np.uint8)
            aruco_marker = cv2.aruco.drawMarker(marker_dict, mid, msize, aruco_marker, 1)

            # Координаты вставки
            x1 = int(mx - msize/2)
            y1 = int(my - msize/2)
            x2 = x1 + msize
            y2 = y1 + msize

            # Защита от выхода за границы
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > w: x2 = w
            if y2 > h: y2 = h

            marker_bgr = cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2BGR)
            board_img[y1:y2, x1:x2] = marker_bgr[:y2-y1, :x2-x1]

    return board_img


def create_cube_sprite(size=50, color=(150, 120, 100)):
    """
    Создаёт простой квадрат (BGR) для имитации 'кубика',
    добавляя шум для разнообразия.
    """
    cube_img = np.full((size, size, 3), color, dtype=np.uint8)
    # Небольшой шум
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    cube_img = cv2.add(cube_img, noise)
    return cube_img


def place_cube_on_board(board_img, cube_img, center):
    """
    Размещает 'cube_img' на 'board_img',
    центрируя кубик в заданной точке center=(x, y).
    """
    h_board, w_board = board_img.shape[:2]
    h_cube, w_cube = cube_img.shape[:2]

    x_center, y_center = center
    x1 = int(x_center - w_cube/2)
    y1 = int(y_center - h_cube/2)
    x2 = x1 + w_cube
    y2 = y1 + h_cube

    # Ограничим границы
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > w_board: x2 = w_board
    if y2 > h_board: y2 = h_board

    cube_part = cube_img[0:(y2-y1), 0:(x2-x1)]
    board_img[y1:y2, x1:x2] = cube_part


def generate_uniform_background(width, height, base_color=220):
    """
    Генерирует равномерный фон (однотонный).
    """
    bg = np.full((height, width, 3), base_color, dtype=np.uint8)
    return bg


def tilt_board(img, angle_degs=7, output_size=(1920, 1080)):
    """
    Имитация наклона доски:
      - angle_degs: угол (7 или 15),
      - output_size: (width, height).
    Возвращает изображение размера output_size после перспективной трансформации.
    """
    out_w, out_h = output_size
    src_h, src_w = img.shape[:2]

    # Исходные углы
    src_pts = np.float32([
        [0,      0],
        [src_w,  0],
        [src_w,  src_h],
        [0,      src_h]
    ])

    # Угол в радианы через NumPy
    rad = np.deg2rad(angle_degs)
    cos_angle = np.cos(rad)
    # Уменьшим высоту при наклоне
    new_h = src_h * cos_angle

    # Смещаем доску, чтобы была в кадре
    top_margin_x = 100
    top_margin_y = 100

    dst_pts = np.float32([
        [top_margin_x,                 top_margin_y],
        [top_margin_x + src_w,         top_margin_y],
        [top_margin_x + src_w,         top_margin_y + new_h],
        [top_margin_x,                 top_margin_y + new_h]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        img, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return warped


def add_noise_and_blur(img):
    """
    Лёгкий гауссов шум и размытие.
    """
    # Преобразуем к float32
    noisy = img.astype(np.float32)
    # Генерируем шум
    gauss_noise = np.random.normal(0, 5, img.shape)
    # Добавляем
    noisy += gauss_noise
    # Обрезаем [0..255], переводим в uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    # Размываем
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)
    return blurred


# -------------------------------------------------
# 3) ГЛАВНАЯ ФУНКЦИЯ
# -------------------------------------------------
def main():
    # Пусть ваш CSV лежит рядом и называется "board_data.csv"
    csv_path = "boards/positions_plate_01-02.csv"

    # Считываем данные из CSV
    marker_ids, cube_positions = parse_csv(csv_path)
    print("Прочитаны маркеры:", marker_ids)
    print("Прочитаны координаты кубиков:", cube_positions)

    # Подготовим координаты маркеров для доски. 
    # Допустим, у нас доска 400х400. 
    # Скажем, 1-й маркер (marker_ids[0]) пойдёт в левый нижний угол,
    # а 2-й (marker_ids[1]) — в правый верхний. 
    # Размер маркера (пусть 50 px).
    # В реальном коде вы можете вычислять точные координаты по чертежу.
    board_w, board_h = 400, 400
    marker_size = 50

    # Пример расстановки (нижний левый, верхний правый):
    # (mx, my, msize, marker_id)
    marker_coords = [
        (marker_size/2, board_h - marker_size/2, marker_size, marker_ids[0]),  # левый нижний
        (board_w - marker_size/2, marker_size/2, marker_size, marker_ids[1])  # правый верхний
    ]

    # Преобразуем координаты для кубиков:
    # В CSV указаны (x, y), но вы можете захотеть интерпретировать их 
    # как именно координаты в пикселях на "доске".
    # Пусть гнездо будет размером 40 px:
    csize = 40
    cube_coords = []
    for (cx, cy) in cube_positions:
        cube_coords.append((cx, cy, csize))

    # Создаём словарь ArUco
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


    # Генерируем изображение доски
    board_img = create_board_sprite(
        board_size=(board_w, board_h),
        marker_dict=aruco_dict,
        marker_coords=marker_coords,
        cube_coords=cube_coords
    )

    # Случайно решаем, сколько кубиков реально поставить
    n_cubes = np.random.randint(0, len(cube_coords) + 1)
    chosen_idxs = np.random.choice(len(cube_coords), size=n_cubes, replace=False)

    # Раскладываем кубики
    for idx in chosen_idxs:
        (cx, cy, sz) = cube_coords[idx]
        # Случайный реальный размер кубика (±20%)
        real_cube_size = np.random.randint(int(sz*0.8), int(sz*1.2) + 1)
        color = (
            np.random.randint(80, 181),
            np.random.randint(80, 181),
            np.random.randint(80, 181)
        )
        cube_img = create_cube_sprite(size=real_cube_size, color=color)
        # Ставим кубик так, чтобы его центр пришёлся в (cx, cy)
        place_cube_on_board(board_img, cube_img, (cx, cy))

    # Генерируем равномерный фон 1920x1080
    final_w, final_h = 1920, 1080
    background = generate_uniform_background(final_w, final_h, base_color=220)

    # Выбираем угол наклона (7 или 15)
    angle = np.random.choice([7, 15])
    # "Наклоняем" доску
    tilted_board = tilt_board(board_img, angle_degs=angle, output_size=(final_w, final_h))

    # Наложим доску на фон (белый пиксель => фон)
    result = background.copy()
    mask_white = np.all(tilted_board == [255,255,255], axis=-1)
    result[~mask_white] = tilted_board[~mask_white]

    # Добавим шум и размытие
    final = add_noise_and_blur(result)

    # Сохраним итог
    out_name = "synthetic_from_csv.png"
    cv2.imwrite(out_name, final)
    print(f"Готово! Сохранено как {out_name}")

# --------------------------------
# Запуск
# --------------------------------
if __name__ == "__main__":
    main()
