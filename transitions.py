import random

import cv2
import numpy as np

from options import HEIGHT, WIDTH

TRANSITIONS = [
    "Spiral",
    "Square",
    "Diamond",
    "Left",
    "Right",
    "Bounce",
    "Fly3D",
]


# ============================================================
# ЭТАЛОННЫЕ ТРАЕКТОРИИ ПЕРЕХОДОВ
# ============================================================
# Координаты взяты из присланных пользователем роликов 1920x1080.
# Время эталона здесь не важно: переход просто сжимается до
# TRANSITION_DURATION, сохраняя саму траекторию.

# Spiral: x1, y1, x2, y2. Белый прямоугольник эталона = область,
# куда показывается новое изображение.
SPIRAL_KEYFRAMES = [
    (30, 0.000000, 0.183333, 0.013021, 0.689815),
    (33, 0.000000, 0.000000, 0.083333, 0.521296),
    (36, 0.000000, 0.000000, 0.195312, 0.379630),
    (39, 0.000000, 0.000000, 0.339583, 0.273148),
    (42, 0.000000, 0.000000, 0.504167, 0.208333),
    (45, 0.068750, 0.000000, 0.677083, 0.187963),
    (48, 0.217188, 0.000000, 0.846875, 0.211111),
    (51, 0.351042, 0.000000, 1.000000, 0.274074),
    (54, 0.460417, 0.000000, 1.000000, 0.369444),
    (57, 0.538021, 0.000000, 1.000000, 0.488889),
    (60, 0.580729, 0.000000, 1.000000, 0.621296),
    (63, 0.586979, 0.025926, 1.000000, 0.756481),
    (66, 0.559375, 0.133333, 1.000000, 0.885185),
    (69, 0.502604, 0.224074, 1.000000, 0.996296),
    (72, 0.423958, 0.292593, 1.000000, 1.000000),
    (75, 0.332292, 0.334259, 1.000000, 1.000000),
    (78, 0.236458, 0.348148, 1.000000, 1.000000),
    (81, 0.144792, 0.335185, 0.998958, 1.000000),
    (84, 0.065625, 0.299074, 0.939583, 1.000000),
    (87, 0.004167, 0.246296, 0.898958, 1.000000),
    (90, 0.000000, 0.185185, 0.879167, 1.000000),
    (93, 0.000000, 0.123148, 0.881771, 1.000000),
    (96, 0.000000, 0.067593, 0.904167, 1.000000),
    (99, 0.000000, 0.025926, 0.942708, 1.000000),
    (102, 0.000000, 0.001852, 0.991146, 0.998148),
    (104, 0.000000, 0.000000, 1.000000, 1.000000),
]

# ============================================================
# Fly3D
# ============================================================
# Картинка прилетает справа в перспективе.
# Она НЕ вращается вокруг Z, поэтому всегда остаётся нормально
# ориентированной относительно экрана.
#
# В начале:
#   - картинка находится справа за пределами экрана;
#   - немного сжата по ширине перспективой;
#   - вертикальная ориентация сохраняется.
#
# В конце:
#   - картинка полностью закрывает экран.
# ============================================================


def fly3d_ease(t):
    # Плавное ускорение и торможение.
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def transition_fly3d(old, new, t):
    p = fly3d_ease(t)

    # --------------------------------------------------------
    # Начальное положение.
    #
    # Картинка находится справа от кадра и немного наклонена
    # перспективой вокруг вертикальной оси.
    #
    # ВАЖНО:
    # верхний и нижний края остаются горизонтальными.
    # Поэтому изображение не ложится на бок.
    # --------------------------------------------------------

    start = np.array(
        [
            [1.08, 0.17],  # левый верхний
            [1.40, 0.27],  # правый верхний
            [1.40, 0.73],  # правый нижний
            [1.08, 0.83],  # левый нижний
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Конечное положение:
    # обычная картинка во весь экран.
    # --------------------------------------------------------

    end = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dst = start * (1.0 - p) + end * p

    dst[:, 0] *= WIDTH
    dst[:, 1] *= HEIGHT

    dst = dst.astype(np.float32)

    # Если плоскость ещё полностью вне экрана,
    # просто оставляем старый кадр.
    area = abs(cv2.contourArea(dst))

    if area < 20:
        return old

    src = np.array(
        [
            [0, 0],
            [WIDTH - 1, 0],
            [WIDTH - 1, HEIGHT - 1],
            [0, HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    moving = cv2.warpPerspective(
        new,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)

    frame = old.copy()

    frame[mask > 0] = moving[mask > 0]

    return frame


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def bounce_ease(t):
    if t < 1 / 2.75:
        return 7.5625 * t * t
    if t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    if t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    t -= 2.625 / 2.75
    return 7.5625 * t * t + 0.984375


def transform_image(image, matrix):
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def interpolate_keyframes(keyframes, t):
    """Линейно интерполирует эталонные координаты по t=0..1."""
    if t <= 0:
        return np.asarray(keyframes[0][1:], dtype=np.float32)
    if t >= 1:
        return np.asarray(keyframes[-1][1:], dtype=np.float32)

    source_first = keyframes[0][0]
    source_last = keyframes[-1][0]
    source_frame = source_first + (source_last - source_first) * t

    for i in range(1, len(keyframes)):
        if source_frame <= keyframes[i][0]:
            a = keyframes[i - 1]
            b = keyframes[i]
            p = (source_frame - a[0]) / max(1e-6, b[0] - a[0])
            av = np.asarray(a[1:], dtype=np.float32)
            bv = np.asarray(b[1:], dtype=np.float32)
            return av * (1 - p) + bv * p

    return np.asarray(keyframes[-1][1:], dtype=np.float32)


def interpolate_smooth_keyframes(keyframes, t):
    """Плавно интерполирует небольшое число кейфреймов."""
    if t <= 0:
        return np.asarray(keyframes[0][1:], dtype=np.float32)

    if t >= 1:
        return np.asarray(keyframes[-1][1:], dtype=np.float32)

    source_first = keyframes[0][0]
    source_last = keyframes[-1][0]
    source_frame = source_first + (source_last - source_first) * t

    index = 1

    for i in range(1, len(keyframes)):
        if source_frame <= keyframes[i][0]:
            index = i
            break

    a = keyframes[index - 1]
    b = keyframes[index]

    p = (source_frame - a[0]) / max(1e-6, b[0] - a[0])

    # Крайние точки повторяем, чтобы не было рывка на концах.
    p0 = np.asarray(
        keyframes[max(0, index - 2)][1:],
        dtype=np.float32,
    )

    p1 = np.asarray(a[1:], dtype=np.float32)
    p2 = np.asarray(b[1:], dtype=np.float32)

    p3 = np.asarray(
        keyframes[min(len(keyframes) - 1, index + 1)][1:],
        dtype=np.float32,
    )

    # Catmull-Rom
    result = 0.5 * (
        2 * p1
        + (-p0 + p2) * p
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * p * p
        + (-p0 + 3 * p1 - 3 * p2 + p3) * p * p * p
    )

    return np.clip(result, 0.0, 1.0)


def transition_spiral(old, new, t):
    """
    Новое изображение само перемещается по спиральной
    траектории, одновременно изменяя размер.
    """

    x1, y1, x2, y2 = interpolate_keyframes(
        SPIRAL_KEYFRAMES,
        t,
    )

    x1 = int(round(x1 * WIDTH))
    y1 = int(round(y1 * HEIGHT))
    x2 = int(round(x2 * WIDTH))
    y2 = int(round(y2 * HEIGHT))

    x1 = max(0, min(WIDTH - 1, x1))
    y1 = max(0, min(HEIGHT - 1, y1))
    x2 = max(x1 + 2, min(WIDTH, x2))
    y2 = max(y1 + 2, min(HEIGHT, y2))

    # Размер и положение самого "летящего" видео.
    dst = np.array(
        [
            [x1, y1],
            [x2 - 1, y1],
            [x2 - 1, y2 - 1],
            [x1, y2 - 1],
        ],
        dtype=np.float32,
    )

    src = np.array(
        [
            [0, 0],
            [WIDTH - 1, 0],
            [WIDTH - 1, HEIGHT - 1],
            [0, HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    moving = cv2.warpPerspective(
        new,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Область самого движущегося изображения.
    mask = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        mask,
        np.round(dst).astype(np.int32),
        255,
    )

    frame = old.copy()
    frame[mask > 0] = moving[mask > 0]

    return frame


def transition_frame(old, new, name, t):
    t = max(0.0, min(1.0, t))

    if name == "Spiral":
        return transition_spiral(old, new, t)

    if name == "Fly3D":
        return transition_fly3d(old, new, t)

    if name == "Left":
        x = int(WIDTH * (1 - ease_in_out(t)))
        frame = np.zeros_like(old)

        if x > 0:
            frame[:, :x] = old[:, WIDTH - x :]
        if x < WIDTH:
            frame[:, x:] = new[:, : WIDTH - x]
        return frame

    if name == "Right":
        x = int(WIDTH * (1 - ease_in_out(t)))
        frame = np.zeros_like(old)

        if x > 0:
            frame[:, WIDTH - x :] = old[:, :x]
        if x < WIDTH:
            frame[:, : WIDTH - x] = new[:, x:]
        return frame

    if name == "Square":
        size = int(max(WIDTH, HEIGHT) * ease_in_out(t))
        frame = old.copy()
        cx = WIDTH // 2
        cy = HEIGHT // 2

        x1 = max(0, cx - size // 2)
        x2 = min(WIDTH, cx + size // 2)
        y1 = max(0, cy - size // 2)
        y2 = min(HEIGHT, cy + size // 2)

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = new[y1:y2, x1:x2]
        return frame

    if name == "Diamond":
        frame = old.copy()
        p = ease_in_out(t)
        cx = WIDTH // 2
        cy = HEIGHT // 2
        half_w = int(WIDTH * p)
        half_h = int(HEIGHT * p)

        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        pts = np.array(
            [
                [cx, cy - half_h],
                [cx + half_w, cy],
                [cx, cy + half_h],
                [cx - half_w, cy],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(mask, pts, 255)
        frame[mask > 0] = new[mask > 0]
        return frame

    if name == "Bounce":
        p = bounce_ease(t)
        # Новое фото прилетает сверху: его нижняя часть появляется у верхнего края
        visible = int(round(HEIGHT * p))
        visible = max(0, min(HEIGHT, visible))
        frame = old.copy()

        if visible > 0:
            frame[:visible] = new[HEIGHT - visible :]
        return frame

    return new


def choose_transitions(count):
    result = []

    for _ in range(count):
        available = [x for x in TRANSITIONS if x not in result[-2:]]
        if not available:
            available = TRANSITIONS
        result.append(random.choice(available))

    return result
