import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

import cv2
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# НАСТРОЙКИ
# ============================================================

WIDTH = 1920
HEIGHT = 1080
FPS = 30

if "--720" in sys.argv:
    WIDTH = 1280
    HEIGHT = 720

MIN_IMAGE_DURATION = 3.0
TRANSITION_DURATION = 0.45
END_FADE_DURATION = 0.45
AFTER_SPEECH = 1.5
BEFORE_SPEECH = 0.3

WATCH_MODE = "--watch" in sys.argv
POLL_INTERVAL = 2.0
STABLE_WAIT = 3.0

ZOOM_AMOUNT = 0.08
OCR_FAST = True
OCR_CACHE_FILE = "ocr_cache.json"

MAX_TEXT_LENGTH = 99
LONG_TEXT_REPLACEMENT = "Я не буду это читать"

OUTPUT = "Порно.mp4"
LESSON_PREFIX = "Дистанционный урок рутипи"
LESSON_COUNTER_FILE = "lesson_counter.txt"
MUSIC = "Музыка.mp3"

VOICE_FILE = "voice.wav"
OCR_TEXT_FILE = "Распознанный_текст.txt"
VOICE_OUTPUT = "Озвучка.wav"

TRANSITIONS = [
    "Spiral",
    "Square",
    "Diamond",
    "Left",
    "Right",
    "Bounce",
    "Fly3D",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


# ============================================================
# OCR
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["COQUI_TOS_AGREED"] = "1"

from paddleocr import PaddleOCR
from TTS.api import TTS

_OCR = None
_TTS = None


def get_ocr():
    global _OCR

    if _OCR is None:
        print("Загружаю PaddleOCR (один раз)...")

        if OCR_FAST:
            try:
                _OCR = PaddleOCR(
                    lang="ru",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
                    text_det_limit_type="max",
                    text_det_limit_side_len=960,
                    cpu_threads=os.cpu_count() or 4,
                )
                print("PaddleOCR: быстрые мобильные модели")
            except Exception as e:
                print(f"Быстрые модели не подошли ({e}), беру обычные.")
                _OCR = None

        if _OCR is None:
            _OCR = PaddleOCR(
                lang="ru",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    return _OCR


def get_tts():
    global _TTS

    if _TTS is None:
        print("Загружаю XTTS...")
        _TTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    return _TTS


BAD_PHRASES = [
    "activate windows",
    "go to pc settings",
    "предыдущее видео",
    "следующее видео",
    "назад",
    "поделиться",
    "комментарии",
    "тыс. просмотров",
    "месяцев",
    "дн.",
    "просмотров",
    "дней",
    "макофонь",
    "шарашич",
    "Новинка",
    "пуп то",
    "пуп тв",
    "пуп",
]

CORRECTIONS = {
    "Hacpal": "Насрал",
    "hасpal": "насрал",
    "ношел": "пошел",
    "PНTП": "РИТП",
    "nYП": "ПУП",
    "РУТР": "ритэпэ",
    "3": "З",
    "барбосины": "барбоскины",
    "нидарасы": "пидарасы",
    "нидорасы": "пидорасы",
    "*": "звиздачка",
    "cum": "Critical error: code one four eight eight. A critical virus has been detected on the computer. If the device is not turned off, the computer is critically done for. I am not going to read this. Three. Two. One. Ахахаахаххах чё повелись лошки ебаные",
}

LATIN_TO_CYR = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)


def has_cyrillic(text):
    return any("А" <= c <= "я" or c in "Ёё" for c in text)


def normalize_text(text):
    text = str(text).strip()

    for old, new in CORRECTIONS.items():
        text = text.replace(old, new)

    text_is_russian = has_cyrillic(text)
    result = []

    for word in text.split():
        if has_cyrillic(word) or (text_is_russian and len(word) == 1):
            word = word.translate(LATIN_TO_CYR)
        result.append(word)

    return " ".join(result)


def is_bad_text(text):
    low = text.lower().strip()

    if not low:
        return True

    for bad in BAD_PHRASES:
        if bad in low:
            return True

    if re.fullmatch(r"[\d\s:.,+%\-_/]+", low):
        return True

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", low)

    if len(letters) <= 1:
        return True

    # Полностью латинский текст НЕ отбрасываем: он тоже озвучивается.
    return False


def prepare_image_for_ocr(path):
    image = cv2.imread(str(path))
    if image is None:
        return None

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > 1280:
        scale = 1280 / longest
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return image


def extract_ocr_lines(result):
    lines = []

    if result is None:
        return lines

    try:
        for page in result:
            data = getattr(page, "json", None)
            if callable(data):
                data = data()
            if not data:
                continue
            if isinstance(data, list):
                data = data[0]

            res = data.get("res", data)
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("rec_boxes", [])
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys")
            if polys is None:
                polys = []

            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 1.0
                box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]

                try:
                    x1, y1, x2, y2 = map(float, box)
                except Exception:
                    x1 = y1 = x2 = y2 = 0

                angle = 0.0
                try:
                    (ax, ay), (bx, by) = polys[i][0], polys[i][1]
                    angle = math.degrees(math.atan2(by - ay, bx - ax))
                    while angle > 90:
                        angle -= 180
                    while angle <= -90:
                        angle += 180
                except Exception:
                    angle = 0.0

                lines.append(
                    {
                        "text": str(text),
                        "score": score,
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "h": abs(y2 - y1),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "angle": angle,
                    }
                )
    except Exception:
        pass

    return lines


# ============================================================
# ФИЛЬТР ЛИШНЕГО ТЕКСТА: вотермарки, интерфейс, мелочь
# ============================================================
# Ничего не вносится в чёрный список слов: фильтр смотрит только на
# положение, размер, наклон и фон вокруг текста, поэтому ему всё равно,
# на каком языке интерфейс и какая именно вотермарка.

FILTER_ENABLED = True
FILTER_DEBUG = True  # печатать, что убрано и почему

FILTER_MAX_TILT = 8.0  # наклонный текст (вывески, штампы) не читаем, градусы
FILTER_CORNER_X = 0.20  # угол: доля ширины от края
FILTER_CORNER_Y = 0.25  # угол: доля высоты от края
FILTER_CORNER_MAX_W = 0.40  # вотермарка в углу уже 40% ширины кадра
FILTER_SMALL_RATIO = 0.5  # строка ниже 50% самой крупной = мелкая
FILTER_SMALL_RATIO_TINY = 0.7  # ...или ниже 70%, если она ещё и крошечная в кадре
FILTER_TINY_HEIGHT = 0.045  # крошечная = ниже 4.5% высоты кадра
FILTER_TITLE_RATIO = 0.75  # название берём от 75% самой крупной строки
FILTER_TITLE_LIGHT = (
    165  # белый текст на тёмной теме: яркость от (серый второстепенный ~135)
)
FILTER_TITLE_DARK = 90  # чёрный текст на светлой теме: яркость до


def _fdbg(text):
    if FILTER_DEBUG:
        print(f"[ФИЛЬТР] {text}")


def _flat_theme_mask(gray, dark):
    """Ровный чёрный (тёмная тема) или белый (светлая тема) фон."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    sq = cv2.blur(g * g, (5, 5))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    base = (gray <= 45) if dark else (gray >= 240)
    return base & (std < 6)


def _runs(flags, min_len):
    runs = []
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None

    if start is not None and len(flags) - start >= min_len:
        runs.append((start, len(flags) - 1))

    return runs


def detect_ui_layout(gray):
    """
    Ищет скриншот интерфейса: сбоку (или снизу) большая полоса ровного
    чёрного/белого фона с текстом, а рядом превью-картинка.
    Чёрные полосы по обе стороны кадра (letterbox/pillarbox) — это не интерфейс.
    Возвращает {"rect": (x1, y1, x2, y2), "dark": bool} или None.
    """
    height, width = gray.shape

    for dark in (True, False):
        flat = _flat_theme_mask(gray, dark)
        if flat.mean() < 0.10:
            continue

        for axis in ("x", "y"):
            if axis == "x":
                flags = flat.mean(axis=0) > 0.7
                size = width
            else:
                flags = flat.mean(axis=1) > 0.7
                size = height

            runs = _runs(flags, max(3, int(size * 0.03)))
            if not runs:
                continue

            a, b = max(runs, key=lambda r: r[1] - r[0])
            if not (0.2 <= (b - a + 1) / size <= 0.85):
                continue

            touches_start = a <= size * 0.03
            touches_end = b >= size - 1 - size * 0.03
            if touches_start == touches_end:
                continue

            lo, hi = (0, a - 1) if touches_end else (b + 1, size - 1)
            if hi <= lo:
                continue

            if axis == "x":
                rows = flat[:, lo : hi + 1].mean(axis=1) < 0.9
                ys = np.where(rows)[0]
                if ys.size == 0:
                    continue
                rect = (lo, int(ys[0]), hi, int(ys[-1]))
            else:
                cols = flat[lo : hi + 1, :].mean(axis=0) < 0.9
                xs = np.where(cols)[0]
                if xs.size == 0:
                    continue
                rect = (int(xs[0]), lo, int(xs[-1]), hi)

            area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
            if area < 0.15 * width * height:
                continue

            return {"rect": rect, "dark": dark}

    return None


def _on_theme_background(gray, line, dark):
    """Строка лежит прямо на чёрном/белом фоне интерфейса (а не на картинке)?"""
    height, width = gray.shape
    x1 = max(0, int(round(line["x1"])))
    y1 = max(0, int(round(line["y1"])))
    x2 = min(width, int(round(line["x2"])))
    y2 = min(height, int(round(line["y2"])))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return False

    pad = max(3, int((y2 - y1) * 0.3))
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(width, x2 + pad), min(height, y2 + pad)

    ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
    ring_mask[y1 - oy1 : y2 - oy1, x1 - ox1 : x2 - ox1] = False
    ring = gray[oy1:oy2, ox1:ox2][ring_mask]

    if ring.size == 0:
        return False

    theme = (ring <= 45) if dark else (ring >= 240)
    return float(theme.mean()) >= 0.8


def _text_gray(gray, line):
    """Средняя яркость самих букв (то, что сильно отличается от фона вокруг)."""
    height, width = gray.shape
    x1 = max(0, int(round(line["x1"])))
    y1 = max(0, int(round(line["y1"])))
    x2 = min(width, int(round(line["x2"])))
    y2 = min(height, int(round(line["y2"])))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = gray[y1:y2, x1:x2]
    pad = max(3, int((y2 - y1) * 0.3))
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(width, x2 + pad), min(height, y2 + pad)

    ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
    ring_mask[y1 - oy1 : y2 - oy1, x1 - ox1 : x2 - ox1] = False
    ring = gray[oy1:oy2, ox1:ox2][ring_mask]

    bg = float(np.median(ring)) if ring.size else float(np.median(crop))
    mask = np.abs(crop.astype(np.int32) - int(bg)) > 70

    if int(mask.sum()) < 6:
        return None

    return float(crop[mask].mean())


def _apply_group_rules(lines, rect):
    """Угол, наклон и мелкий текст внутри области rect (кадр или превью)."""
    x0, y0, x1, y1 = rect
    rw = max(1, x1 - x0 + 1)
    rh = max(1, y1 - y0 + 1)
    kept = []

    for ln in lines:
        w = ln["x2"] - ln["x1"]
        h = max(1.0, ln["y2"] - ln["y1"])
        cx = (ln["x1"] + ln["x2"]) / 2
        cy = (ln["y1"] + ln["y2"]) / 2
        rx = (cx - x0) / rw
        ry = (cy - y0) / rh
        angle = ln.get("angle", 0.0)

        if w >= 1.5 * h and abs(angle) > FILTER_MAX_TILT:
            _fdbg(f"убрано «{ln['text']}»: наклонный текст ({angle:.0f}°)")
            continue

        in_x = rx < FILTER_CORNER_X or rx > 1 - FILTER_CORNER_X
        in_y = ry < FILTER_CORNER_Y or ry > 1 - FILTER_CORNER_Y
        if in_x and in_y and w < FILTER_CORNER_MAX_W * rw:
            _fdbg(f"убрано «{ln['text']}»: текст в углу (вотермарка)")
            continue

        kept.append(ln)

    if not kept:
        return kept

    max_h = max(ln["y2"] - ln["y1"] for ln in kept)
    result = []

    for ln in kept:
        h = ln["y2"] - ln["y1"]
        small = h < FILTER_SMALL_RATIO * max_h or (
            h < FILTER_SMALL_RATIO_TINY * max_h and h / rh < FILTER_TINY_HEIGHT
        )
        if small:
            _fdbg(f"убрано «{ln['text']}»: мелкий текст")
        else:
            result.append(ln)

    return result


def _find_height_step(gray, ln):
    """
    Если в одной строке слева (или справа) мелкие буквы вплотную к крупным
    (вотермарка, приклеенная к субтитру), находит границу между ними.
    Возвращает (левый_бокс, правый_бокс) или None.
    """
    x1, y1 = max(0, int(round(ln["x1"]))), max(0, int(round(ln["y1"])))
    x2, y2 = int(round(ln["x2"])), int(round(ln["y2"]))
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    thr, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Светлые буквы (белая заливка субтитров): берём только по-настоящему яркое,
    # чтобы чёрная обводка и фон не слипались с буквами.
    if crop.max() >= 230:
        mask = (crop > max(thr, 200)).astype(np.uint8)
    else:
        mask = (crop > thr).astype(np.uint8)

    if mask.mean() > 0.5:
        mask = (crop <= thr).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    line_h = y2 - y1
    comps = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 8 or h < 0.2 * line_h or h > 1.05 * line_h:
            continue
        comps.append((int(x), int(y), int(w), int(h)))

    if len(comps) < 5:
        return None

    comps.sort(key=lambda c: c[0] + c[2] / 2)
    heights = [c[3] for c in comps]
    best = None

    # Ищем такое место, где слева все буквы заметно мельче, чем справа (или наоборот),
    # и группы не перемешаны по высоте.
    for k in range(2, len(comps) - 1):
        left, right = heights[:k], heights[k:]
        ml, mr = float(np.mean(left)), float(np.mean(right))
        ratio = max(ml, mr) / max(1.0, min(ml, mr))
        if ratio < 1.6:
            continue

        small, big = (left, right) if ml < mr else (right, left)
        if max(small) >= 0.9 * min(big):
            continue

        if best is None or ratio > best[0]:
            best = (ratio, k)

    if best is None:
        return None

    k = best[1]
    boundary = x1 + (comps[k - 1][0] + comps[k - 1][2] + comps[k][0]) // 2
    parts = []

    for group in (comps[:k], comps[k:]):
        gx1 = x1 + min(c[0] for c in group)
        gx2 = x1 + max(c[0] + c[2] for c in group)
        gy1 = y1 + min(c[1] for c in group)
        gy2 = y1 + max(c[1] + c[3] for c in group)
        parts.append((gx1, gy1, gx2, gy2))

    parts[0] = (parts[0][0], parts[0][1], min(parts[0][2], boundary), parts[0][3])
    parts[1] = (max(parts[1][0], boundary), parts[1][1], parts[1][2], parts[1][3])
    return parts


def _read_box(ocr, image, box):
    """Заново читает только один кусок строки."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    crop = image[
        max(0, y1 - 4) : min(height, y2 + 4), max(0, x1 - 2) : min(width, x2 + 2)
    ]

    if crop.size == 0:
        return "", 0.0

    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    found = extract_ocr_lines(ocr.predict(crop))

    if not found:
        return "", 0.0

    found.sort(key=lambda s: s["x"])
    text = normalize_text(" ".join(s["text"] for s in found))
    return text, min(s["score"] for s in found)


def _split_edge_lines(ocr, image, gray, lines):
    """Разделяет «вотермарка + субтитр», которые детектор склеил в одну строку."""
    height, width = gray.shape
    result = []

    for ln in lines:
        w = ln["x2"] - ln["x1"]
        h = ln["y2"] - ln["y1"]
        cy = (ln["y1"] + ln["y2"]) / 2
        touches_edge = ln["x1"] <= 0.06 * width or ln["x2"] >= 0.94 * width
        edge_row = cy < 0.3 * height or cy > 0.7 * height

        if not (touches_edge and edge_row and w >= 4 * h):
            result.append(ln)
            continue

        try:
            parts = _find_height_step(gray, ln)
            if not parts:
                result.append(ln)
                continue

            new_lines = []
            for box in parts:
                text, score = _read_box(ocr, image, box)
                if not text or is_bad_text(text):
                    new_lines = None
                    break
                new_lines.append(
                    {
                        "text": text,
                        "score": score,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "x": (box[0] + box[2]) / 2,
                        "y": (box[1] + box[3]) / 2,
                        "h": box[3] - box[1],
                        "angle": ln.get("angle", 0.0),
                    }
                )

            if new_lines:
                _fdbg(
                    f"строка «{ln['text']}» разделена: "
                    + " | ".join(f"«{x['text']}»" for x in new_lines)
                )
                result.extend(new_lines)
            else:
                result.append(ln)
        except Exception as e:
            _fdbg(f"не удалось разделить «{ln['text']}»: {e}")
            result.append(ln)

    return result


def _filter_lines(ocr, image, lines):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cand = []
    for ln in lines:
        text = normalize_text(ln["text"])
        if is_bad_text(text):
            continue
        ln = dict(ln)
        ln["text"] = text
        cand.append(ln)

    if not cand:
        return []

    cand = _split_edge_lines(ocr, image, gray, cand)
    ui = detect_ui_layout(gray)

    # Обычный кадр: субтитры, вотермарки в углах.
    if ui is None:
        return _apply_group_rules(cand, (0, 0, width - 1, height - 1))

    # Скриншот интерфейса: превью-картинка + оформление вокруг.
    rect = ui["rect"]
    dark = ui["dark"]
    _fdbg(f"найден интерфейс ({'тёмная' if dark else 'светлая'} тема), превью: {rect}")

    preview, page = [], []
    for ln in cand:
        cx = (ln["x1"] + ln["x2"]) / 2
        cy = (ln["y1"] + ln["y2"]) / 2
        inside = rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]
        if inside and not _on_theme_background(gray, ln, dark):
            preview.append(ln)
        else:
            page.append(ln)

    kept = _apply_group_rules(preview, rect)
    if kept:
        for ln in page:
            _fdbg(f"убрано «{ln['text']}»: интерфейс (на превью есть свой текст)")
        return kept

    # На превью нет важного текста: читаем только название (крупное и контрастное).
    title = []
    for ln in page:
        tone = _text_gray(gray, ln)
        contrast = tone is not None and (
            tone >= FILTER_TITLE_LIGHT if dark else tone <= FILTER_TITLE_DARK
        )
        if contrast:
            title.append(ln)
        else:
            _fdbg(f"убрано «{ln['text']}»: интерфейс (не жирный контрастный)")

    if not title:
        return []

    max_h = max(ln["y2"] - ln["y1"] for ln in title)
    result = []
    for ln in title:
        if ln["y2"] - ln["y1"] >= FILTER_TITLE_RATIO * max_h:
            result.append(ln)
        else:
            _fdbg(f"убрано «{ln['text']}»: интерфейс (мельче названия)")

    return result


def filter_lines(ocr, image, lines):
    if not FILTER_ENABLED or not lines:
        return lines

    try:
        return _filter_lines(ocr, image, lines)
    except Exception as e:
        print(f"[ФИЛЬТР] Ошибка, фильтр пропущен: {e}")
        return lines


def make_groups(lines):
    if not lines:
        return []

    lines = sorted(lines, key=lambda x: (x["y"], x["x"]))
    rows = []

    for item in lines:
        text = normalize_text(item["text"])
        if is_bad_text(text):
            continue

        item = dict(item)
        item["text"] = text
        target = None

        for row in rows:
            avg_y = sum(x["y"] for x in row) / len(row)
            avg_h = max(1, sum(x["h"] for x in row) / len(row))

            if abs(item["y"] - avg_y) < avg_h * 0.65:
                target = row
                break

        if target is None:
            rows.append([item])
        else:
            target.append(item)

    rows.sort(key=lambda row: min(x["y"] for x in row))
    result = []

    for row in rows:
        row.sort(key=lambda x: x["x"])
        text = normalize_text(" ".join(x["text"] for x in row))
        if not is_bad_text(text):
            result.append(text)

    return result


def ocr_one_image(ocr, path):
    image = prepare_image_for_ocr(path)

    if image is None:
        print(f"[OCR] Не удалось открыть: {path.name}")
        return ""

    try:
        result = ocr.predict(image)
        lines = filter_lines(ocr, image, extract_ocr_lines(result))
        groups = make_groups(lines)
        text = normalize_text(" ".join(groups))

        if not text:
            print(f"[OCR] {path.name}: подходящего текста нет")
            return ""

        print(f"[OCR] {path.name}: {text}")
        return text
    except Exception as e:
        print(f"[OCR] Ошибка {path.name}: {e}")
        return ""


def run_ocr(photos):
    print("\n================ OCR ================\n")
    ocr = get_ocr()

    try:
        with open(OCR_CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    texts = []

    for done, photo in enumerate(photos, 1):
        try:
            st = photo.stat()
            key = (
                f"v3|fast={OCR_FAST}|filter={FILTER_ENABLED}|{photo.name}"
                f"|{st.st_size}|{st.st_mtime_ns}"
            )
        except OSError:
            key = None

        if key is not None and key in cache:
            print(f"[OCR {done}/{len(photos)}] из кэша: {photo.name}")
            texts.append(cache[key])
            continue

        print(f"[OCR {done}/{len(photos)}]")
        text = ocr_one_image(ocr, photo)
        texts.append(text)

        if key is not None:
            cache[key] = text
            try:
                with open(OCR_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            except Exception:
                pass

    with open(OCR_TEXT_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text)
            f.write("\n")

    print(f"\nТекст сохранён в: {OCR_TEXT_FILE}")
    return texts


# ============================================================
# XTTS
# ============================================================


def trim_silence(wav, rate, threshold=0.01, keep=0.04):
    if wav.size == 0:
        return wav

    idx = np.where(np.abs(wav) > threshold)[0]
    if idx.size == 0:
        return wav[:0]

    pad = int(rate * keep)
    return wav[max(0, idx[0] - pad) : min(len(wav), idx[-1] + pad + 1)]


def load_wav_mono(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())

    audio = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32), rate


def check_points_silent(points, path):
    audio, rate = load_wav_mono(path)
    loud = 0

    for i, (a, b) in enumerate(points, 1):
        seg = audio[int(a * rate) : int(b * rate)]
        if seg.size == 0:
            continue

        rms = np.sqrt(np.mean(seg * seg))
        db = 20 * np.log10(max(rms, 1) / 32768)

        if db > -45:
            loud += 1
            print(f"[!] Переход {i} ({a:.2f}-{b:.2f}) попал не в тишину: {db:.1f} дБ")

    if loud == 0:
        print("Проверка: все смены фото попадают в тишину Озвучка.wav")


def make_voice(texts):
    print("\n================ XTTS ================\n")

    voice = Path(VOICE_FILE)
    if not voice.exists():
        raise FileNotFoundError(f"Не найден файл голоса: {VOICE_FILE}")

    if not any(x.strip() for x in texts):
        print("OCR не нашёл ни одного текста.")
        return None

    tts = get_tts()
    rate = int(tts.synthesizer.output_sample_rate)

    pause = AFTER_SPEECH + TRANSITION_DURATION + BEFORE_SPEECH
    outro = max(AFTER_SPEECH, END_FADE_DURATION + 0.1)

    def silence(sec):
        return np.zeros(int(round(sec * rate)), dtype=np.float32)

    chunks = [silence(BEFORE_SPEECH)]
    position = len(chunks[0])
    speech_end = []
    pause_samples = len(silence(pause))

    for i, text in enumerate(texts):
        text = text.strip()
        wav = None

        if text:
            print(f"[XTTS] {i + 1}/{len(texts)}: {text}")
            try:
                raw = tts.tts(
                    text=text,
                    speaker_wav=str(voice),
                    language="ru",
                    split_sentences=True,
                )
                wav = trim_silence(np.asarray(raw, dtype=np.float32), rate)
            except Exception as e:
                print(f"[XTTS] Ошибка на фото {i + 1}: {e}")

        if wav is None or wav.size == 0:
            wav = silence(max(0.0, MIN_IMAGE_DURATION - BEFORE_SPEECH - AFTER_SPEECH))

        chunks.append(wav)
        position += len(wav)
        speech_end.append(position)

        if i < len(texts) - 1:
            chunks.append(silence(pause))
            position += pause_samples

    chunks.append(silence(outro))
    audio = np.clip(np.concatenate(chunks), -1.0, 1.0)

    with wave.open(VOICE_OUTPUT, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())

    total_duration = len(audio) / rate
    points = []

    for end_sample in speech_end[:-1]:
        start = end_sample / rate + AFTER_SPEECH
        points.append((start, start + TRANSITION_DURATION))

    print("\nГраницы смены фотографий:")
    for i, (a, b) in enumerate(points, 1):
        print(f"{i}: {a:.2f}–{b:.2f} сек")

    print(f"\nОзвучка сохранена: {VOICE_OUTPUT} ({total_duration:.1f} сек)")
    check_points_silent(points, VOICE_OUTPUT)

    return points, total_duration


# ============================================================
# ИЗОБРАЖЕНИЯ
# ============================================================


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось открыть изображение: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    scale = min(WIDTH / w, HEIGHT / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    x = (WIDTH - nw) // 2
    y = (HEIGHT - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized

    return canvas


# ============================================================
# ПЕРЕХОДЫ
# ============================================================


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


def transition_fly3d(old, new, t):
    # Чуть более плавный разгон/торможение.
    t = ease_in_out(t)
    t = t**1.15

    values = interpolate_smooth_keyframes(
        FLY3D_KEYFRAMES,
        t,
    )
    dst = values.reshape(4, 2).copy()
    dst[:, 0] *= WIDTH
    dst[:, 1] *= HEIGHT

    # Если карточка почти строго ребром, warpPerspective может получить
    # плохо обусловленную матрицу. На несколько пикселей это незаметно,
    # поэтому оставляем старый кадр до нормального размера плоскости.
    area = abs(cv2.contourArea(dst.astype(np.float32)))
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

    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

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


# ============================================================
# ВИДЕО
# ============================================================


def make_video(photos, points, audio_duration, output=OUTPUT):
    print("\n================ ВИДЕО ================\n")

    if not photos:
        raise RuntimeError("Нет фотографий для видео.")

    if not Path(VOICE_OUTPUT).exists():
        raise FileNotFoundError(f"Не найден файл {VOICE_OUTPUT}")

    images = []
    for photo in photos:
        print(f"Загрузка: {photo.name}")
        images.append(load_image(photo))

    transitions = choose_transitions(max(0, len(images) - 1))
    print("\nПереходы:")
    for i, tr in enumerate(transitions, 1):
        print(f"{i}: {tr}")

    zoom_starts = [0.0] + [a for a, _ in points]
    zoom_ends = [b for _, b in points] + [audio_duration]

    def zoomed(index, time_sec):
        if ZOOM_AMOUNT <= 0:
            return images[index]

        span = max(0.001, zoom_ends[index] - zoom_starts[index])
        p = (time_sec - zoom_starts[index]) / span
        p = max(0.0, min(1.0, p))

        if p <= 0:
            return images[index]

        matrix = cv2.getRotationMatrix2D(
            (WIDTH / 2, HEIGHT / 2),
            0,
            1.0 + ZOOM_AMOUNT * p,
        )
        return transform_image(images[index], matrix)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg не найден в PATH.")

    has_music = Path(MUSIC).exists()

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-i",
        VOICE_OUTPUT,
    ]

    if has_music:
        command += ["-stream_loop", "-1", "-i", MUSIC]
        filter_complex = (
            "[2:a]volume=0.25[music];"
            "[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[audio]"
        )
    else:
        filter_complex = "[1:a]anull[audio]"

    command += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[audio]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{audio_duration:.4f}",
        output,
    ]

    print(f"\nДлительность озвучки: {audio_duration:.2f} сек")
    print("Создаю видео по временной шкале аудио...\n")

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_count = int(np.ceil(audio_duration * FPS))
    point_index = 0

    try:
        for frame_number in range(frame_count):
            time_sec = frame_number / FPS

            while point_index < len(points) and time_sec >= points[point_index][1]:
                point_index += 1

            if point_index < len(points):
                start, end = points[point_index]

                if start <= time_sec < end:
                    old_image = zoomed(point_index, time_sec)
                    new_image = zoomed(point_index + 1, time_sec)
                    progress = (time_sec - start) / max(0.001, end - start)

                    frame = transition_frame(
                        old_image,
                        new_image,
                        transitions[point_index],
                        progress,
                    )
                else:
                    frame = zoomed(point_index, time_sec)
            else:
                frame = zoomed(len(images) - 1, time_sec)

            fade_start = max(0.0, audio_duration - END_FADE_DURATION)
            if time_sec >= fade_start:
                fade = 1.0 - ((time_sec - fade_start) / max(0.001, END_FADE_DURATION))
                frame = (frame.astype(np.float32) * max(0.0, min(1.0, fade))).astype(
                    np.uint8
                )

            process.stdin.write(frame.tobytes())

            if frame_number % (FPS * 5) == 0:
                print(f"Рендер: {time_sec:.1f} / {audio_duration:.1f} сек")
    finally:
        process.stdin.close()

    code = process.wait()
    if code != 0:
        raise RuntimeError(f"FFmpeg завершился с кодом {code}")

    print(f"\nГотово: {output}")


# ============================================================
# ОБРАБОТКА
# ============================================================


def next_lesson_name(folder):
    """Берёт следующий номер только по реально существующим видео."""
    biggest = 0
    scan_dirs = [Path(folder)]

    for extra in os.environ.get("LESSON_SCAN_DIRS", "").split(";"):
        extra = extra.strip()
        if extra and os.path.isdir(extra):
            scan_dirs.append(Path(extra))

    for scan_dir in scan_dirs:
        for f in scan_dir.glob(f"{LESSON_PREFIX} *.mp4"):
            m = re.fullmatch(rf"{re.escape(LESSON_PREFIX)} (\d+)", f.stem)
            if m:
                biggest = max(biggest, int(m.group(1)))

    number = biggest + 1

    try:
        with open(Path(folder) / LESSON_COUNTER_FILE, "w", encoding="utf-8") as f:
            f.write(str(number))
    except Exception:
        pass

    return f"{LESSON_PREFIX} {number}.mp4"


def scan_photos(folder):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.name not in {OUTPUT, VOICE_OUTPUT}
    )


def snapshot(photos):
    result = []
    for p in photos:
        try:
            st = p.stat()
            result.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
    return tuple(result)


def process_batch(photos, output):
    photos = list(photos)
    random.shuffle(photos)

    print("\nПорядок фотографий:")
    for i, photo in enumerate(photos, 1):
        print(f"{i:03d}. {photo.name}")

    texts = run_ocr(photos)
    voice_texts = []
    replaced = 0

    for text in texts:
        if len(text.strip()) > MAX_TEXT_LENGTH:
            voice_texts.append(LONG_TEXT_REPLACEMENT)
            replaced += 1
        else:
            voice_texts.append(text)

    if replaced:
        print(
            f"\nДлинный текст (>{MAX_TEXT_LENGTH} символов) "
            f"заменён на заглушку у {replaced} фото"
        )

    result = make_voice(voice_texts)
    if result is None:
        raise RuntimeError("Не удалось создать озвучку.")

    points, duration = result
    make_video(photos, points, duration, output)

    print("\n======================================")
    print("ВСЁ ГОТОВО")
    print("======================================")
    print(f"Текст:     {OCR_TEXT_FILE}")
    print(f"Озвучка:   {VOICE_OUTPUT}")
    print(f"Видео:     {output}")
    print("======================================")


def disable_quickedit():
    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()

        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return

        ENABLE_QUICK_EDIT = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT) | ENABLE_EXTENDED_FLAGS
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


def main():
    disable_quickedit()
    folder = Path(__file__).resolve().parent
    os.chdir(folder)

    print("======================================")
    print("        OCR + XTTS + СЛАЙДШОУ")
    print("======================================\n")

    get_ocr()
    get_tts()

    last = None
    waiting_shown = False

    while True:
        photos = scan_photos(folder)
        snap = snapshot(photos)

        if photos and snap != last:
            waiting_shown = False

            time.sleep(STABLE_WAIT if WATCH_MODE else 0)
            if snapshot(scan_photos(folder)) != snap:
                continue

            output = next_lesson_name(folder)

            try:
                process_batch(photos, output)
            except Exception:
                if not WATCH_MODE:
                    raise
                traceback.print_exc()

            last = snap

            if WATCH_MODE:
                print("\nЖду новые фотографии... (Ctrl+C — выход)")

        elif not photos:
            if not WATCH_MODE:
                raise RuntimeError("В папке не найдено ни одной фотографии.")

            if not waiting_shown:
                print("В папке нет фотографий, жду...")
                waiting_shown = True

        if not WATCH_MODE:
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено.")
