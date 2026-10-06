import json
import math
import os
import re

import cv2
import numpy as np
from paddleocr import PaddleOCR

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

OCR_FAST = True
OCR_CACHE_FILE = "ocr_cache.json"
OCR_TEXT_FILE = "Распознанный_текст.txt"

_OCR = None


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
