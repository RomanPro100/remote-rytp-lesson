import os
import random
import re
import shutil
import subprocess
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ============================================================
# НАСТРОЙКИ
# ============================================================

WIDTH = 1920
HEIGHT = 1080
FPS = 60

# Сколько фото без текста висит на экране (всего, сек)
MIN_IMAGE_DURATION = 3.0

# Переходы
TRANSITION_DURATION = 0.45
END_FADE_DURATION = 0.45

# Сколько фото остаётся на экране ПОСЛЕ окончания озвучки, сек
AFTER_SPEECH = 1.5
# Сколько новое фото висит до начала его озвучки, сек
BEFORE_SPEECH = 0.3

# Режим ожидания: модели загружаются один раз, скрипт остаётся запущенным
# и сам делает новое видео, когда в папке меняется набор фотографий.
# Поставь False, если нужен обычный одноразовый запуск.
WATCH_MODE = True
POLL_INTERVAL = 2.0  # как часто смотреть папку, сек
STABLE_WAIT = 3.0  # ждать, пока фото докопируются, сек

# Медленное увеличение каждого фото: на сколько оно вырастет за время показа
# (0.08 = на 8%). Поставь 0, чтобы отключить.
ZOOM_AMOUNT = 0.08

OUTPUT = "Слайдшоу.mp4"
MUSIC = "Музыка.mp3"

VOICE_FILE = "voice.wav"
OCR_TEXT_FILE = "Распознанный_текст.txt"
VOICE_OUTPUT = "Озвучка.wav"

TRANSITIONS = ["Spiral", "Square", "Diamond", "Left", "Right", "Bounce"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================
# OCR
# ============================================================

# Отключаем проблемные CPU/PIR-механизмы Paddle
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Автоматически принимаем лицензию Coqui XTTS, чтобы скрипт не ждал ввода
os.environ["COQUI_TOS_AGREED"] = "1"

from paddleocr import PaddleOCR
from TTS.api import TTS

# ============================================================
# МОДЕЛИ (загружаются один раз за запуск)
# ============================================================

_OCR = None
_TTS = None


def get_ocr():
    global _OCR

    if _OCR is None:
        print("Загружаю PaddleOCR (один раз)...")
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
        print("Загружаю XTTS (один раз)...")
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
    "макофонь",
    "шарашич",
]

CORRECTIONS = {
    "Hacpal": "Насрал",
    "hасpal": "насрал",
    "ношел": "пошел",
    "PНTП": "РИТП",
    "nYП": "ПУП",
}


def normalize_text(text):
    text = str(text).strip()

    for old, new in CORRECTIONS.items():
        text = text.replace(old, new)

    # Частые визуальные подмены латиницей внутри кириллических слов
    result = []

    for word in text.split():
        if any("А" <= c.upper() <= "Я" for c in word):
            mapping = str.maketrans(
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
                }
            )
            word = word.translate(mapping)

        result.append(word)

    return " ".join(result)


def is_bad_text(text):
    low = text.lower().strip()

    if not low:
        return True

    for bad in BAD_PHRASES:
        if bad in low:
            return True

    # Только числа / время / символы
    if re.fullmatch(r"[\d\s:.,+%\-_/]+", low):
        return True

    # Слишком короткий мусор
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", low)

    if len(letters) <= 1:
        return True

    # Полностью английский текст обычно является интерфейсным мусором.
    # Смешанный текст оставляем, потому что OCR любит путать
    # кириллицу и латиницу.
    if letters and all("a" <= c.lower() <= "z" for c in letters):
        return True

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
            image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    return image


def extract_ocr_lines(result):
    """
    Достаёт текстовые блоки PaddleOCR.
    Поддерживает актуальный формат PaddleOCR.
    """

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

            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 1.0

                box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]

                try:
                    x1, y1, x2, y2 = map(float, box)
                except Exception:
                    x1 = y1 = x2 = y2 = 0

                lines.append(
                    {
                        "text": str(text),
                        "score": score,
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "h": abs(y2 - y1),
                    }
                )

    except Exception:
        pass

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
        lines = extract_ocr_lines(result)
        groups = make_groups(lines)

        text = " ".join(groups)
        text = normalize_text(text)

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

    texts = []

    for photo in photos:
        texts.append(ocr_one_image(ocr, photo))

    # Сохраняем текст по одному блоку на каждое фото.
    # Пустые фото сохраняются как пустые строки, чтобы индексы
    # всегда совпадали с фотографиями.
    with open(OCR_TEXT_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text)
            f.write("\n")

    print(f"\nТекст сохранён в: {OCR_TEXT_FILE}")

    return texts


# ============================================================
# XTTS: озвучка по каждому фото + точная временная шкала
# ============================================================


def trim_silence(wav, rate, threshold=0.01, keep=0.04):
    """Обрезает тишину по краям фрагмента озвучки (порог ~ -40 дБ)."""

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
        frames = wav.getnframes()
        data = wav.readframes(frames)

    audio = np.frombuffer(data, dtype=np.int16)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32), rate


def check_points_silent(points, path):
    """Проверяет по готовому WAV, что смена фото попала в тишину."""

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
    """
    Озвучивает каждое фото ОТДЕЛЬНО и склеивает в Озвучка.wav,
    вставляя между фрагментами чистую тишину. Благодаря этому
    границы смены фото известны точно: они стоят ровно
    посередине тишины после озвучки каждого фото.

    Возвращает (points, total_duration) или None, если текста нет.
    """

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

    speech_end = []  # конец озвучки каждого фото, в сэмплах
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
            # Нет текста (или озвучка не вышла) — короткий показ фото
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

    # Переход начинается через AFTER_SPEECH секунд тишины после озвучки
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


def ease_out(t):
    return 1 - (1 - t) ** 3


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


def transition_frame(old, new, name, t):
    t = max(0.0, min(1.0, t))

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
            new_resized = cv2.resize(new, (x2 - x1, y2 - y1))

            frame[y1:y2, x1:x2] = new_resized

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

    if name == "Spiral":
        p = ease_in_out(t)

        angle = p * np.pi * 2
        scale = 1.0 + 0.7 * p

        dx = np.cos(angle) * WIDTH * 0.10 * (1 - p)
        dy = np.sin(angle) * HEIGHT * 0.10 * (1 - p)

        matrix = cv2.getRotationMatrix2D(
            (WIDTH / 2, HEIGHT / 2), angle * 180 / np.pi, scale
        )

        matrix[0, 2] += dx
        matrix[1, 2] += dy

        moving = transform_image(new, matrix)

        alpha = p
        return cv2.addWeighted(old, 1 - alpha, moving, alpha, 0)

    if name == "Bounce":
        p = bounce_ease(t)

        scale = 0.7 + 0.3 * p

        matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 0, scale)

        moving = transform_image(new, matrix)

        alpha = ease_out(t)

        return cv2.addWeighted(old, 1 - alpha, moving, alpha, 0)

    return new


def choose_transitions(count):
    result = []

    for _ in range(count):
        available = [x for x in TRANSITIONS if x not in result[-2:]]

        if not available:
            available = TRANSITIONS

        result.append(random.choice(available))

    return result


def make_video(photos, points, audio_duration, output=OUTPUT):
    print("\n================ ВИДЕО ================\n")

    if not photos:
        raise RuntimeError("Нет фотографий для видео.")

    audio_path = Path(VOICE_OUTPUT)
    if not audio_path.exists():
        raise FileNotFoundError(f"Не найден файл {VOICE_OUTPUT}")

    images = []

    for photo in photos:
        print(f"Загрузка: {photo.name}")
        images.append(load_image(photo))

    transitions = choose_transitions(max(0, len(images) - 1))

    # Время жизни каждого фото: с начала входящего перехода
    # до конца исходящего. За это время оно плавно увеличивается.
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
            (WIDTH / 2, HEIGHT / 2), 0, 1.0 + ZOOM_AMOUNT * p
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
            "[1:a][music]amix=inputs=2:"
            "duration=first:dropout_transition=2[audio]"
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

            # Выбираем, какая фотография сейчас активна.
            # Во время перехода показываем смесь двух кадров.
            while point_index < len(points) and time_sec >= points[point_index][1]:
                point_index += 1

            if point_index < len(points):
                start, end = points[point_index]

                if start <= time_sec < end:
                    old_image = zoomed(point_index, time_sec)
                    new_image = zoomed(point_index + 1, time_sec)

                    progress = (time_sec - start) / max(0.001, end - start)

                    frame = transition_frame(
                        old_image, new_image, transitions[point_index], progress
                    )
                else:
                    # До начала перехода остаётся старое фото
                    frame = zoomed(point_index, time_sec)
            else:
                frame = zoomed(len(images) - 1, time_sec)

            # Плавное затемнение в самом конце видео
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
# MAIN
# ============================================================


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

    # ВАЖНО:
    # Перемешиваем ДО OCR.
    # Поэтому порядок фотографий,
    # порядок OCR и порядок озвучки совпадают.
    random.shuffle(photos)

    print("\nПорядок фотографий:")

    for i, photo in enumerate(photos, 1):
        print(f"{i:03d}. {photo.name}")

    texts = run_ocr(photos)

    result = make_voice(texts)

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


def main():
    folder = Path(__file__).resolve().parent

    os.chdir(folder)

    print("======================================")
    print("        OCR + XTTS + СЛАЙДШОУ")
    print("======================================\n")

    # Модели грузим ОДИН раз в самом начале
    get_ocr()
    get_tts()

    last = None
    first = True
    waiting_shown = False

    while True:
        photos = scan_photos(folder)
        snap = snapshot(photos)

        if photos and snap != last:
            waiting_shown = False

            # Ждём, пока файлы перестанут копироваться
            time.sleep(STABLE_WAIT if WATCH_MODE else 0)

            if snapshot(scan_photos(folder)) != snap:
                continue

            if first:
                output = OUTPUT
            else:
                output = f"Слайдшоу_{datetime.now():%Y%m%d_%H%M%S}.mp4"

            try:
                process_batch(photos, output)
            except Exception:
                if not WATCH_MODE:
                    raise
                traceback.print_exc()

            last = snap
            first = False

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
