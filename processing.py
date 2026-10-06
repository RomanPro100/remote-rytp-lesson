import os
import random
import re
from pathlib import Path

from ocr import OCR_TEXT_FILE, run_ocr
from render import OUTPUT, VOICE_OUTPUT, make_video
from tts import make_voice

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


LESSON_PREFIX = "Дистанционный урок рутипи"
LESSON_COUNTER_FILE = "lesson_counter.txt"

MAX_TEXT_LENGTH = 99
LONG_TEXT_REPLACEMENT = "Я не буду это читать"


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
