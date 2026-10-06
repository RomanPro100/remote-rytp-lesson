import os
import sys
import time
import traceback
from pathlib import Path

from ocr import get_ocr
from processing import (
    disable_quickedit,
    next_lesson_name,
    process_batch,
    scan_photos,
    snapshot,
)
from tts import get_tts

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


WATCH_MODE = "--watch" in sys.argv
POLL_INTERVAL = 2.0
STABLE_WAIT = 3.0


def main():
    disable_quickedit()
    # TODO: добавить параметры: папка с сурсами и папка с результатами
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
