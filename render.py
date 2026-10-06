import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from options import (
    HEIGHT,
    WIDTH,
)
from transitions import choose_transitions, transform_image, transition_frame
from tts import END_FADE_DURATION, VOICE_OUTPUT

FPS = 30
ZOOM_AMOUNT = 0.08
MUSIC = "Музыка.mp3"
OUTPUT = "Порно.mp4"


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
