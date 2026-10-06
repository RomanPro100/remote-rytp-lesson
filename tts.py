import os
import wave
from pathlib import Path

import numpy as np
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"


END_FADE_DURATION = 0.45
VOICE_FILE = "voice.wav"
VOICE_OUTPUT = "Озвучка.wav"

MIN_IMAGE_DURATION = 3.0
TRANSITION_DURATION = 0.45
AFTER_SPEECH = 1.5
BEFORE_SPEECH = 0.3

_TTS: TTS | None = None


def get_tts():
    global _TTS

    if _TTS is None:
        print("Загружаю XTTS...")
        _TTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    return _TTS


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
