#!/usr/bin/env python3
"""Offline transcription of saved recordings using mlx-whisper."""

import argparse
import wave
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_LANGUAGE = "en"
RECORDINGS_DIR = Path("recordings")

VAD_WINDOW_MS = 30        # RMS window size for VAD
VAD_PAD_MS = 300          # padding around detected speech regions
VAD_THRESHOLD = 0.005     # RMS threshold below which audio is considered silence


def latest_recording():
    recordings = sorted(
        [p for p in RECORDINGS_DIR.glob("*.wav") if ".eq" not in p.name],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return recordings[0] if recordings else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an existing WAV file with mlx-whisper."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("audio", nargs="?", type=Path, help="Audio file to transcribe")
    input_group.add_argument("--latest", action="store_true", help="Use newest WAV in recordings/")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX Whisper model repo/path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code, or 'auto'. Default: en",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output text file. Default: <audio-stem>.mlx.txt",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include segment timestamps in the output file.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Zero out silent regions before transcription to suppress hallucinations.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=VAD_THRESHOLD,
        help=f"RMS silence threshold for VAD. Default: {VAD_THRESHOLD}",
    )
    return parser.parse_args()


def apply_vad(audio, rate, threshold, window_ms=VAD_WINDOW_MS, pad_ms=VAD_PAD_MS):
    window = int(rate * window_ms / 1000)
    pad_windows = int(rate * pad_ms / 1000) // window

    n = len(audio) // window
    is_speech = np.array([
        float(np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2))) > threshold
        for i in range(n)
    ])

    # expand each speech window left and right by pad_windows
    padded = np.zeros(n, dtype=bool)
    for i in np.where(is_speech)[0]:
        padded[max(0, i - pad_windows):min(n, i + pad_windows + 1)] = True

    result = audio.copy()
    for i in range(n):
        if not padded[i]:
            result[i * window:(i + 1) * window] = 0.0

    speech_ratio = padded.sum() / n if n > 0 else 0.0
    return result, speech_ratio


def load_wav_as_float32(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        if width != 2:
            raise SystemExit(f"Only 16-bit PCM WAV supported, got sample width {width}")
        audio = np.frombuffer(wav.readframes(frames), dtype=np.int16)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = audio.astype(np.float32) / 32768.0
    if rate != 16000:
        raise SystemExit(
            f"{path} is {rate} Hz. Convert to 16 kHz mono WAV first."
        )
    return audio


def format_segments(result):
    lines = []
    for seg in result.get("segments", []):
        start, end = seg.get("start", 0.0), seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{start:8.2f} -> {end:8.2f}] {text}")
    return "\n".join(lines).strip()


def main():
    args = parse_args()

    audio_path = latest_recording() if args.latest else args.audio
    if audio_path is None:
        raise SystemExit("No WAV files found in recordings/")
    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")

    output = args.output or audio_path.with_suffix(".mlx.txt")
    language = None if args.language == "auto" else args.language

    print(f"Audio:    {audio_path}")
    print(f"Model:    {args.model}")
    print(f"Language: {args.language}")
    print(f"Output:   {output}")

    audio = load_wav_as_float32(audio_path)

    if args.vad:
        print(f"VAD:      enabled (threshold={args.vad_threshold})")
        audio, speech_ratio = apply_vad(audio, 16000, threshold=args.vad_threshold)
        print(f"          {speech_ratio * 100:.1f}% of audio classified as speech")
    else:
        print("VAD:      disabled (use --vad to suppress silence hallucinations)")

    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=args.model,
        language=language,
        verbose=False,
    )

    text = result["text"].strip()
    if args.word_timestamps:
        text = format_segments(result) or text

    output.write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
