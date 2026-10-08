import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


def srt_time(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe video locally with faster-whisper")
    parser.add_argument("video", type=Path, help="Path to the input video")
    args = parser.parse_args()

    output = Path("output")
    output.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        audio = Path(temp_dir) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(args.video), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
            check=True,
        )
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True)
        transcript = {"source": str(args.video), "language": info.language, "segments": []}
        srt_blocks = []
        for index, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            transcript["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "words": [
                    {"start": word.start, "end": word.end, "text": word.word}
                    for word in (segment.words or [])
                ],
            })
            srt_blocks.append(
                f"{index}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{text}"
            )

    (output / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "captions.srt").write_text("\n\n".join(srt_blocks) + "\n", encoding="utf-8")
    print("Created output/transcript.json and output/captions.srt")


if __name__ == "__main__":
    main()
