"""
Transcribe all MP3 files in e:/Fitting/ to text using faster-whisper.
Output: e:/Fitting/transcripts/{title}.txt
"""

import os
os.environ["HF_HOME"] = "e:/hf_cache"  # Download models to E drive

from pathlib import Path
from faster_whisper import WhisperModel

AUDIO_DIR = Path("e:/Fitting")
TRANSCRIPT_DIR = AUDIO_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)

MODEL_SIZE = "small"  # small / medium / large-v3

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

mp3_files = sorted(AUDIO_DIR.glob("*.mp3"))
total = len(mp3_files)
print(f"Found {total} MP3 files.\n")

for i, mp3_path in enumerate(mp3_files, 1):
    txt_path = TRANSCRIPT_DIR / (mp3_path.stem + ".txt")
    if txt_path.exists():
        print(f"[{i}/{total}] Skip: {mp3_path.name}")
        continue

    print(f"[{i}/{total}] Transcribing: {mp3_path.name}")
    try:
        segments, info = model.transcribe(
            str(mp3_path),
            language="zh",
            beam_size=5,
            vad_filter=True,
        )
        text = "\n".join(seg.text.strip() for seg in segments if seg.text.strip())
        txt_path.write_text(f"# {mp3_path.stem}\n\n{text}\n", encoding="utf-8")
        print(f"     -> {len(text)} chars saved.")
    except Exception as e:
        print(f"     !! Error: {e}")

print("\nDone! Transcripts saved to:", TRANSCRIPT_DIR)
