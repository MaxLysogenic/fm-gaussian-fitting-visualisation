import subprocess
import tempfile
from pathlib import Path

script_directory = Path(__file__).resolve().parent
video_directory = script_directory / "generated_videos"

# Define input files in order
clips = [video_directory / f"smt_{i}.mp4" for i in range(1, 6)]

# Verify all files exist before starting
for clip in clips:
    if not clip.is_file():
        raise FileNotFoundError(
            f"Missing video clip: {clip}. Run smt_1.py through smt_5.py first."
        )

output_filename = video_directory / "smt_all.mp4"

# Add a three-second freeze-frame to each clip before concatenating them.
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    list_filename = temp_path / "concat_list.txt"
    processed_clips = []

    for index, clip in enumerate(clips, start=1):
        processed_clip = temp_path / f"clip_{index}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(clip),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "tpad=stop_mode=clone:stop_duration=3",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(processed_clip),
        ]
        subprocess.run(cmd, check=True)
        processed_clips.append(processed_clip.name)

    with list_filename.open("w") as file_list:
        for clip in processed_clips:
            file_list.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_filename),
        "-c",
        "copy",
        output_filename,
    ]
    subprocess.run(cmd, check=True)

print(f"\nSuccess! Master video saved as: {output_filename}")
