# Video Overlay Pipeline

This project turns a raw talking-head video into an edited clip with animated
overlays, timed subtitles, and background music. All of the heavy lifting is
handled by `video_overlay_script.py`, which orchestrates OpenCV for frame-level
work and MoviePy for audio mixing.

## Features
- **Web Interface:** Easy-to-use Flask web application for uploading videos and managing edits
- **TXT File Upload:** Upload your script as a simple .txt file - no Whisper AI needed!
- **Interactive Word Selection:** Click and drag to select multiple words for subtitle assignments
- **Highlight overlays:** Drop a secondary clip on top of the main footage for
  specific phrases. Re‑used clips keep playing across consecutive subtitles and
  stay visible until the next subtitle begins.
- **Subtitle control:** Upload your own transcript as a TXT file. Words are evenly distributed
  across the video duration. Subtitles stay on screen until the next subtitle starts,
  and you can override the text, styling, and layout.
- **Audio mixing:** Preserve the original audio, add per-highlight music beds,
  and/or score the entire output with a global background track.
- **One-stop render:** Outputs the composited video (and, when audio mixing is
  required, a temporary silent pass that is cleaned up automatically).

## Requirements
- Python 3.9+ (MoviePy 2.x requires a fairly recent Python).
- System packages for video encoding/decoding (FFmpeg, libx264).
- Install Python dependencies:

```bash
pip install -r requirements.txt
```

GPU support is optional but recommended for large overlay clips.

## Project Layout
- `app.py` – Flask web server for the frontend interface
- `video_overlay_script.py` – main video processing pipeline
- `templates/index.html` – web interface
- `static/script.js` – frontend JavaScript for interactive features
- `static/styles.css` – modern purple gradient styling
- `demo_project.json` – example configuration describing highlights, subtitles,
  music, and styling (for CLI usage)
- `clips/` & `audio_files/` – reusable overlay/video and music assets
- `uploads/` – uploaded videos
- `outputs/` – processed videos

## Quick Start (Web Interface)

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
   - Navigate to `http://localhost:5000`

3. **Upload your files:**
   - Choose your main video file
   - Choose a .txt file with your script (one line = one subtitle!)
   - Click "Upload & Process"

4. **Assign clips/audio to each subtitle:**
   - You'll see subtitle boxes for each line in your TXT file
   - Click "📹 Assign Clip/Audio" on each subtitle box
   - Choose a video clip or audio file to overlay
   - Set volume and click "Add Highlight"
   - The subtitle box updates to show the assignment ✅
   - Repeat for all subtitles you want to have overlays

5. **Process video:**
   - Click "Process Video"
   - Wait for processing to complete
   - Download your edited video!

### TXT File Format (IMPORTANT!)

**Each line in your TXT file becomes ONE subtitle box!**

Example `script.txt`:
```
Look if you're suffering from neuropathy I need to ask you three quick questions
I need you to answer them honestly
Because what I'm about to reveal could explain why nothing you've tried has ever worked
```

This creates **3 subtitle boxes**:
- Subtitle #1: "Look if you're suffering from neuropathy I need to ask you three quick questions"
- Subtitle #2: "I need you to answer them honestly"
- Subtitle #3: "Because what I'm about to reveal could explain why nothing you've tried has ever worked"

**Tips:**
- Press Enter to create a new subtitle
- Each line = one subtitle
- Empty lines are ignored
- Words are automatically distributed evenly across your video duration

## CLI Usage (Advanced)

For advanced users, you can use the command-line interface directly:

```bash
python video_overlay_script.py \
  --main-video input.mp4 \
  --config demo_project.json \
  --output output.mp4
```

### Configuration File

All runtime options live in a JSON file. Key sections from `demo_project.json`:

- `transcript_text` *(optional)* – supply a transcript instead of running Whisper
- `highlight_assignments` – list of phrases to highlight. Each item can specify:
  - `phrase` / manual `start_word` & `end_word`
  - `clip_path` overlay video
  - `music_path` & `music_volume`
- `global_music_path` / `global_music_volume` – looped music bed that covers the entire output
- `subtitle_sentences` – custom subtitle text mapped to phrases
- `subtitle_design` – colour, font, padding, etc.
- `preserve_audio` – mix the original soundtrack into the final render

You can also provide precomputed `subtitle_segments` (word index pairs) when you want full manual control.

### What Happens Under the Hood

1. The transcript (from TXT file or Whisper) generates word timestamps
2. Highlight phrases are resolved to word ranges and paired with overlay clips
3. Subtitles are rendered frame-by-frame; overlays loop or hold the last frame until the next subtitle starts so there are no gaps
4. If audio mixing is required, a silent video is produced first, then MoviePy merges the requested music layers and deletes the temporary file

### Demo Mode

```bash
python video_overlay_script.py --demo
```

Creates synthetic media, runs the full pipeline, and writes `demo_output.mp4`.

## Tips

- **Web Interface:** The easiest way to use this tool is through the web interface at `http://localhost:5000`
- **TXT Files:** Just write your script in a plain text file - no special formatting needed
- **Multi-word Selection:** Click and drag across words to select multiple words at once
- **Overlay Clips:** Keep video clips in the `clips/` folder and audio files in `audio_files/` folder
- **Frame Rate:** For best results, match the overlay frame rate to the main video
- **Subtitle Styling:** When experimenting with subtitle styling, tweak `subtitle_design` in the config and rerun
- **Server Restart:** If you make code changes, restart the Flask server with `python app.py`

## Troubleshooting

- **"Failed to fetch" error:** Make sure the Flask server is running without auto-reload (`use_reloader=False`)
- **Video processing takes too long:** The server processes videos frame-by-frame, which can take time for long videos
- **Clips not showing:** Ensure video clips are in `clips/` and audio files are in `audio_files/`

---

With the web interface, you can quickly create professional video edits by simply uploading your video and script, selecting words, and assigning clips. Perfect for templated ad creation or social-media highlight packaging!
