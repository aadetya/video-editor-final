from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import cv2
import numpy as np

try:
    import moviepy.editor as mpy

    HAVE_MOVIEPY = True
except ImportError:
    HAVE_MOVIEPY = False

try:
    import whisper
    HAVE_WHISPER = True
except ImportError:
    HAVE_WHISPER = False

try:
    import torch
    import whisperx

    HAVE_WHISPERX = True
except ImportError:
    HAVE_WHISPERX = False

try:
    from PIL import Image, ImageDraw, ImageFont

    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False



PIL_FONT_CACHE: Dict[Tuple[str, int], "ImageFont.FreeTypeFont"] = {}



# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


@dataclass
class SubtitleDesign:
    """Configuration parameters controlling the look and feel of subtitles."""

    bar_color: Tuple[int, int, int] = (0, 0, 0)  # Background colour (BGR)
    bar_opacity: float = 0.75  # Opacity of subtitle background (0–1)
    text_color: Tuple[int, int, int] = (255, 255, 255)  # Primary subtitle text colour
    text_scale: float = 1.25  # Scale factor for cv2.putText (fallback)
    text_thickness: int = 2  # Thickness for cv2.putText (fallback)
    outline_color: Tuple[int, int, int] = (0, 0, 0)  # Colour for text outline
    outline_thickness: int = 0  # Thickness of the outline
    highlight_color: Tuple[int, int, int] = (255, 230, 90)  # Highlight pill colour (BGR)
    highlight_text_color: Tuple[int, int, int] = (255, 255, 255)  # Highlighted text colour
    margin: int = 0  # Legacy inner padding (use margin_x/margin_y)
    margin_x: int = 6  # Horizontal padding inside subtitle box
    margin_y: int = 0  # Vertical padding inside subtitle box
    bottom_margin: int = 30  # Gap between subtitle box and frame bottom
    max_line_width_ratio: float = 0.72  # Max text block width relative to frame width
    line_spacing: int = 10  # Pixels between lines inside subtitle box
    corner_radius: int = 4  # Rounded corner radius in pixels
    highlight_padding: Tuple[int, int] = (3, 1)  # Extra padding around highlighted words
    box_shadow_offset: Tuple[int, int] = (8, 10)  # Drop shadow offset for the box
    box_shadow_blur: int = 25  # Gaussian blur kernel size for the box shadow
    box_shadow_alpha: float = 0.55  # Alpha applied to the box shadow
    shadow_color: Tuple[int, int, int] = (0, 0, 0)  # Drop shadow colour
    shadow_offset: Tuple[int, int] = (8, 10)  # Drop shadow pixel offset
    shadow_thickness: int = 10  # Drop shadow thickness
    font: int = cv2.FONT_HERSHEY_SIMPLEX  # Fallback Hershey font
    font_path: Optional[str] = "fonts/Montserrat-SemiBold.ttf"  # Optional path to a TTF font
    font_size_px: int = 54  # Font size in pixels when using TTF fonts


@dataclass
class HighlightAssignment:
    """Input description of a highlight segment selected by the user."""

    phrase: Optional[str] = None  # Natural language selection (exact words)
    clip_path: Optional[str] = None  # Optional overlay clip
    music_path: Optional[str] = None  # Optional music file
    music_volume: float = 1.0  # Gain to apply to the music
    occurrence: int = 1  # When the phrase appears multiple times, which one to use
    start_word: Optional[int] = None  # Manual override for the first word index
    end_word: Optional[int] = None  # Manual override for the last word index


@dataclass
class SubtitleSentence:
    """Optional per-sentence subtitle override."""

    text: str  # Text to render on screen
    phrase: Optional[str] = None  # Phrase to align within the transcript (defaults to ``text``)
    occurrence: int = 1  # Which occurrence to align if the phrase repeats
    start_word: Optional[int] = None  # Manual override for the first word index
    end_word: Optional[int] = None  # Manual override for the last word index


@dataclass
class ProjectConfig:
    """All inputs required to render a project."""

    main_video_path: str
    output_path: str = "output.mp4"
    transcript_text: Optional[str] = None  # Manual transcript content (unused when Whisper enforced)
    whisper_model: str = "base"
    highlight_assignments: List[HighlightAssignment] = field(default_factory=list)
    preserve_audio: bool = True
    global_music_path: Optional[str] = None  # Optional background music for the entire video
    global_music_volume: float = 1.0  # Gain applied to the global music track
    subtitle_design: SubtitleDesign = field(default_factory=SubtitleDesign)
    subtitle_segments: Optional[List[Tuple[int, int]]] = None
    subtitle_sentences: List[SubtitleSentence] = field(default_factory=list)
    aspect_ratio: str = "4:5"  # Aspect ratio: "4:5" or "9:16"


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def get_subtitle_design_for_aspect_ratio(aspect_ratio: str) -> SubtitleDesign:
    """Get default subtitle design optimized for the given aspect ratio."""
    
    if aspect_ratio == "9:16":
        # TikTok-style portrait/vertical video design with Proxima Nova
        return SubtitleDesign(
            bar_color=(0, 0, 0),
            bar_opacity=0.0,  # No background box for TikTok style
            text_color=(255, 255, 255),  # White fill
            text_scale=1.2,
            text_thickness=3,  # Bold text
            outline_color=(0, 0, 0),  # Black outline/stroke
            outline_thickness=5,  # Thick black stroke/outline (5px)
            highlight_color=(255, 230, 90),
            highlight_text_color=(255, 255, 255),
            margin=0,
            margin_x=6,  # Same padding as 4:5 for consistent alignment
            margin_y=0,
            bottom_margin=400,  # Position higher up for better alignment
            max_line_width_ratio=0.72,  # Same as 4:5 for consistent text wrapping and alignment
            line_spacing=4,  # Tight line spacing
            corner_radius=0,  # No rounded corners without box
            highlight_padding=(4, 2),
            box_shadow_offset=(0, 0),  # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),  # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Poppins-SemiBold.ttf",  # Proxima Nova Bold/Black
            font_size_px=54,  # Same size as 4:5
        )
    else:
        # Default 4:5 design (landscape/square) - exact attributes as specified
        return SubtitleDesign(
            bar_color=(0, 0, 0),  # Background colour (BGR)
            bar_opacity=0.75,  # Opacity of subtitle background (0–1)
            text_color=(255, 255, 255),  # Primary subtitle text colour
            text_scale=1.25,  # Scale factor for cv2.putText (fallback)
            text_thickness=2,  # Thickness for cv2.putText (fallback)
            outline_color=(0, 0, 0),  # Colour for text outline
            outline_thickness=0,  # Thickness of the outline
            highlight_color=(255, 230, 90),  # Highlight pill colour (BGR)
            highlight_text_color=(255, 255, 255),  # Highlighted text colour
            margin=0,  # Legacy inner padding (use margin_x/margin_y)
            margin_x=6,  # Horizontal padding inside subtitle box
            margin_y=0,  # Vertical padding inside subtitle box
            bottom_margin=30,  # Gap between subtitle box and frame bottom
            max_line_width_ratio=0.72,  # Max text block width relative to frame width
            line_spacing=10,  # Pixels between lines inside subtitle box
            corner_radius=4,  # Rounded corner radius in pixels
            highlight_padding=(3, 1),  # Extra padding around highlighted words
            box_shadow_offset=(8, 10),  # Drop shadow offset for the box
            box_shadow_blur=25,  # Gaussian blur kernel size for the box shadow
            box_shadow_alpha=0.55,  # Alpha applied to the box shadow
            shadow_color=(0, 0, 0),  # Drop shadow colour
            shadow_offset=(8, 10),  # Drop shadow pixel offset
            shadow_thickness=10,  # Drop shadow thickness
            font=cv2.FONT_HERSHEY_SIMPLEX,  # Fallback Hershey font
            font_path="fonts/Montserrat-SemiBold.ttf",  # Optional path to a TTF font
            font_size_px=54,  # Font size in pixels when using TTF fonts
        )


def parse_aspect_ratio(aspect_ratio_str: str) -> float:
    """Parse aspect ratio string (e.g., '4:5' or '9:16') to float."""
    if aspect_ratio_str == "9:16":
        return 9.0 / 16.0
    elif aspect_ratio_str == "4:5":
        return 4.0 / 5.0
    else:
        # Default to 4:5 if invalid
        return 4.0 / 5.0


def normalise_word(token: str) -> str:
    """Lower-case alphanumeric tokenisation for fuzzy matching."""

    return "".join(ch for ch in token.lower() if ch.isalnum())


def slow_down_video(input_path: str, output_path: str, speed_factor: float) -> None:
    """
    Slow down a video using ffmpeg to match a target duration.
    
    Args:
        input_path: Path to input video
        output_path: Path to output slowed-down video
        speed_factor: Speed factor (e.g., 0.5 for half speed, 2.0 for double speed)
                      If clip is 5s and needs to be 10s, speed_factor = 5/10 = 0.5
    """
    if speed_factor >= 1.0:
        # No need to slow down, just copy the file
        import shutil
        shutil.copy2(input_path, output_path)
        return
    
    # Check if video has audio stream
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", input_path],
            capture_output=True,
            text=True,
            check=True
        )
        has_audio = result.stdout.strip() == "audio"
    except Exception:
        has_audio = False
    
    video_filter = f"setpts={1.0/speed_factor}*PTS"
    
    if has_audio:
        # Use ffmpeg to slow down the video with audio
        # atempo filter works on audio (0.5 to 2.0 range, can chain for more)
        # For speed_factor < 0.5, we need to chain multiple atempo=0.5 filters
        audio_filters = []
        remaining_factor = speed_factor
        
        # Chain atempo=0.5 filters until remaining factor is >= 0.5
        # Use a small epsilon to handle floating point precision
        while remaining_factor < 0.5 - 1e-6:
            audio_filters.append("atempo=0.5")
            remaining_factor = remaining_factor / 0.5
        
        # Apply the final atempo filter if needed
        if remaining_factor != 1.0:
            # Clamp to valid range [0.5, 2.0]
            final_tempo = max(0.5, min(2.0, remaining_factor))
            # Safety check: if somehow still below 0.5, add another atempo=0.5
            if final_tempo < 0.5:
                audio_filters.append("atempo=0.5")
                final_tempo = final_tempo / 0.5
                final_tempo = max(0.5, min(2.0, final_tempo))
            if abs(final_tempo - 1.0) > 1e-6:  # Only add if significantly different from 1.0
                audio_filters.append(f"atempo={final_tempo}")
        
        audio_filter = ",".join(audio_filters) if audio_filters else "atempo=1.0"
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex",
            f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ]
    else:
        # Video without audio - only slow down video
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:v", video_filter,
            "-c:v", "libx264",
            "-an",  # No audio
            output_path
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"  [CLIP PROCESSING] Slowed down clip: {os.path.basename(input_path)} by factor {speed_factor:.2f}")
    except subprocess.CalledProcessError as e:
        logger.error(f"  [CLIP PROCESSING] Error slowing down clip: {e.stderr}")
        # Fallback: just copy the original
        import shutil
        shutil.copy2(input_path, output_path)


def probe_video_metadata(path: str) -> Tuple[float, int, int, int, float]:
    """Return fps, frame_count, width, height, duration for ``path``."""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0
    duration = frame_count / fps if frame_count else 0.0
    return fps, frame_count, width, height, duration


# --------------------------------------------------------------------------- #
# Transcript generation
# --------------------------------------------------------------------------- #


def evenly_spaced_transcript(
    text: str, total_duration: float
) -> List[Dict[str, float]]:
    """Generate timestamps by distributing words uniformly across the duration."""

    words = text.strip().split()
    if not words:
        return []
    duration_per_word = total_duration / len(words) if total_duration > 0 else 0.5
    transcript: List[Dict[str, float]] = []
    pointer = 0.0
    for word in words:
        start = pointer
        end = pointer + duration_per_word
        transcript.append({"word": word, "start_time": start, "end_time": end})
        pointer = end
    return transcript

def save_the_transcribe_text(text:str, filename:str):
    filename, _ = os.path.splitext(filename)
    file_name = f"{filename}.txt"
    with open(file_name,"w") as f:
        f.write(text)

    print(f"File Transcript text saved to {file_name}")

def transcribe_audio_whisper(
    audio_path: str, model_size: str = "base"
) -> List[Dict[str, float]]:
    """Transcribe an audio or video file using Whisper (word level timestamps)."""

    if not HAVE_WHISPER:
        raise ImportError(
            "Whisper is not installed. Please install openai-whisper to transcribe automatically."
        )

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)
    transcript: List[Dict[str, float]] = []
    save_the_transcribe_text(result['text'], audio_path)

    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            word = word_data.get("word", "").strip()
            if not word:
                continue
            transcript.append(
                {
                    "word": word,
                    "start_time": float(word_data["start"]),
                    "end_time": float(word_data["end"]),
                }
            )
    return transcript

def transcribe_audio_whisperx(
    audio_path: str,
    model_size: str = "base",
) -> List[Dict[str, float]]:
    """
    Transcribe + ALIGN using WhisperX for more accurate, waveform-based word timings.
    Returns the same transcript format as transcribe_audio_whisper:
        [{"word": "...", "start_time": float, "end_time": float}, ...]
    """

    if not HAVE_WHISPERX:
        raise ImportError(
            "WhisperX is not installed. Please install whisperx + torch to use waveform alignment."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANT:
    # - On GPU we can use float16 (fast).
    # - On CPU we can only use int8 or float32 instead.
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(
        f"  [TRANSCRIPT] Using WhisperX on {device} (model: {model_size}) for waveform-aligned transcript..."
    )

    # 1) Load audio
    audio = whisperx.load_audio(audio_path)

    # 2) Run WhisperX transcription (segment-level)
    model = whisperx.load_model(model_size, device=device,compute_type=compute_type)
    # You can tune batch_size if needed
    result = model.transcribe(audio, batch_size=16)

    # Save raw text for debugging (reuses your helper)
    if "text" in result:
        save_the_transcribe_text(result["text"], audio_path)

    # 3) Load alignment model for the detected language
    lang = result.get("language", "en")
    logger.info(f"  [TRANSCRIPT] WhisperX detected language: {lang}")
    align_model, metadata = whisperx.load_align_model(
        language_code=lang, device=device
    )

    # 4) Run alignment to get precise word timings
    aligned_result = whisperx.align(
        result["segments"], align_model, metadata, audio, device
    )

    word_segments = aligned_result.get("word_segments", [])
    transcript: List[Dict[str, float]] = []

    for w in word_segments:
        # WhisperX usually uses "word"; older versions may use "text"
        token = (w.get("word") or w.get("text") or "").strip()
        if not token:
            continue
        start = float(w["start"])
        end = float(w["end"])
        transcript.append(
            {
                "word": token,
                "start_time": start,
                "end_time": end,
            }
        )

    logger.info(
        f"  [TRANSCRIPT] WhisperX produced {len(transcript)} waveform-aligned words."
    )
    return transcript

def build_transcript(
    video_path: str,
    transcript_text: Optional[str],
    whisper_model: str,
) -> List[Dict[str, float]]:
    """
    Create a per-word transcript, preferring waveform-aligned WhisperX if available,
    and falling back to standard Whisper word timestamps otherwise.

    Returns:
        List[{"word": str, "start_time": float, "end_time": float}, ...]
    """

    # 1) Prefer WhisperX if installed
    if HAVE_WHISPERX:
        try:
            logger.info(
                f"  [TRANSCRIPT] Trying WhisperX waveform-aligned transcript (model: {whisper_model})..."
            )
            start_time = time.time()
            transcript = transcribe_audio_whisperx(video_path, whisper_model)
            duration = time.time() - start_time

            if transcript:
                # Save aligned transcript to JSON, same as before
                write_subtitle_into_file(video_path, transcript)
                logger.info(
                    f"  [TRANSCRIPT] ✓ WhisperX transcript completed in {duration:.2f}s ({duration/60:.2f} min)"
                )
                logger.info(
                    f"  [TRANSCRIPT] Generated {len(transcript)} waveform-aligned words"
                )
                return transcript
            else:
                logger.warning(
                    "  [TRANSCRIPT] WhisperX returned an empty transcript; falling back to Whisper."
                )
        except Exception as exc:
            logger.error(
                f"  [TRANSCRIPT] ✗ WhisperX alignment failed: {exc}. Falling back to Whisper."
            )

    # 2) Fallback: standard Whisper (your old behaviour)
    try:
        logger.info(
            f"  [TRANSCRIPT] Transcribing with Whisper (model: {whisper_model})..."
        )
        whisper_start = time.time()
        transcript = transcribe_audio_whisper(video_path, whisper_model)
        whisper_duration = time.time() - whisper_start

        if transcript:
            write_subtitle_into_file(video_path, transcript)
            logger.info(
                f"  [TRANSCRIPT] ✓ Whisper transcription completed in {whisper_duration:.2f}s ({whisper_duration/60:.2f} min)"
            )
            logger.info(
                f"  [TRANSCRIPT] Generated {len(transcript)} words (non-aligned Whisper)"
            )
            return transcript
        else:
            logger.error(
                "  [TRANSCRIPT] ✗ Whisper returned an empty transcript; cannot proceed."
            )
            raise RuntimeError("Whisper returned an empty transcript; cannot proceed.")
    except Exception as exc:
        logger.error(f"  [TRANSCRIPT] ✗ Whisper transcription failed: {exc}")
        raise RuntimeError(
            "Both WhisperX (if available) and Whisper transcription failed. "
            "Ensure the models are installed and the video audio is accessible."
        ) from exc




def write_subtitle_into_file(
    input_file_name: str, transcript: List[Dict[str, float]]
):
    filename, _ = os.path.splitext(input_file_name)
    file_name = f"{filename}_subtitle.json"

    with open(file_name, "w") as f:
        json.dump(transcript, f, indent=4)

    print(f"Saved the video file transcription in {file_name}")




# --------------------------------------------------------------------------- #
# Highlight mapping helpers
# --------------------------------------------------------------------------- #


def _find_exact_phrase_match(
    transcript_words: Sequence[str],
    tokens: Sequence[str],
    occurrence: int,
    start_index: int,
) -> Optional[Tuple[int, int]]:
    """Return the ``occurrence``-th exact match after ``start_index`` if available."""

    token_list = list(tokens)
    target_len = len(token_list)
    if target_len == 0:
        return None

    match_count = 0
    for idx in range(start_index, len(transcript_words) - target_len + 1):
        if transcript_words[idx : idx + target_len] == token_list:
            match_count += 1
            if match_count == occurrence:
                return idx, idx + target_len - 1
    return None


def _find_fuzzy_phrase_match(
    transcript_words: Sequence[str],
    tokens: Sequence[str],
    occurrence: int,
    start_index: int,
    max_window_delta: int = 2,
    min_similarity: float = 0.80,
) -> Optional[Tuple[int, int]]:
    """Find an approximate match allowing small spelling/word-count deviations.

    When multiple fuzzy matches are possible, we prefer:
      1. Highest similarity ratio
      2. Then earliest start index
    so that occurrence=1 gives you the best match, not just the first one
    in time.
    """
    token_list = list(tokens)
    if not token_list or start_index >= len(transcript_words):
        return None

    min_window = max(1, len(token_list) - max_window_delta)
    max_window = max(min_window, len(token_list) + max_window_delta)
    phrase_str = " ".join(token_list)

    matches: List[Tuple[int, int, float]] = []

    for idx in range(start_index, len(transcript_words) - min_window + 1):
        # First word must still match to keep things sane
        if transcript_words[idx] != token_list[0]:
            continue

        remaining = len(transcript_words) - idx
        window_max = min(max_window, remaining)

        best_ratio = 0.0
        best_end: Optional[int] = None

        for window_len in range(min_window, window_max + 1):
            window_tokens = transcript_words[idx : idx + window_len]
            ratio = SequenceMatcher(None, phrase_str, " ".join(window_tokens)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_end = idx + window_len - 1

        if best_end is not None and best_ratio >= min_similarity:
            matches.append((idx, best_end, best_ratio))

    if not matches:
        return None

    if occurrence > len(matches):
        return None

    # Sort: highest similarity first, then earliest index
    matches.sort(key=lambda item: (-item[2], item[0]))

    start, end, _ratio = matches[occurrence - 1]
    return start, end



def find_phrase_indices(
    transcript_words: Sequence[str],
    phrase: str,
    occurrence: int = 1,
    start_index: int = 0,
) -> Tuple[int, int]:
    """Locate phrase within transcript_words returning (start, end) indices.

    When an exact match cannot be found, a fuzzy search is attempted that tolerates
    small spelling differences (e.g. "infirmary" vs "infermary") or short missing
    words. This keeps subtitle alignment resilient to light transcript noise.
    """

    if not phrase:
        raise ValueError("Phrase must be provided when start/end indices are omitted.")

    target_tokens = [normalise_word(tok) for tok in phrase.split()]
    target_tokens = [tok for tok in target_tokens if tok]
    if not target_tokens:
        raise ValueError("Phrase must contain at least one word.")

    start_index = max(0, int(start_index or 0))
    target_occurrence = max(1, int(occurrence or 1))

    match = _find_exact_phrase_match(
        transcript_words, target_tokens, target_occurrence, start_index
    )
    if match:
        return match

    match = _find_fuzzy_phrase_match(
        transcript_words, target_tokens, target_occurrence, start_index
    )
    if match:
        return match

    raise ValueError(
        f"Phrase '{phrase}' occurrence {target_occurrence} not found in transcript."
    )


def map_assignments_to_segments(
    transcript: List[Dict[str, float]],
    assignments: Sequence[HighlightAssignment],
) -> List[Dict[str, Optional[object]]]:
    """Convert user highlight selections into rendering segments.

    This version is phrase-driven whenever a phrase is provided:
    - We always remap (start_word, end_word) from the phrase text in the transcript,
      ignoring any pre-filled indices for that assignment.
    - We then tighten the span so that it starts on the first token of the phrase
      and ends on the last token of the phrase.
    """

    if not transcript or not assignments:
        return []

    normalised_transcript = [normalise_word(entry["word"]) for entry in transcript]
    mapped: List[Dict[str, Optional[object]]] = []

    # Enforce monotonic mapping: every subsequent phrase is searched *after*
    # the end of the previous one.
    search_start = 0

    for assignment in assignments:
        phrase_text = (assignment.phrase or "").strip()
        start_word = assignment.start_word
        end_word = assignment.end_word

        if phrase_text:
            # Always derive indices from the phrase when available,
            # so the clip is anchored to the actual phrase text in the audio.
            try:
                start_word, end_word = find_phrase_indices(
                    normalised_transcript,
                    phrase_text,
                    occurrence=assignment.occurrence,
                    start_index=search_start,
                )
            except ValueError:
                # Fallback: try full transcript so we fail less catastrophically.
                start_word, end_word = find_phrase_indices(
                    normalised_transcript,
                    phrase_text,
                    occurrence=assignment.occurrence,
                    start_index=0,
                )
        else:
            # No phrase text: we require explicit indices.
            if start_word is None or end_word is None:
                raise ValueError(
                    "HighlightAssignment without phrase must include "
                    "explicit start_word/end_word indices."
                )
            start_word = int(start_word)
            end_word = int(end_word)

        start_word = int(start_word)
        end_word = int(end_word)

        if (
            start_word < 0
            or end_word >= len(transcript)
            or start_word > end_word
        ):
            raise ValueError(
                f"Invalid word indices resolved for assignment {assignment}: "
                f"({start_word}, {end_word})"
            )

        # If we have phrase text, tighten the span so it *exactly* covers the phrase.
        if phrase_text:
            phrase_tokens = [
                normalise_word(tok) for tok in phrase_text.split()
                if normalise_word(tok)
            ]

            if phrase_tokens:
                first_token = phrase_tokens[0]
                last_token = phrase_tokens[-1]

                segment_tokens = normalised_transcript[start_word : end_word + 1]

                # Move start_word FORWARD to the first occurrence of the first token,
                # if it exists inside the current segment. This prevents leading
                # words like "correctly." from being included when the phrase is
                # "Because it's been tested.".
                try:
                    offset = segment_tokens.index(first_token)
                except ValueError:
                    offset = None

                if offset is not None:
                    new_start = start_word + offset
                    if new_start <= end_word:
                        start_word = new_start

                # Ensure the mapped span includes the LAST token of the phrase.
                segment_tokens = normalised_transcript[start_word : end_word + 1]
                if last_token not in segment_tokens:
                    for j in range(end_word + 1, len(normalised_transcript)):
                        if normalised_transcript[j] == last_token:
                            end_word = j
                            break

                    segment_tokens = normalised_transcript[start_word : end_word + 1]
                    if last_token not in segment_tokens:
                        logger.warning(
                            "Phrase %r not fully matched in transcript "
                            "(last token %r missing between words %d and %d).",
                            phrase_text,
                            last_token,
                            start_word,
                            end_word,
                        )

        # Final safety clamp
        if start_word < 0:
            start_word = 0
        if end_word >= len(transcript):
            end_word = len(transcript) - 1
        if end_word < start_word:
            end_word = start_word

        snippet_words = " ".join(
            entry["word"] for entry in transcript[start_word : end_word + 1]
        )
        start_time = transcript[start_word]["start_time"]
        end_time = transcript[end_word]["end_time"]
        logger.info(
            "[MAP] clip=%s phrase=%r -> words[%d:%d] (%.3fs-%.3fs): %s",
            getattr(assignment, "clip_path", None),
            phrase_text,
            start_word,
            end_word,
            start_time,
            end_time,
            snippet_words,
        )

        mapped.append(
            {
                "start_word": start_word,
                "end_word": end_word,
                "clip_path": assignment.clip_path,
                "music_path": assignment.music_path,
                "music_volume": float(assignment.music_volume),
            }
        )

        # Move the search window forward so later phrases can’t snap backwards
        search_start = max(search_start, end_word + 1)

    return mapped


def map_subtitle_sentences(
    transcript: List[Dict[str, float]],
    sentences: Sequence[SubtitleSentence],
) -> List[Dict[str, object]]:
    """Align custom subtitle sentences with the transcript."""

    if not transcript or not sentences:
        return []

    normalised_transcript = [normalise_word(entry["word"]) for entry in transcript]
    mapped: List[Dict[str, object]] = []
    search_start = 0

    for sentence in sentences:
        text = sentence.text.strip()
        if not text:
            continue
        phrase = (sentence.phrase or sentence.text).strip()
        start_word = sentence.start_word
        end_word = sentence.end_word

        if start_word is not None and end_word is not None:
            start_word = int(start_word)
            end_word = int(end_word)
        else:
            target_occurrence = max(1, int(sentence.occurrence or 1))
            start_word, end_word = find_phrase_indices(
                normalised_transcript,
                phrase,
                occurrence=target_occurrence,
                start_index=search_start,
            )

        if start_word < 0 or end_word >= len(transcript) or start_word > end_word:
            raise ValueError(f"Invalid indices resolved for subtitle sentence '{sentence.text}'.")

        mapped.append(
            {
                "start_word": start_word,
                "end_word": end_word,
                "text": text,
            }
        )

        search_start = max(search_start, end_word + 1)

    return mapped


def generate_default_subtitle_segments(
    transcript: List[Dict[str, float]],
    highlight_segments: Sequence[Dict[str, Optional[object]]],
    block_size: int = 8,
) -> List[Tuple[int, int]]:
    """Generate steady subtitle groupings covering the full transcript."""

    total_words = len(transcript)
    if total_words == 0:
        return []

    sorted_highlights = sorted(
        [
            (int(seg["start_word"]), int(seg["end_word"]))
            for seg in highlight_segments
            if seg
        ],
        key=lambda pair: pair[0],
    )

    segments: List[Tuple[int, int]] = []
    highlight_idx = 0
    current_word = 0

    while current_word < total_words:
        if highlight_idx < len(sorted_highlights):
            highlight_start, highlight_end = sorted_highlights[highlight_idx]
            if current_word > highlight_end:
                highlight_idx += 1
                continue
            if current_word == highlight_start:
                segments.append((highlight_start, highlight_end))
                current_word = highlight_end + 1
                highlight_idx += 1
                continue
            next_highlight_start = highlight_start
        else:
            next_highlight_start = total_words

        block_end = min(next_highlight_start - 1, current_word + block_size - 1)
        if block_end < current_word:
            current_word = next_highlight_start
            continue
        segments.append((current_word, block_end))
        current_word = block_end + 1

    return segments


def safe_audio_subclip(
    audio_clip: Optional[mpy.AudioClip], start: float, end: float
) -> Optional[mpy.AudioClip]:
    """Return a trimmed audio clip compatible across MoviePy versions."""

    if audio_clip is None:
        return None
    if end <= start:
        return None
    if hasattr(audio_clip, "subclip"):
        return audio_clip.subclip(start, end)
    if hasattr(audio_clip, "subclipped"):
        return audio_clip.subclipped(start, end)
    raise AttributeError("Audio clip does not support subclip/subclipped trimming.")


# --------------------------------------------------------------------------- #
# Video overlay / subtitle rendering    
# --------------------------------------------------------------------------- #


def crop_to_aspect_ratio(frame: np.ndarray, target_ratio: float) -> np.ndarray:
    """Centre-crop ``frame`` to match ``target_ratio`` expressed as width / height."""

    if frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 1e-3:
        return frame
    if current_ratio > target_ratio:
        # Too wide
        new_width = int(height * target_ratio)
        offset = max((width - new_width) // 2, 0)
        return frame[:, offset : offset + new_width]
    # Too tall
    new_height = int(width / target_ratio)
    offset = max((height - new_height) // 2, 0)
    return frame[offset : offset + new_height, :]


def compute_cropped_dimensions(
    width: int, height: int, target_ratio: float
) -> Tuple[int, int]:
    """Return (width, height) after centre-cropping to ``target_ratio``."""

    if width <= 0 or height <= 0:
        return width, height
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 1e-6:
        return width, height
    if current_ratio > target_ratio:
        cropped_width = int(height * target_ratio)
        return cropped_width, height
    cropped_height = int(width / target_ratio)
    return width, cropped_height


def resize_overlay_for_canvas(
    frame: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    aspect_ratio: float,
    coverage: float = 1.0,
) -> np.ndarray:
    """Resize overlay so it fits within the canvas while keeping ``aspect_ratio``."""

    if frame.size == 0:
        return frame
    target_height = int(canvas_height * coverage)
    target_width = int(target_height * aspect_ratio)
    if target_width > canvas_width * coverage:
        target_width = int(canvas_width * coverage)
        target_height = int(target_width / aspect_ratio)
    target_width = max(1, min(canvas_width, target_width))
    target_height = max(1, min(canvas_height, target_height))
    return cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def shadowed_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    box_color: Tuple[int, int, int],
    box_alpha: float,
    shadow_offset: Tuple[int, int],
    shadow_blur: int,
    shadow_alpha: float,
    radius: int,
) -> np.ndarray:
    """Draw a rounded rectangle with a blurred drop shadow onto ``img``."""

    x = int(round(x))
    y = int(round(y))
    w = max(0, int(round(w)))
    h = max(0, int(round(h)))
    if w == 0 or h == 0:
        return img

    base = img.copy()

    def round_fill(dst: np.ndarray, x0: int, y0: int, width: int, height: int, rad: int, color: Tuple[int, int, int]) -> None:
        rad = max(0, min(rad, min(width, height) // 2))
        cv2.rectangle(dst, (x0 + rad, y0), (x0 + width - rad, y0 + height), color, -1)
        cv2.rectangle(dst, (x0, y0 + rad), (x0 + width, y0 + height - rad), color, -1)
        for cx, cy in (
            (x0 + rad, y0 + rad),
            (x0 + width - rad, y0 + rad),
            (x0 + rad, y0 + height - rad),
            (x0 + width - rad, y0 + height - rad),
        ):
            cv2.circle(dst, (cx, cy), rad, color, -1)

    if shadow_alpha > 0 and shadow_blur > 0:
        shadow = np.zeros_like(img)
        sx = x + int(shadow_offset[0])
        sy = y + int(shadow_offset[1])
        round_fill(shadow, sx, sy, w, h, radius, (0, 0, 0))
        ksize = shadow_blur | 1  # ensure odd
        shadow = cv2.GaussianBlur(shadow, (ksize, ksize), 0)
        img = cv2.addWeighted(shadow, shadow_alpha, img, 1.0, 0)

    overlay = base.copy()
    round_fill(overlay, x, y, w, h, radius, box_color)
    if box_alpha >= 1:
        img = overlay
    else:
        img = cv2.addWeighted(overlay, box_alpha, img, 1.0 - box_alpha, 0)
    return img


def get_pil_font(font_path: str, font_size: int) -> "ImageFont.FreeTypeFont":
    """Load and cache a PIL font."""

    cache_key = (font_path, font_size)
    font = PIL_FONT_CACHE.get(cache_key)
    if font is None:
        font = ImageFont.truetype(font_path, font_size)
        PIL_FONT_CACHE[cache_key] = font
    return font


def draw_subtitle_on_frame(
    frame: np.ndarray,
    transcript: List[Dict[str, float]],
    current_time: float,
    design: SubtitleDesign,
    highlight_ranges: List[Tuple[int, int]],
    subtitle_segments: Optional[List[Tuple[int, int]]] = None,
    custom_subtitles: Optional[List[str]] = None,
    aspect_ratio: Optional[str] = None,
) -> np.ndarray:
    """Draw a subtitle bar on ``frame`` based on the current playback time."""

    height, width = frame.shape[:2]
    annotated = frame.copy()

    if not transcript:
        return annotated

    active_segment_index: Optional[int] = None
    if subtitle_segments:
        previous_candidate: Optional[int] = None
        for idx, (seg_start, seg_end) in enumerate(subtitle_segments):
            start_t = transcript[seg_start]["start_time"]
            end_t = transcript[seg_end]["end_time"]
            if start_t <= current_time <= end_t:
                active_segment_index = idx
                break
            if current_time < start_t:
                if previous_candidate is not None:
                    active_segment_index = previous_candidate
                break
            previous_candidate = idx
        else:
            if previous_candidate is not None:
                active_segment_index = previous_candidate

    words_to_display: List[Tuple[int, str]] = []
    if active_segment_index is not None and subtitle_segments:
        seg_start, seg_end = subtitle_segments[active_segment_index]
        words_to_display = [
            (idx, transcript[idx]["word"]) for idx in range(seg_start, seg_end + 1)
        ]
    elif subtitle_segments is None:
        display_window = 2.6
        for idx, entry in enumerate(transcript):
            midpoint = (entry["start_time"] + entry["end_time"]) / 2.0
            if abs(midpoint - current_time) <= display_window / 2:
                words_to_display.append((idx, entry["word"]))

    if subtitle_segments and active_segment_index is None:
        # No subtitle for this moment when explicit segments are supplied.
        return annotated

    use_pil_font = (
        HAVE_PIL and design.font_path is not None and os.path.exists(design.font_path)
    )
    pil_font: Optional["ImageFont.FreeTypeFont"] = None
    pil_ascent = 0
    if use_pil_font:
        pil_font = get_pil_font(design.font_path, int(design.font_size_px))
        pil_ascent, pil_descent = pil_font.getmetrics()
        default_line_height = pil_ascent + pil_descent

        def measure_word(text: str) -> Tuple[int, int, int]:
            render_text = text if text else " "
            bbox = pil_font.getbbox(render_text)
            width = int(math.ceil(bbox[2] - bbox[0]))
            height = int(math.ceil(bbox[3] - bbox[1]))
            if width <= 0:
                width = int(math.ceil(pil_font.getlength(render_text)))
            height = max(height, default_line_height)
            ascent = pil_ascent
            return max(width, 1), height, ascent

        space_width = int(math.ceil(pil_font.getlength(" "))) or 6
    else:

        def measure_word(text: str) -> Tuple[int, int, int]:
            ((word_w, word_h), baseline) = cv2.getTextSize(
                text if text else " ",
                design.font,
                design.text_scale,
                design.text_thickness,
            )
            word_w = max(word_w, 1)
            word_h = max(word_h, 1)
            ascent = word_h - baseline
            if ascent <= 0:
                ascent = word_h
            return word_w, word_h, ascent

        space_width = cv2.getTextSize(
            " ", design.font, design.text_scale, design.text_thickness
        )[0][0]
    max_line_width = max(1, int(width * design.max_line_width_ratio))

    def compute_line_width(word_list: List[Dict[str, object]]) -> int:
        width_acc = 0
        for idx_w, word_info in enumerate(word_list):
            if idx_w > 0:
                width_acc += space_width
            width_acc += word_info["width"]
        return width_acc

    word_entries: List[Dict[str, object]] = []
    if (
        custom_subtitles
        and subtitle_segments
        and active_segment_index is not None
        and 0 <= active_segment_index < len(custom_subtitles)
    ):
        custom_text = custom_subtitles[active_segment_index]
        text_lines = [
            line.strip()
            for line in custom_text.replace("\r", "").splitlines()
            if line.strip()
        ]
        if not text_lines:
            text_lines = [custom_text.strip() or custom_text]

        seg_start, seg_end = subtitle_segments[active_segment_index]
        highlight_active = any(
            not (end < seg_start or start > seg_end) for start, end in highlight_ranges
        )

        for idx_line, line_text in enumerate(text_lines):
            words = line_text.split()
            if not words:
                continue
            for word in words:
                word_width, word_height, word_ascent = measure_word(word)
                word_entries.append(
                    {
                        "word": word,
                        "is_highlighted": highlight_active,
                        "width": word_width,
                        "height": word_height,
                        "ascent": word_ascent,
                        "descent": max(0, word_height - word_ascent),
                        "is_forced_break": False,
                    }
                )
            if idx_line != len(text_lines) - 1:
                word_entries.append({"is_forced_break": True})
    else:
        segments_with_highlights: List[Tuple[str, bool]] = []
        for idx, word in words_to_display:
            is_highlighted = any(start <= idx <= end for start, end in highlight_ranges)
            segments_with_highlights.append((word, is_highlighted))

        for word, is_highlighted in segments_with_highlights:
            word_width, word_height, word_ascent = measure_word(word)
            word_entries.append(
                {
                    "word": word,
                    "is_highlighted": is_highlighted,
                    "width": word_width,
                    "height": word_height,
                    "ascent": word_ascent,
                    "descent": max(0, word_height - word_ascent),
                    "is_forced_break": False,
                }
            )

    lines: List[Dict[str, object]] = []
    current_line: List[Dict[str, object]] = []
    current_width = 0

    for entry in word_entries:
        if entry.get("is_forced_break", False):
            if current_line:
                lines.append({"words": current_line, "width": current_width})
                current_line = []
                current_width = 0
            continue

        word_width = entry["width"]

        if current_line:
            prospective_width = current_width + space_width + word_width
        else:
            prospective_width = word_width

        if current_line and prospective_width > max_line_width:
            lines.append({"words": current_line, "width": current_width})
            current_line = []
            current_width = 0

        if current_line:
            current_width += space_width + word_width
        else:
            current_width = word_width

        current_line.append(entry)

    if current_line:
        lines.append({"words": current_line, "width": current_width})

    if len(lines) > 2:
        flattened_words: List[Dict[str, object]] = [
            word_info for line in lines for word_info in line["words"]
        ]
        if flattened_words:
            best_lines: Optional[List[Dict[str, object]]] = None
            best_score = float("inf")
            total_tokens = len(flattened_words)
            for split_idx in range(1, total_tokens):
                first_line = flattened_words[:split_idx]
                second_line = flattened_words[split_idx:]
                if not second_line:
                    continue
                width1 = compute_line_width(first_line)
                width2 = compute_line_width(second_line)
                overflow = max(0, width1 - max_line_width) + max(
                    0, width2 - max_line_width
                )
                score = abs(width1 - width2) + overflow * 5
                if score < best_score:
                    best_score = score
                    best_lines = [
                        {"words": first_line, "width": width1},
                        {"words": second_line, "width": width2},
                    ]
            if best_lines:
                lines = [line for line in best_lines if line["words"]]
            else:
                lines = [
                    {"words": flattened_words, "width": compute_line_width(flattened_words)}
                ]

    if not lines:
        return annotated

    text_block_width = max(line["width"] for line in lines)
    line_ascents: List[int] = []
    line_descents: List[int] = []
    line_heights: List[int] = []
    for line in lines:
        if line["words"]:
            asc = max(word["ascent"] for word in line["words"] if not word.get("is_forced_break", False))
            desc = max(
                word["descent"]
                for word in line["words"]
                if not word.get("is_forced_break", False)
            )
        else:
            asc = 0
            desc = 0
        line_ascents.append(asc)
        line_descents.append(desc)
        line_heights.append(asc + desc)
    line_spacing = max(0, int(design.line_spacing))
    if line_heights:
        text_block_height = (
            sum(line_heights) + max(0, len(line_heights) - 1) * line_spacing
        )
    else:
        text_block_height = 0

    padding_x = getattr(design, "margin_x", design.margin)
    padding_y = getattr(design, "margin_y", design.margin)
    box_width = int(text_block_width + 2 * padding_x)
    box_height = int(text_block_height + 2 * padding_y)
    # Center the box (same as original)
    box_left = int(max(0, (width - box_width) / 2))
    box_right = int(min(width, box_left + box_width))
    line_count = len(lines)
    bottom_margin_dynamic = design.bottom_margin
    if line_count == 1:
        bottom_margin_dynamic = max(0, int(design.bottom_margin * 0.85))
    elif line_count >= 2:
        bottom_margin_dynamic = design.bottom_margin + 8
    box_bottom = height - max(0, bottom_margin_dynamic)
    box_top = box_bottom - box_height
    if box_top < 0:
        box_top = 0
        box_bottom = min(height, box_height)

    # Detect aspect ratio: 4:5 videos have bottom_margin=30, 9:16 have bottom_margin>=300
    is_9_16 = design.bottom_margin >= 300  # 9:16 videos have much higher bottom_margin
    
    # For 4:5 videos, use the exact old script logic (no fade effects, direct opacity)
    if not is_9_16:
        # 4:5 video - use exact old script logic
        annotated = shadowed_rect(
            annotated,
            box_left,
            box_top,
            box_width,
            box_height,
            box_color=design.bar_color,
            box_alpha=design.bar_opacity,
            shadow_offset=getattr(design, "box_shadow_offset", (0, 0)),
            shadow_blur=getattr(design, "box_shadow_blur", 0),
            shadow_alpha=getattr(design, "box_shadow_alpha", 0.0),
            radius=design.corner_radius,
        )
    else:
        # 9:16 video - use current logic with fade effects
        subtitle_opacity = 1.0
        fade_duration = 0.3  # 300ms fade in/out
        if subtitle_segments and active_segment_index is not None:
            seg_start, seg_end = subtitle_segments[active_segment_index]
            start_t = transcript[seg_start]["start_time"]
            end_t = transcript[seg_end]["end_time"]
            
            # Fade in at the start
            if current_time < start_t + fade_duration:
                subtitle_opacity = (current_time - start_t) / fade_duration
                subtitle_opacity = max(0.0, min(1.0, subtitle_opacity))
            # Fade out at the end
            elif current_time > end_t - fade_duration:
                subtitle_opacity = (end_t - current_time) / fade_duration
                subtitle_opacity = max(0.0, min(1.0, subtitle_opacity))
        
        # Apply opacity to box if needed
        effective_box_opacity = design.bar_opacity * subtitle_opacity if design.bar_opacity > 0 else 0.0
        
        # Skip shadow rendering for 9:16 videos for better performance (no shadow/box needed)
        # Only render box if 9:16 has a visible box (bar_opacity > 0)
        if effective_box_opacity > 0:
            annotated = shadowed_rect(
                annotated,
                box_left,
                box_top,
                box_width,
                box_height,
                box_color=design.bar_color,
                box_alpha=effective_box_opacity,
                shadow_offset=getattr(design, "box_shadow_offset", (0, 0)),
                shadow_blur=getattr(design, "box_shadow_blur", 0),
                shadow_alpha=0.0,  # No shadow for 9:16
                radius=design.corner_radius,
            )

    pil_image = None
    pil_draw = None

    y_cursor = box_top + padding_y
    for line_index, line in enumerate(lines):
        words = line["words"]
        if not words:
            continue
        line_ascent = line_ascents[line_index]
        line_descent = line_descents[line_index]
        top_line = y_cursor
        baseline_y = int(top_line + line_ascent)
        line_width = line["width"]
        x_cursor = int((width - line_width) / 2)
        for word_position, word_info in enumerate(words):
            if word_position > 0:
                x_cursor += space_width
            word = word_info["word"]
            word_width = word_info["width"]
            word_height = word_info["height"]
            draw_highlight = False  # disable text colour change when highlighted segments active
            if draw_highlight:
                padding_word_x, padding_word_y = design.highlight_padding
                rect_top_left = (
                    x_cursor - padding_word_x,
                    int(top_line - padding_word_y),
                )
                rect_bottom_right = (
                    x_cursor + word_width + padding_word_x,
                    int(baseline_y + word_info["descent"] + padding_word_y),
                )
                cv2.rectangle(
                    annotated,
                    rect_top_left,
                    rect_bottom_right,
                    design.highlight_color,
                    thickness=-1,
                )
                text_color = design.highlight_text_color
            else:
                text_color = design.text_color

            if use_pil_font and pil_font is not None:
                if pil_image is None:
                    pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    pil_draw = ImageDraw.Draw(pil_image)
                rgb_color = (
                    int(text_color[2]),
                    int(text_color[1]),
                    int(text_color[0]),
                )
                
                if is_9_16 and design.outline_thickness > 0:
                    # For 9:16 videos, use PIL stroke_width for thick black outline
                    outline_rgb = (
                        int(design.outline_color[2]),
                        int(design.outline_color[1]),
                        int(design.outline_color[0]),
                    )
                    # Use stroke_width for thick black outline (PIL 8.0+)
                    try:
                        pil_draw.text(
                            (x_cursor, baseline_y - line_ascent),
                            word,
                            font=pil_font,
                            fill=rgb_color,
                            stroke_width=design.outline_thickness,
                            stroke_fill=outline_rgb,
                        )
                    except TypeError:
                        # Fallback for older PIL versions - draw outline in 8 directions
                        outline_thickness = design.outline_thickness
                        offsets = [
                            (-outline_thickness, 0), (outline_thickness, 0),
                            (0, -outline_thickness), (0, outline_thickness),
                            (-outline_thickness, -outline_thickness),
                            (outline_thickness, outline_thickness),
                            (-outline_thickness, outline_thickness),
                            (outline_thickness, -outline_thickness),
                        ]
                        for adj_x, adj_y in offsets:
                            pil_draw.text(
                                (x_cursor + adj_x, baseline_y - line_ascent + adj_y),
                                word,
                                font=pil_font,
                                fill=outline_rgb,
                            )
                        # Draw main text
                        pil_draw.text(
                            (x_cursor, baseline_y - line_ascent),
                            word,
                            font=pil_font,
                            fill=rgb_color,
                        )
                else:
                    # For 4:5 videos - use exact old script logic (no outline, no opacity adjustments)
                    pil_draw.text(
                        (x_cursor, baseline_y - line_ascent),
                        word,
                        font=pil_font,
                        fill=rgb_color,
                    )
            else:
                # Fallback to cv2.putText
                if is_9_16:
                    # For 9:16, apply fade effects if needed
                    subtitle_opacity = 1.0
                    fade_duration = 0.3
                    if subtitle_segments and active_segment_index is not None:
                        seg_start, seg_end = subtitle_segments[active_segment_index]
                        start_t = transcript[seg_start]["start_time"]
                        end_t = transcript[seg_end]["end_time"]
                        if current_time < start_t + fade_duration:
                            subtitle_opacity = (current_time - start_t) / fade_duration
                            subtitle_opacity = max(0.0, min(1.0, subtitle_opacity))
                        elif current_time > end_t - fade_duration:
                            subtitle_opacity = (end_t - current_time) / fade_duration
                            subtitle_opacity = max(0.0, min(1.0, subtitle_opacity))
                    if subtitle_opacity < 1.0:
                        text_color = tuple(int(c * subtitle_opacity) for c in text_color)
                        outline_color = tuple(int(c * subtitle_opacity) for c in design.outline_color)
                    else:
                        outline_color = design.outline_color
                else:
                    # For 4:5 - use exact old script logic (no opacity adjustments)
                    outline_color = design.outline_color
                
                if design.outline_thickness > 0:
                    cv2.putText(
                        annotated,
                        word,
                        (x_cursor, baseline_y),
                        design.font,
                        design.text_scale,
                        outline_color,
                        thickness=design.outline_thickness,
                        lineType=cv2.LINE_AA,
                    )

                cv2.putText(
                    annotated,
                    word,
                    (x_cursor, baseline_y),
                    design.font,
                    design.text_scale,
                    text_color,
                    thickness=design.text_thickness,
                    lineType=cv2.LINE_AA,
                )
            x_cursor += word_width
        y_cursor = baseline_y + line_descent + line_spacing

    if pil_image is not None:
        annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return annotated


def process_video_with_overlays(
    main_video_path: str,
    transcript: List[Dict[str, float]],
    highlight_segments: List[Dict[str, Optional[object]]],
    subtitle_design: SubtitleDesign,
    output_path: str,
    subtitle_segments: Optional[List[Tuple[int, int]]] = None,
    custom_subtitles: Optional[List[str]] = None,
    aspect_ratio: str = "4:5",
) -> None:
    """Stream through the video, overlay clips, and draw subtitles."""
    step_start = time.time()

    logger.info("  [VIDEO PROCESSING] Opening video file...")
    cap = cv2.VideoCapture(main_video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open main video: {main_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = total_frames / fps if fps > 0 else 0

    target_aspect_ratio = parse_aspect_ratio(aspect_ratio)
    width, height = compute_cropped_dimensions(
        source_width, source_height, target_aspect_ratio
    )

    logger.info(
        f"  [VIDEO PROCESSING] Video info: {source_width}x{source_height} @ {fps:.2f}fps, {video_duration:.2f}s duration"
    )
    logger.info(
        f"  [VIDEO PROCESSING] Target size: {width}x{height} (aspect ratio: {aspect_ratio})"
    )
    logger.info(
        f"  [VIDEO PROCESSING] Total frames to process: {total_frames}"
    )

    segment_clip_paths: List[Optional[str]] = []
    clip_state: Dict[str, Dict[str, object]] = {}

    # First pass: Calculate segment durations (kept for possible future use / logging)
    processed_clips: Dict[str, str] = {}  # original clip path -> processed clip path
    temp_files: List[str] = []  # Track temp files for cleanup (will stay empty now)

    segment_durations: Dict[int, float] = {}
    for idx, segment in enumerate(highlight_segments):
        start_word = int(segment["start_word"])
        end_word = int(segment["end_word"])
        start_time = transcript[start_word]["start_time"]
        end_time = transcript[end_word]["end_time"]
        segment_durations[idx] = max(0.0, end_time - start_time)

    # Compute, per clip, the longest segment duration it needs to cover
    clip_target_durations: Dict[str, float] = {}
    for idx, segment in enumerate(highlight_segments):
        clip_path = segment.get("clip_path")
        if not clip_path:
            continue
        seg_dur = segment_durations.get(idx, 0.0)
        if seg_dur <= 0:
            continue
        prev = clip_target_durations.get(clip_path)
        if prev is None or seg_dur > prev:
            clip_target_durations[clip_path] = seg_dur

    # Second pass: associate each highlight with a clip path and prepare capture state.
    # Slow down any clip that is shorter than the longest segment that uses it.
    for idx, segment in enumerate(highlight_segments):
        clip_path = segment.get("clip_path")
        if not clip_path:
            segment_clip_paths.append(None)
            continue

        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Overlay clip not found: {clip_path}")

        if clip_path not in processed_clips:
            target_duration = clip_target_durations.get(clip_path, 0.0)

            # If we don't have a positive target duration, just reuse the clip as-is.
            if target_duration <= 0.0:
                logger.info(
                    f"  [CLIP] {clip_path}: target_duration={target_duration:.3f}s (<= 0) -> using clip as-is (no slowdown)."
                )
                processed_clips[clip_path] = clip_path
            else:
                # Measure original clip duration with OpenCV
                tmp_cap = cv2.VideoCapture(clip_path)
                clip_fps = tmp_cap.get(cv2.CAP_PROP_FPS) or fps
                clip_frame_count = tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                tmp_cap.release()

                if clip_fps <= 0 or clip_frame_count <= 0:
                    logger.info(
                        f"  [CLIP] {clip_path}: unable to measure duration "
                        f"(fps={clip_fps}, frames={clip_frame_count}) -> using clip as-is."
                    )
                    processed_clips[clip_path] = clip_path
                else:
                    clip_duration = float(clip_frame_count) / float(clip_fps)

                    # We want the clip to be at least as long as the longest
                    # segment that uses it, plus a tiny safety margin (~1 frame).
                    required_duration = target_duration + (1.0 / fps)

                    if clip_duration >= required_duration:
                        # Clip is already long enough; no slowdown necessary.
                        logger.info(
                            f"  [CLIP] {clip_path}: orig_dur={clip_duration:.3f}s, "
                            f"target_dur={target_duration:.3f}s, "
                            f"required={required_duration:.3f}s -> no slowdown."
                        )
                        processed_clips[clip_path] = clip_path
                    else:
                        # Slow down so that output duration ~= required_duration
                        speed_factor = clip_duration / required_duration
                        logger.info(
                            f"  [CLIP] {clip_path}: orig_dur={clip_duration:.3f}s, "
                            f"target_dur={target_duration:.3f}s, "
                            f"required={required_duration:.3f}s -> "
                            f"slowing down (speed_factor={speed_factor:.4f})."
                        )

                        tmp_file = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".mp4"
                        )
                        tmp_path = tmp_file.name
                        tmp_file.close()

                        # slow_down_video uses ffmpeg with setpts based on speed_factor
                        slow_down_video(clip_path, tmp_path, speed_factor)
                        processed_clips[clip_path] = tmp_path
                        temp_files.append(tmp_path)



        processed_clip_path = processed_clips[clip_path]
        segment_clip_paths.append(processed_clip_path)

        # Create and cache VideoCapture + metadata once per unique processed clip path.
        if processed_clip_path in clip_state:
            continue

        overlay_capture = cv2.VideoCapture(processed_clip_path)
        if not overlay_capture.isOpened():
            raise IOError(
                f"Cannot open overlay clip: {processed_clip_path}"
            )
        clip_total_frames = int(
            overlay_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )
        clip_state[processed_clip_path] = {
            "capture": overlay_capture,
            "total_frames": clip_total_frames,
            "fps": overlay_capture.get(cv2.CAP_PROP_FPS) or fps,
            "next_frame": 0,
            "current_segment_index": None,
            "current_subtitle_index": None,
            "last_segment_index": None,
            "last_subtitle_index": None,
            "continuation_pending": False,
            "needs_seek": True,
            "seek_frame": 0,
            "frames_to_drop": 0,
            "last_frame": None,
            "hold_last_frame": False,
            "last_capture_index": -1,
            "finished": clip_total_frames <= 0,
        }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot create output file: {output_path}")

    # Build highlight frame ranges and map to subtitles (if any)
    highlight_frame_ranges: List[List[int]] = []
    highlight_subtitle_indices: List[Optional[int]] = []

    for idx, segment in enumerate(highlight_segments):
        start_word = int(segment["start_word"])
        end_word = int(segment["end_word"])
        start_time = transcript[start_word]["start_time"]
        end_time = transcript[end_word]["end_time"]

        # Compute a slightly extended end time so the overlay comfortably
        # covers the last word, without bleeding into the next spoken word.
        phrase_end_time = transcript[end_word]["end_time"]

        # How much earlier than the first word we want the overlay to appear
        # (so we don't see the first syllable on the main shot).
        lead_margin = 0.15      # try 0.15s; tweak 0.12–0.18 if needed
        gap_threshold = 0.15    # defines "this is a main→overlay gap"

        # Small global tail margin in seconds (tweakable)
        tail_margin = 0.12  # try 0.12; bump more if required
        # Default: no early shift
        start_time_early = start_time

        if idx > 0:
            prev_seg = highlight_segments[idx - 1]
            prev_end_word = int(prev_seg["end_word"])
            prev_end_time = transcript[prev_end_word]["end_time"]

            gap = start_time - prev_end_time

            if gap > gap_threshold:
                # MAIN → OVERLAY transition.
                #
                # Start slightly before the phrase, but never before the previous
                # highlight actually ended. This keeps previous overlay intact,
                # and pulls the new overlay into the last part of the "main" gap.
                tentative_early = start_time - lead_margin
                start_time_early = max(prev_end_time, tentative_early)
                logger.info(
                "[HFR_SETUP] seg=%d MAIN→OVERLAY gap=%.3fs "
                "start=%.3fs early=%.3fs prev_end=%.3fs",
                idx,
                gap,
                start_time,
                start_time_early,
                prev_end_time,
                )
            else:
                # OVERLAY → OVERLAY transition. Don't shift earlier here.
                start_time_early = start_time
                logger.info(
                "[HFR_SETUP] seg=%d OVERLAY→OVERLAY gap=%.3fs "
                "start=%.3fs (no early shift)",
                idx,
                gap,
                start_time,
                )
        else:
            # First overlay: safe to pull slightly earlier
            start_time_early = max(0.0, start_time - lead_margin)


        # Start by adding the tail margin
        extended_end_time = phrase_end_time + tail_margin

        # If there is a next word, do not extend past its start minus
        # one frame, so we never cover the next word.
        if end_word + 1 < len(transcript):
            next_start_time = transcript[end_word + 1]["start_time"]
            guard = 1.0 / fps  # leave at least one frame for the next word
            max_safe_end_time = max(
                phrase_end_time,  # never earlier than the phrase end
                min(extended_end_time, next_start_time - guard),
            )
        else:
            max_safe_end_time = extended_end_time

        # Convert to frames
        #   start_frame: floor so we never start late
        #   end_frame:   ceil - 1 so we fully cover up to max_safe_end_time
        start_frame = int(math.floor(start_time_early * fps))
        end_frame = int(math.ceil(max_safe_end_time * fps)) - 1
        if end_frame < start_frame:
            end_frame = start_frame

        highlight_frame_ranges.append([start_frame, end_frame, idx])

        # Log final timing decision for this segment
        logger.info(
        "[HFR_SETUP] seg=%d words[%d:%d] "
        "start=%.3fs early=%.3fs end=%.3fs -> frames=[%d,%d]",
        idx,
        start_word,
        end_word,
        start_time,
        start_time_early,
        max_safe_end_time,
        start_frame,
        end_frame,
        )


        # Track which subtitle block this phrase sits in (for subtitles logic)
        subtitle_index: Optional[int] = None
        if subtitle_segments:
            for sub_idx, (sub_start, sub_end) in enumerate(subtitle_segments):
                if sub_end < start_word:
                    continue
                if sub_start > end_word:
                    break
                subtitle_index = sub_idx
                break

        highlight_subtitle_indices.append(subtitle_index)

    # ─────────────────────────────────────────────────────────────────────────
    # Enforce time ordering & non-overlap between highlight ranges
    # This prevents a later overlay from grabbing frames
    # that still belong to the previous phrase.
    # ─────────────────────────────────────────────────────────────────────────
    highlight_frame_ranges.sort(key=lambda r: r[0])

    for i in range(1, len(highlight_frame_ranges)):
        prev = highlight_frame_ranges[i - 1]
        curr = highlight_frame_ranges[i]
        prev_start, prev_end, _ = prev
        curr_start, curr_end, _ = curr

        if curr_start <= prev_end:
            # Push the start of this range to *after* the previous one ends
            new_start = prev_end + 1
            if new_start > curr_end:
                # Degenerate case: squash to a single frame if needed
                new_start = curr_end
            curr[0] = new_start
    # ─────────────────────────────────────────────────────────────────────────
    # Close small gaps between consecutive overlays so the main video does not
    # flash briefly between them.
    #
    # If the gap between [prev_end] and [next_start] is <= max_gap_seconds,
    # we extend the PREVIOUS overlay to cover up to next_start - 1.
    # ─────────────────────────────────────────────────────────────────────────
    max_gap_seconds = 0.75  # tunable: up to ~0.75s of gap gets "bridged"
    max_gap_frames = max(1, int(round(max_gap_seconds * fps)))

    for i in range(1, len(highlight_frame_ranges)):
        prev = highlight_frame_ranges[i - 1]
        curr = highlight_frame_ranges[i]
        prev_end = prev[1]
        curr_start = curr[0]

        gap = curr_start - prev_end - 1  # frames strictly between prev_end and curr_start
        if gap > 0 and gap <= max_gap_frames:
            # Extend previous overlay to fill the gap
            prev[1] = curr_start - 1
    # ─────────────────────────────────────────────────────────────────────────
    # If the last overlay ends just a short time before the end of the video,
    # extend it all the way to the final frame so the video ends on the overlay
    # instead of flashing back to the main video.
    # ─────────────────────────────────────────────────────────────────────────
    if highlight_frame_ranges:
        last_frame_index = total_frames - 1
        last_start, last_end, last_seg_idx = highlight_frame_ranges[-1]

        # How much "tail" (in frames) is left after the last overlay?
        tail_gap = last_frame_index - last_end

        # Only extend if the leftover tail is small (e.g. <= 0.75s)
        max_tail_to_video_seconds = 0.75
        max_tail_to_video_frames = max(
            1, int(round(max_tail_to_video_seconds * fps))
        )

        if tail_gap > 0 and tail_gap <= max_tail_to_video_frames:
            highlight_frame_ranges[-1][1] = last_frame_index
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tiny "tail" extension for overlay → main transitions.
    #
    # We extend each overlay segment by a small fixed number of frames
    # (e.g. 1 frame ≈ 40ms at 25fps), but:
    #   - never beyond the first frame of the NEXT overlay, and
    #   - never beyond the last frame of the video.
    #
    # This lets the final phonemes of the last word (like the "er" in "danger")
    # still be visually covered by the overlay instead of flashing back to main.
    # ─────────────────────────────────────────────────────────────────────────
    overlay_tail_frames = 1  # 1 frame ≈ 0.04s at 25fps

    if overlay_tail_frames > 0 and total_frames > 0:
        last_frame_index = total_frames - 1
        for i, rng in enumerate(highlight_frame_ranges):
            start_f, end_f, seg_idx = rng

            # Where does the next overlay start?
            if i + 1 < len(highlight_frame_ranges):
                next_start = highlight_frame_ranges[i + 1][0]
            else:
                # No next overlay: treat video end as the next boundary
                next_start = total_frames

            # We can extend up to:
            #   - end_f + overlay_tail_frames
            #   - but strictly before next_start (so we don't collide with next overlay)
            #   - and not beyond last_frame_index
            max_end = min(end_f + overlay_tail_frames, next_start - 1, last_frame_index)

            if max_end > end_f:
                rng[1] = max_end



    # DEBUG: log all highlight ranges after non-overlap adjustment
    logger.info("  [HFR] Highlight frame ranges (after adjust):")
    for start_f, end_f, seg_idx in highlight_frame_ranges:
        seg = highlight_segments[seg_idx]
        sw = int(seg["start_word"])
        ew = int(seg["end_word"])
        st = transcript[sw]["start_time"]
        et = transcript[ew]["end_time"]
        words = " ".join(
            entry["word"] for entry in transcript[sw : ew + 1]
        )
        clip = seg.get("clip_path")
        logger.info(
            "  [HFR] seg=%d clip=%s frames=%d-%d time=%.3f-%.3fs words=%s",
            seg_idx,
            clip,
            start_f,
            end_f,
            st,
            et,
            words,
        )


    frame_index = 0
    highlight_ranges_for_words = [
        (seg["start_word"], seg["end_word"]) for seg in highlight_segments
    ]
    prev_active_overlay_index: Optional[int] = None
    logger.info("  [VIDEO PROCESSING] Starting frame processing loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect_ratio(frame, target_aspect_ratio)

        # Use actual video position for accurate timing when available
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec >= 0:
            current_time = pos_msec / 1000.0
        else:
            current_time = frame_index / fps

        active_overlay_index: Optional[int] = None
        segment_start_frame: Optional[int] = None
        for start_f, end_f, seg_idx in highlight_frame_ranges:
            if start_f <= frame_index <= end_f:
                active_overlay_index = seg_idx
                segment_start_frame = start_f
                break
        
        # DEBUG: log overlay activation / deactivation edges
        if active_overlay_index != prev_active_overlay_index:
            if active_overlay_index is None:
                logger.info(
                    "  [OVERLAY] frame=%d time=%.3fs -> NO OVERLAY (prev seg=%s)",
                    frame_index,
                    current_time,
                    str(prev_active_overlay_index),
                )
            else:
                seg = highlight_segments[active_overlay_index]
                sw = int(seg["start_word"])
                ew = int(seg["end_word"])
                st = transcript[sw]["start_time"]
                et = transcript[ew]["end_time"]
                words = " ".join(
                    entry["word"] for entry in transcript[sw : ew + 1]
                )
                clip = segment_clip_paths[active_overlay_index]
                logger.info(
                    "  [OVERLAY] frame=%d time=%.3fs -> seg=%d clip=%s "
                    "words[%d:%d](%.3f-%.3fs): %s",
                    frame_index,
                    current_time,
                    active_overlay_index,
                    clip,
                    sw,
                    ew,
                    st,
                    et,
                    words,
                )
            prev_active_overlay_index = active_overlay_index


        if active_overlay_index is not None:
            clip_path = segment_clip_paths[active_overlay_index]
            if clip_path:
                clip_info = clip_state.get(clip_path)
                if clip_info is not None:
                    overlay_cap = clip_info["capture"]
                    if overlay_cap is not None and not clip_info.get(
                        "finished", False
                    ):
                        current_subtitle_index = highlight_subtitle_indices[
                            active_overlay_index
                        ]
                        current_segment_index = clip_info.get(
                            "current_segment_index"
                        )

                        if current_segment_index != active_overlay_index:
                            # We are switching to a new segment for this clip
                            if current_segment_index is not None:
                                clip_info["last_segment_index"] = (
                                    current_segment_index
                                )
                            if clip_info.get("current_subtitle_index") is not None:
                                clip_info["last_subtitle_index"] = clip_info[
                                    "current_subtitle_index"
                                ]

                            prev_segment_index = clip_info.get("last_segment_index")
                            prev_subtitle_index = clip_info.get("last_subtitle_index")

                            if subtitle_segments:
                                should_continue = (
                                    prev_subtitle_index is not None
                                    and current_subtitle_index is not None
                                    and current_subtitle_index
                                    == prev_subtitle_index + 1
                                )
                            else:
                                should_continue = (
                                    prev_segment_index is not None
                                    and active_overlay_index
                                    == prev_segment_index + 1
                                )

                            if not should_continue:
                                # NEW segment with this clip: sync clip start with segment start.
                                if segment_start_frame is None:
                                    frames_into_segment = 0
                                else:
                                    frames_into_segment = max(
                                        0, frame_index - segment_start_frame
                                    )

                                clip_info["next_frame"] = frames_into_segment
                                clip_info["seek_frame"] = frames_into_segment
                                clip_info["needs_seek"] = True
                                clip_info["continuation_pending"] = False
                                clip_info["frames_to_drop"] = 0
                                clip_info["last_frame"] = None
                                clip_info["hold_last_frame"] = False
                                clip_info["last_capture_index"] = -1
                            else:
                                target_next = max(
                                    int(clip_info.get("next_frame", 0)), 0
                                )
                                clip_info["seek_frame"] = target_next
                                clip_info["needs_seek"] = True
                                clip_info["continuation_pending"] = True
                                clip_info["frames_to_drop"] = 0
                                clip_info["hold_last_frame"] = False

                            clip_info["finished"] = (
                                clip_info["total_frames"] <= 0
                            )
                            clip_info["current_segment_index"] = (
                                active_overlay_index
                            )
                            clip_info["current_subtitle_index"] = (
                                current_subtitle_index
                            )
                        else:
                            clip_info["current_subtitle_index"] = (
                                current_subtitle_index
                            )

                        overlay_total_frames = clip_info["total_frames"]
                        current_index = int(clip_info.get("next_frame", 0))
                        frame_to_overlay: Optional[np.ndarray] = None

                        if overlay_total_frames <= 0:
                            frame_to_overlay = clip_info.get("last_frame")
                        else:
                            last_capture_index = clip_info.get(
                                "last_capture_index", -1
                            )
                            expected_next = (
                                last_capture_index + 1
                                if last_capture_index != -1
                                and last_capture_index
                                < overlay_total_frames - 1
                                else current_index
                            )

                            if clip_info.get(
                                "needs_seek", False
                            ) or current_index != expected_next:
                                overlay_cap.set(
                                    cv2.CAP_PROP_POS_FRAMES, current_index
                                )
                                clip_info["needs_seek"] = False

                            ret_o, overlay_frame = overlay_cap.read()
                            if not ret_o:
                                clip_info["finished"] = True
                                overlay_frame = clip_info.get("last_frame")

                            if overlay_frame is not None:
                                clip_info["last_frame"] = overlay_frame
                                clip_info["last_capture_index"] = current_index
                                frame_to_overlay = overlay_frame
                                clip_info["next_frame"] = current_index + 1
                            else:
                                frame_to_overlay = clip_info.get("last_frame")

                        if frame_to_overlay is not None:
                            # Ensure the overlay uses the *same* crop and canvas
                            # size as the main video so they line up perfectly.
                            # 1) Crop the B-roll to the target aspect ratio.
                            overlay_cropped = crop_to_aspect_ratio(
                                frame_to_overlay, target_aspect_ratio
                            )

                            oh, ow = overlay_cropped.shape[:2]
                            if oh > 0 and ow > 0:
                                h, w = frame.shape[:2]

                                # 2) Resize the cropped overlay to exactly fill
                                #    the current frame. Because both `frame` and
                                #    `overlay_cropped` share the same aspect
                                #    ratio, this does *not* distort the image.
                            if ow != w or oh != h:
                                resized_overlay = cv2.resize(
                                    overlay_cropped, (w, h)
                                )
                            else:
                                resized_overlay = overlay_cropped

                            # 3) Replace the entire frame with the overlay so
                            #    the original and B-roll align pixel-perfectly.
                            frame[:, :] = resized_overlay


        # Draw subtitles on this frame
        frame_with_subtitles = draw_subtitle_on_frame(
            frame,
            transcript,
            current_time,
            subtitle_design,
            highlight_ranges_for_words,
            subtitle_segments=subtitle_segments,
            custom_subtitles=custom_subtitles,
        )
        writer.write(frame_with_subtitles)
        frame_index += 1

    cap.release()
    for clip_info in clip_state.values():
        overlay_cap = clip_info.get("capture")
        if overlay_cap is not None:
            overlay_cap.release()
    writer.release()

    # Clean up any temporary slowed-down clip files (should be empty now)
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"  [CLEANUP] Removed temporary file: {temp_file}")
        except Exception as e:
            logger.warning(
                f"  [CLEANUP] Could not remove temporary file {temp_file}: {e}"
            )

    step_duration = time.time() - step_start
    logger.info(
        f"  [VIDEO PROCESSING] ✓ Completed processing {frame_index} frames in {step_duration:.2f}s ({step_duration/60:.2f} min)"
    )
    if step_duration > 0:
        logger.info(
            f"  [VIDEO PROCESSING] Average processing speed: {frame_index/step_duration:.2f} fps"
        )

def merge_audio_tracks(
    silent_video_path: str,
    main_video_path: str,
    transcript: List[Dict[str, float]],
    highlight_segments: List[Dict[str, Optional[object]]],
    final_output_path: str,
    preserve_main_audio: bool = True,
    global_music_path: Optional[str] = None,
    global_music_volume: float = 1.0,
) -> None:
    """Attach the original audio, per-segment music, and optional global music using ffmpeg."""
    step_start = time.time()
    logger.info("  [AUDIO MERGE] Starting audio merge process...")

    # Get video duration and FPS from silent video
    logger.info("  [AUDIO MERGE] Reading video metadata...")
    cap = cv2.VideoCapture(silent_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open silent video: {silent_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    video_duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    if video_duration <= 0:
        # Fallback: try to get duration using ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    silent_video_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            video_duration = float(result.stdout.strip())
        except Exception:
            video_duration = 0
    
    if video_duration <= 0:
        raise ValueError(f"Could not determine video duration for {silent_video_path}")
    
    logger.info(f"  [AUDIO MERGE] Video duration: {video_duration:.2f}s")

    # Build list of audio inputs for ffmpeg
    # Input 0 is the silent video, audio inputs start at 1
    audio_input_files = []
    filter_complex_parts = []
    audio_labels = []
    input_idx = 1  # Start at 1 because 0 is the silent video
    
    # 1. Extract and add main video audio if needed
    logger.info("  [AUDIO MERGE] Processing audio tracks...")
    if preserve_main_audio:
        logger.info("  [AUDIO MERGE] - Adding main video audio track...")
        try:
            # Check if main video has audio
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    main_video_path
                ],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip() == "audio":
                audio_input_files.append(main_video_path)
                audio_labels.append("main_audio")
                # Trim main audio to video duration
                filter_complex_parts.append(
                    f"[{input_idx}:a]atrim=0:{video_duration},asetpts=PTS-STARTPTS,volume=1.0[main_audio]"
                )
                input_idx += 1
        except Exception as exc:
            print(f"[warn] Unable to extract audio from main video ({exc}).")

    # 2. Add global music if provided
    if global_music_path:
        logger.info(f"  [AUDIO MERGE] - Adding global music: {global_music_path}")
        if not os.path.exists(global_music_path):
            raise FileNotFoundError(f"Global music file not found: {global_music_path}")
        
        audio_input_files.append(global_music_path)
        current_input_idx = input_idx
        input_idx += 1
        
        # Get global music duration
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    global_music_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            music_duration = float(result.stdout.strip())
        except Exception:
            music_duration = video_duration
        
        # Loop music if needed
        if music_duration < video_duration:
            loops = math.ceil(video_duration / music_duration)
            # Use concat filter to loop
            concat_inputs = ",".join([f"[{current_input_idx}:a]"] * loops)
            filter_complex_parts.append(
                f"{concat_inputs}concat=n={loops}:v=0:a=1[global_looped]"
            )
            global_input = "global_looped"
        else:
            global_input = f"{current_input_idx}:a"
        
        # Trim to video duration and apply volume
        volume = float(global_music_volume)
        filter_complex_parts.append(
            f"[{global_input}]atrim=0:{video_duration},asetpts=PTS-STARTPTS,volume={volume}[global_music]"
        )
        audio_labels.append("global_music")

    # 3. Add per-segment music
    logger.info(f"  [AUDIO MERGE] - Processing {len(highlight_segments)} highlight segments for music...")
    segment_audio_labels = []
    for idx, segment in enumerate(highlight_segments):
        music_path = segment.get("music_path")
        if not music_path:
            continue
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Music file not found: {music_path}")
        
        start_word = int(segment["start_word"])
        end_word = int(segment["end_word"])
        start_time = transcript[start_word]["start_time"]
        end_time = transcript[end_word]["end_time"]
        duration = max(end_time - start_time, 0.0)
        if duration <= 0:
            continue
        
        audio_input_files.append(music_path)
        current_input_idx = input_idx
        input_idx += 1
        label = f"segment_{idx}"
        segment_audio_labels.append(label)
        
        # Get music duration
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    music_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            music_duration = float(result.stdout.strip())
        except Exception:
            music_duration = duration
        
        # Loop music if needed
        if music_duration < duration:
            loops = math.ceil(duration / music_duration)
            concat_inputs = ",".join([f"[{current_input_idx}:a]"] * loops)
            filter_complex_parts.append(
                f"{concat_inputs}concat=n={loops}:v=0:a=1[{label}_looped]"
            )
            segment_input = f"{label}_looped"
        else:
            segment_input = f"{current_input_idx}:a"
        
        # Trim to segment duration, apply volume, and delay
        volume = float(segment.get("music_volume", 1.0))
        delay_ms = int(start_time * 1000)
        # adelay works for both mono and stereo: delay_ms|delay_ms for stereo, or just delay_ms for mono
        filter_complex_parts.append(
            f"[{segment_input}]atrim=0:{duration},asetpts=PTS-STARTPTS,"
            f"volume={volume},adelay={delay_ms}|{delay_ms}[{label}]"
        )

    # 4. Mix all audio tracks together
    if not audio_input_files:
        # No audio to add, just copy video
        logger.info("  [AUDIO MERGE] No audio tracks to merge, copying video only...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", silent_video_path,
                "-c:v", "copy", "-an", final_output_path
            ],
            check=True,
            capture_output=True
        )
        logger.info(f"  [AUDIO MERGE] ✓ Video copied in {time.time() - step_start:.2f}s")
        return
    
    logger.info(f"  [AUDIO MERGE] Total audio tracks to mix: {len(audio_input_files)}")

    # Build filter complex to mix all audio
    all_audio_labels = []
    if "main_audio" in audio_labels:
        all_audio_labels.append("main_audio")
    if "global_music" in audio_labels:
        all_audio_labels.append("global_music")
    all_audio_labels.extend(segment_audio_labels)
    
    if len(all_audio_labels) == 1:
        # Only one audio track, no mixing needed
        mix_filter = all_audio_labels[0]
    else:
        # Mix multiple audio tracks
        mix_inputs = "".join([f"[{label}]" for label in all_audio_labels])
        filter_complex_parts.append(
            f"{mix_inputs}amix=inputs={len(all_audio_labels)}:duration=longest:dropout_transition=0[mixed_audio]"
        )
        mix_filter = "mixed_audio"
    
    # 5. Combine video with mixed audio
    filter_complex = ";".join(filter_complex_parts)
    
    logger.info("  [AUDIO MERGE] Building ffmpeg command...")
    logger.info(f"  [AUDIO MERGE] Filter complex length: {len(filter_complex)} characters")
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", silent_video_path]
    
    # Add all audio input files
    for audio_file in audio_input_files:
        cmd.extend(["-i", audio_file])
    
    # Add filter complex and output options
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", f"[{mix_filter}]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output_path
    ])
    
    logger.info("  [AUDIO MERGE] Running ffmpeg to merge audio...")
    merge_start = time.time()
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )
    merge_duration = time.time() - merge_start
    logger.info(f"  [AUDIO MERGE] ✓ ffmpeg completed in {merge_duration:.2f}s")
    
    if result.returncode != 0:
        logger.error(f"  [AUDIO MERGE] ✗ ffmpeg failed with return code {result.returncode}")
        logger.error(f"  [AUDIO MERGE] Error output: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    total_duration = time.time() - step_start
    logger.info(f"  [AUDIO MERGE] ✓ Audio merge completed in {total_duration:.2f}s")


# --------------------------------------------------------------------------- #
# High level orchestration
# --------------------------------------------------------------------------- #


def render_project(config: ProjectConfig) -> Dict[str, object]:
    """Run the full pipeline and return metadata for inspection."""
    pipeline_start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING VIDEO RENDERING PIPELINE")
    logger.info(f"Input video: {config.main_video_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Aspect ratio: {config.aspect_ratio or '4:5'}")
    logger.info(f"Highlights: {len(config.highlight_assignments)}")
    logger.info("=" * 80)

    # Step 1: Build transcript
    step_start = time.time()
    logger.info("[STEP 1/4] Building transcript...")
    transcript = build_transcript(
        config.main_video_path,
        transcript_text=config.transcript_text,
        whisper_model=config.whisper_model,
    )
    step_duration = time.time() - step_start
    logger.info(f"[STEP 1/4] ✓ Transcript built in {step_duration:.2f}s ({len(transcript)} words)")

    # Step 2: Map highlights
    step_start = time.time()
    logger.info("[STEP 2/4] Mapping highlight segments...")
    highlight_segments = map_assignments_to_segments(
        transcript, config.highlight_assignments
    )
    step_duration = time.time() - step_start
    logger.info(f"[STEP 2/4] ✓ Highlight segments mapped in {step_duration:.2f}s ({len(highlight_segments)} segments)")

    any_segment_music = any(
        assignment.music_path for assignment in config.highlight_assignments
    )
    needs_audio_merge =  (
        config.preserve_audio or bool(config.global_music_path) or any_segment_music
    )
    final_output_path = config.output_path
    silent_output_path = final_output_path

    subtitle_segments = config.subtitle_segments
    custom_subtitle_texts: Optional[List[str]] = None

    if config.subtitle_sentences:
        mapped_sentences = map_subtitle_sentences(
            transcript, config.subtitle_sentences
        )
        subtitle_segments = [
            (entry["start_word"], entry["end_word"]) for entry in mapped_sentences
        ]
        custom_subtitle_texts = [entry["text"] for entry in mapped_sentences]
    if subtitle_segments is None:
        subtitle_segments = generate_default_subtitle_segments(
            transcript, highlight_segments
        )

    if needs_audio_merge:
        root, ext = os.path.splitext(final_output_path)
        ext = ext or ".mp4"
        silent_output_path = f"{root}.silent{ext}"

    # Get aspect ratio (default to 4:5 if not specified)
    aspect_ratio = config.aspect_ratio or "4:5"
    subtitle_design = config.subtitle_design

    # Step 3: Process video with overlays
    step_start = time.time()
    logger.info("[STEP 3/4] Processing video with overlays and subtitles...")
    logger.info(f"  - Aspect ratio: {aspect_ratio}")
    logger.info(f"  - Subtitle segments: {len(subtitle_segments)}")
    logger.info(f"  - Highlight segments: {len(highlight_segments)}")
    process_video_with_overlays(
        config.main_video_path,
        transcript,
        highlight_segments,
        subtitle_design,
        silent_output_path,
        subtitle_segments=subtitle_segments,
        custom_subtitles=custom_subtitle_texts,
        aspect_ratio=aspect_ratio,
    )
    step_duration = time.time() - step_start
    logger.info(f"[STEP 3/4] ✓ Video processed in {step_duration:.2f}s")

    # Step 4: Merge audio tracks
    if needs_audio_merge:
        step_start = time.time()
        logger.info("[STEP 4/4] Merging audio tracks...")
        logger.info(f"  - Preserve main audio: {config.preserve_audio}")
        logger.info(f"  - Global music: {bool(config.global_music_path)}")
        logger.info(f"  - Segment music: {any_segment_music}")
        merge_audio_tracks(
            silent_output_path,
            config.main_video_path,
            transcript,
            highlight_segments,
            final_output_path,
            preserve_main_audio=config.preserve_audio,
            global_music_path=config.global_music_path,
            global_music_volume=config.global_music_volume,
        )
        step_duration = time.time() - step_start
        logger.info(f"[STEP 4/4] ✓ Audio merged in {step_duration:.2f}s")
        
        if (
            os.path.exists(silent_output_path)
            and silent_output_path != final_output_path
        ):
            os.remove(silent_output_path)
            logger.info(f"  - Cleaned up temporary silent video: {silent_output_path}")
    else:
        logger.info("[STEP 4/4] Skipping audio merge (no audio tracks to merge)")

    total_duration = time.time() - pipeline_start_time
    logger.info("=" * 80)
    logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY in {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    logger.info(f"Output file: {final_output_path}")
    logger.info("=" * 80)

    return {
        "transcript": transcript,
        "highlight_segments": highlight_segments,
        "output_path": final_output_path,
        "subtitle_segments": subtitle_segments,
        "custom_subtitles": custom_subtitle_texts,
    }


# --------------------------------------------------------------------------- #
# Configuration parsing helpers
# --------------------------------------------------------------------------- #


def load_project_config_from_json(
    path: str, base_config: ProjectConfig
) -> ProjectConfig:
    """Populate ``ProjectConfig`` from a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    highlight_items = data.get("highlight_assignments", data.get("highlights", []))
    assignments: List[HighlightAssignment] = []
    
    for item in highlight_items:
        assignments.append(
            HighlightAssignment(
                phrase=item.get("phrase"),
                clip_path=item.get("clip_path"),
                music_path=item.get("music_path"),
                music_volume=float(item.get("music_volume", 1.0)),
                occurrence=int(item.get("occurrence", 1)),
                start_word=item.get("start_word"),
                end_word=item.get("end_word"),
            )
        )
    if assignments:
        base_config.highlight_assignments = assignments

    if "transcript_text" in data:
        base_config.transcript_text = data["transcript_text"]

    if "subtitle_design" in data:
        design_data = data["subtitle_design"]
        kwargs = {}
        for field_name in (
            "bar_color",
            "bar_opacity",
            "text_color",
            "text_scale",
            "text_thickness",
            "outline_color",
            "outline_thickness",
            "highlight_color",
            "highlight_text_color",
            "margin",
            "margin_x",
            "margin_y",
            "bottom_margin",
            "max_line_width_ratio",
            "line_spacing",
            "corner_radius",
            "box_shadow_offset",
            "box_shadow_blur",
            "box_shadow_alpha",
            "shadow_color",
            "shadow_offset",
            "shadow_thickness",
            "highlight_padding",
            "font_path",
            "font_size_px",
        ):
            if field_name in design_data:
                value = design_data[field_name]
                if isinstance(value, list):
                    value = tuple(value)
                kwargs[field_name] = value
        base_config.subtitle_design = SubtitleDesign(**kwargs)

    if "preserve_audio" in data:
        base_config.preserve_audio = bool(data["preserve_audio"])

    if "global_music_path" in data:
        base_config.global_music_path = data["global_music_path"]
    if "global_music_volume" in data:
        base_config.global_music_volume = float(data["global_music_volume"])

    if "subtitle_segments" in data:
        base_config.subtitle_segments = [
            tuple(seg) for seg in data["subtitle_segments"]
        ]

    if "subtitle_sentences" in data:
        sentences_config = data["subtitle_sentences"]
        sentences: List[SubtitleSentence] = []
        if isinstance(sentences_config, list):
            for item in sentences_config:
                if isinstance(item, str):
                    text_value = item.strip()
                    if text_value:
                        sentences.append(
                            SubtitleSentence(text=text_value, phrase=text_value)
                        )
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("display_text") or item.get("phrase")
                    if not text_value:
                        continue
                    sentences.append(
                        SubtitleSentence(
                            text=text_value,
                            phrase=item.get("phrase", text_value),
                            occurrence=int(item.get("occurrence", 1)),
                            start_word=item.get("start_word"),
                            end_word=item.get("end_word"),
                        )
                    )
        if sentences:
            base_config.subtitle_sentences = sentences

    return base_config


# --------------------------------------------------------------------------- #
# Demo / CLI entry point
# --------------------------------------------------------------------------- #


def run_demo(output_path: str = "demo_output.mp4") -> None:
    """Generate a dummy project for quick smoke testing."""

    base_video_path = "demo_base.mp4"
    overlay_clip_path = "demo_overlay.mp4"

    def create_dummy_video(
        path: str,
        duration: float = 6.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (720, 1280),
    ) -> None:
        h, w = resolution
        total_frames = int(duration * fps)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for idx in range(total_frames):
            hue = int((idx / total_frames) * 180) % 180
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = hue
            hsv[..., 1] = 200
            hsv[..., 2] = 220
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.putText(
                frame,
                f"Frame {idx}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()

    def create_overlay_clip(
        path: str,
        duration: float = 2.5,
        fps: int = 30,
        resolution: Tuple[int, int] = (960, 768),
    ) -> None:
        h, w = resolution
        total_frames = int(duration * fps)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for idx in range(total_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            radius = 120
            center_x = w // 2
            center_y = int(
                h * (0.3 + 0.4 * abs(math.sin(math.pi * idx / total_frames)))
            )
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 180), -1)
            cv2.putText(
                frame,
                "Overlay",
                (center_x - 180, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (30, 30, 30),
                3,
                cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()

    create_dummy_video(base_video_path, resolution=(720, 1280))
    create_overlay_clip(overlay_clip_path, resolution=(960, 768))

    demo_text = "We always enjoyed ourselves and did everything together"
    assignments = [
        HighlightAssignment(
            phrase="enjoyed ourselves and",
            clip_path=overlay_clip_path,
            music_path=None,
        )
    ]

    config = ProjectConfig(
        main_video_path=base_video_path,
        output_path=output_path,
        transcript_text=demo_text,
        highlight_assignments=assignments,
        preserve_audio=False,
    )

    render_project(config)
    print(f"[demo] Demo render finished. Output written to {output_path}")


def parse_cli_args() -> argparse.Namespace:
    """Configure and parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Overlay highlight clips and subtitles on a video."
    )

    parser.add_argument("--main-video", help="Path to the main video file.")
    parser.add_argument(
        "--output", default="output.mp4", help="Destination for the rendered video."
    )
    parser.add_argument(
        "--config",
        help="JSON file describing highlight assignments and optional design overrides.",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a self contained demo showcasing the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    if args.demo:
        run_demo()
        return

    if not args.main_video:
        raise SystemExit("Please provide --main-video or use --demo.")

    config = ProjectConfig(
        main_video_path=args.main_video,
        output_path=args.output,
    )

    if args.config:
        config = load_project_config_from_json(args.config, config)

    render_project(config)
    print(
        f"[info] Render completed successfully. Output written to {config.output_path}"
    )


if __name__ == "__main__":
    main()
