from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import moviepy as mpy

    HAVE_MOVIEPY = True
except ImportError:
    HAVE_MOVIEPY = False

try:
    import whisper

    HAVE_WHISPER = True
except ImportError:
    HAVE_WHISPER = False

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


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def normalise_word(token: str) -> str:
    """Lower-case alphanumeric tokenisation for fuzzy matching."""

    return "".join(ch for ch in token.lower() if ch.isalnum())


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


def write_subtitle_into_file(
    input_file_name: str, transcript: List[Dict[str, float]]
):
    filename, _ = os.path.splitext(input_file_name)
    file_name = f"{filename}_subtitle.json"

    with open(file_name, "w") as f:
        json.dump(transcript, f, indent=4)

    print(f"Saved the video file transcription in {file_name}")


def build_transcript(
    video_path: str,
    transcript_text: Optional[str],
    whisper_model: str,
) -> List[Dict[str, float]]:
    """Create a per-word transcript using Whisper timestamps only."""

    try:
        transcript = transcribe_audio_whisper(video_path, whisper_model)
        if transcript:
            write_subtitle_into_file(video_path, transcript)
            return transcript
    except Exception as exc:  # noqa: BLE001 - Whisper issues should surface as warnings
        raise RuntimeError(
            "Whisper transcription failed. Ensure openai-whisper is installed and the model is available."
        ) from exc

    raise RuntimeError("Whisper returned an empty transcript; cannot proceed.")


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
    min_similarity: float = 0.75,
) -> Optional[Tuple[int, int]]:
    """Find an approximate match allowing small spelling/word-count deviations."""

    token_list = list(tokens)
    if not token_list or start_index >= len(transcript_words):
        return None

    min_window = max(1, len(token_list) - max_window_delta)
    max_window = max(min_window, len(token_list) + max_window_delta)
    phrase_str = " ".join(token_list)
    matches: List[Tuple[int, int, float]] = []

    for idx in range(start_index, len(transcript_words) - min_window + 1):
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

    if len(matches) < occurrence:
        return None

    start, end, _ratio = matches[occurrence - 1]
    return start, end


def find_phrase_indices(
    transcript_words: Sequence[str],
    phrase: str,
    occurrence: int = 1,
    start_index: int = 0,
) -> Tuple[int, int]:
    """Locate ``phrase`` within ``transcript_words`` returning (start, end) indices.

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
    """Convert user highlight selections into rendering segments."""

    if not transcript:
        return []

    normalised_transcript = [normalise_word(entry["word"]) for entry in transcript]
    mapped: List[Dict[str, Optional[object]]] = []

    for assignment in assignments:
        start_word = assignment.start_word
        end_word = assignment.end_word

        if start_word is None or end_word is None:
            start_word, end_word = find_phrase_indices(
                normalised_transcript,
                assignment.phrase or "",
                occurrence=assignment.occurrence,
            )
        else:
            start_word = int(start_word)
            end_word = int(end_word)

        if start_word < 0 or end_word >= len(transcript) or start_word > end_word:
            raise ValueError(
                f"Invalid word indices resolved for assignment {assignment}"
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
                pil_draw.text(
                    (x_cursor, baseline_y - line_ascent),
                    word,
                    font=pil_font,
                    fill=rgb_color,
                )
            else:
                if design.outline_thickness > 0:
                    cv2.putText(
                        annotated,
                        word,
                        (x_cursor, baseline_y),
                        design.font,
                        design.text_scale,
                        design.outline_color,
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
) -> None:
    """Stream through the video, overlay clips, and draw subtitles."""

    cap = cv2.VideoCapture(main_video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open main video: {main_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    target_aspect_ratio = 4.0 / 5.0
    width, height = compute_cropped_dimensions(
        source_width, source_height, target_aspect_ratio
    )

    segment_clip_paths: List[Optional[str]] = []
    clip_state: Dict[str, Dict[str, object]] = {}

    for segment in highlight_segments:
        clip_path = segment.get("clip_path")
        segment_clip_paths.append(clip_path if clip_path else None)
        if not clip_path:
            continue
        if clip_path in clip_state:
            continue
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Overlay clip not found: {clip_path}")
        overlay_capture = cv2.VideoCapture(clip_path)
        if not overlay_capture.isOpened():
            raise IOError(f"Cannot open overlay clip: {clip_path}")
        total_frames = int(overlay_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        clip_state[clip_path] = {
            "capture": overlay_capture,
            "total_frames": total_frames,
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
            "finished": total_frames <= 0,
        }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot create output file: {output_path}")

    highlight_frame_ranges: List[List[int]] = []
    highlight_subtitle_indices: List[Optional[int]] = []
    for idx, segment in enumerate(highlight_segments):
        start_word = int(segment["start_word"])
        end_word = int(segment["end_word"])
        start_time = transcript[start_word]["start_time"]
        end_time = transcript[end_word]["end_time"]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        highlight_frame_ranges.append([start_frame, end_frame, idx])
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
        if subtitle_index is not None and subtitle_segments:
            sub_start, sub_end = subtitle_segments[subtitle_index]
            subtitle_end_time = transcript[sub_end]["end_time"]
            subtitle_end_frame = int(math.ceil(subtitle_end_time * fps))
            if subtitle_end_frame > highlight_frame_ranges[-1][1]:
                highlight_frame_ranges[-1][1] = subtitle_end_frame
            next_subtitle_index = subtitle_index + 1
            if 0 <= next_subtitle_index < len(subtitle_segments):
                next_sub_start, _ = subtitle_segments[next_subtitle_index]
                next_start_time = transcript[next_sub_start]["start_time"]
                next_start_frame = int(math.floor(next_start_time * fps))
                candidate_end = max(
                    highlight_frame_ranges[-1][0], next_start_frame - 1
                )
                if candidate_end > highlight_frame_ranges[-1][1]:
                    highlight_frame_ranges[-1][1] = candidate_end

    for idx in range(1, len(highlight_frame_ranges)):
        prev_range = highlight_frame_ranges[idx - 1]
        curr_range = highlight_frame_ranges[idx]
        prev_idx = prev_range[2]
        curr_idx = curr_range[2]
        prev_clip = segment_clip_paths[prev_idx]
        curr_clip = segment_clip_paths[curr_idx]
        consecutive_subtitles = False
        if subtitle_segments:
            prev_sub_idx = highlight_subtitle_indices[prev_idx]
            curr_sub_idx = highlight_subtitle_indices[curr_idx]
            consecutive_subtitles = (
                prev_sub_idx is not None
                and curr_sub_idx is not None
                and curr_sub_idx == prev_sub_idx + 1
            )
        else:
            consecutive_subtitles = curr_range[0] <= prev_range[1] + 1

        if prev_clip and curr_clip and prev_clip == curr_clip and consecutive_subtitles:
            curr_start = curr_range[0]
            prev_start = prev_range[0]
            prev_end = prev_range[1]

            if curr_start > prev_end + 1:
                prev_range[1] = curr_start - 1
            else:
                prev_range[1] = max(prev_start, min(prev_end, curr_start - 1))

            adjusted_curr_start = max(curr_start, prev_range[1] + 1)
            curr_range[0] = min(adjusted_curr_start, curr_range[1])

    frame_index = 0
    highlight_ranges_for_words = [
        (seg["start_word"], seg["end_word"]) for seg in highlight_segments
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect_ratio(frame, target_aspect_ratio)

        current_time = frame_index / fps
        active_overlay_index: Optional[int] = None
        for start_f, end_f, seg_idx in highlight_frame_ranges:
            if start_f <= frame_index <= end_f:
                active_overlay_index = seg_idx
                break

        if active_overlay_index is not None:
            clip_path = segment_clip_paths[active_overlay_index]
            if clip_path:
                clip_info = clip_state.get(clip_path)
                if clip_info is not None:
                    current_subtitle_index = highlight_subtitle_indices[
                        active_overlay_index
                    ]
                    current_segment_index = clip_info.get("current_segment_index")
                    if current_segment_index != active_overlay_index:
                        if current_segment_index is not None:
                            clip_info["last_segment_index"] = current_segment_index
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
                                and current_subtitle_index == prev_subtitle_index + 1
                            )
                        else:
                            should_continue = (
                                prev_segment_index is not None
                                and active_overlay_index == prev_segment_index + 1
                            )
                        if not should_continue:
                            clip_info["next_frame"] = 0
                            clip_info["seek_frame"] = 0
                            clip_info["needs_seek"] = True
                            clip_info["continuation_pending"] = False
                            clip_info["frames_to_drop"] = 0
                            clip_info["last_frame"] = None
                            clip_info["hold_last_frame"] = False
                            clip_info["last_capture_index"] = -1
                        else:
                            target_next = max(int(clip_info.get("next_frame", 0)), 0)
                            clip_info["seek_frame"] = target_next
                            clip_info["needs_seek"] = True
                            clip_info["continuation_pending"] = True
                            clip_info["frames_to_drop"] = 0
                            clip_info["hold_last_frame"] = False
                        clip_info["finished"] = clip_info["total_frames"] <= 0
                        clip_info["current_segment_index"] = active_overlay_index
                        clip_info["current_subtitle_index"] = current_subtitle_index
                    else:
                        clip_info["current_subtitle_index"] = current_subtitle_index

                    overlay_total_frames = clip_info["total_frames"]
                    current_index = int(clip_info.get("next_frame", 0))
                    frame_to_overlay: Optional[np.ndarray] = None

                    if overlay_total_frames <= 0:
                        frame_to_overlay = clip_info.get("last_frame")
                        clip_info["next_frame"] = current_index + 1
                        clip_info["finished"] = frame_to_overlay is None
                    else:
                        overlay_cap = clip_info["capture"]
                        loop_index = current_index % overlay_total_frames
                        last_capture_index = clip_info.get("last_capture_index", -1)
                        expected_next = (
                            (last_capture_index + 1) % overlay_total_frames
                            if last_capture_index != -1
                            else loop_index
                        )
                        if clip_info.get("needs_seek", False) or loop_index != expected_next:
                            overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, loop_index)
                            clip_info["needs_seek"] = False
                        ret_o, overlay_frame = overlay_cap.read()
                        if not ret_o:
                            clip_info["needs_seek"] = True
                            frame_to_overlay = clip_info.get("last_frame")
                            clip_info["finished"] = frame_to_overlay is None
                        else:
                            frame_to_overlay = overlay_frame
                            clip_info["last_frame"] = overlay_frame
                            clip_info["last_capture_index"] = loop_index
                            clip_info["finished"] = False
                        clip_info["next_frame"] = current_index + 1

                    if frame_to_overlay is not None:
                        overlay_frame = crop_to_aspect_ratio(
                            frame_to_overlay, target_ratio=target_aspect_ratio
                        )
                        overlay_frame = resize_overlay_for_canvas(
                            overlay_frame,
                            canvas_width=width,
                            canvas_height=height,
                            aspect_ratio=target_aspect_ratio,
                        )
                        overlay_h, overlay_w = overlay_frame.shape[:2]
                        x_start = (width - overlay_w) // 2
                        y_start = (height - overlay_h) // 2
                        frame[
                            y_start : y_start + overlay_h,
                            x_start : x_start + overlay_w,
                        ] = overlay_frame

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
    """Attach the original audio, per-segment music, and optional global music using MoviePy."""

    if not HAVE_MOVIEPY:
        print("[warn] MoviePy is not installed. Output video will be silent.")
        return

    processed_clip = mpy.VideoFileClip(silent_video_path)
    main_clip = mpy.VideoFileClip(main_video_path, audio=True)

    base_audio: Optional[mpy.AudioClip] = None
    external_audio_clip: Optional[mpy.AudioFileClip] = None
    if preserve_main_audio:
        base_audio = safe_audio_subclip(main_clip.audio, 0, processed_clip.duration)

        if base_audio is None:
            try:
                external_audio_clip = mpy.AudioFileClip(main_video_path)
                base_audio = safe_audio_subclip(
                    external_audio_clip, 0, processed_clip.duration
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Unable to load audio track from main video ({exc}).")
                base_audio = None

    audio_layers: List[mpy.AudioClip] = []
    if base_audio is not None:
        audio_layers.append(base_audio)

    if global_music_path:
        if not os.path.exists(global_music_path):
            raise FileNotFoundError(f"Global music file not found: {global_music_path}")
        global_music_clip = mpy.AudioFileClip(global_music_path)
        if global_music_clip.duration < processed_clip.duration:
            loops = math.ceil(processed_clip.duration / global_music_clip.duration)
            global_music_clip = mpy.concatenate_audioclips(
                [global_music_clip] * max(1, loops)
            )
        global_music_clip = safe_audio_subclip(
            global_music_clip, 0, processed_clip.duration
        )
        if global_music_clip is not None:
            volume = float(global_music_volume)
            if hasattr(global_music_clip, "volumex"):
                global_music_clip = global_music_clip.volumex(volume)
            elif hasattr(global_music_clip, "fx"):
                from moviepy.audio.fx import volumex as volumex_fx

                global_music_clip = volumex_fx(global_music_clip, volume)
            if hasattr(global_music_clip, "set_start"):
                global_music_clip = global_music_clip.set_start(0)
            elif hasattr(global_music_clip, "with_start"):
                global_music_clip = global_music_clip.with_start(0)
            audio_layers.append(global_music_clip)
    for segment in highlight_segments:
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
        music_clip = mpy.AudioFileClip(music_path)
        if music_clip.duration < duration:
            loops = math.ceil(duration / music_clip.duration)
            music_clip = mpy.concatenate_audioclips([music_clip] * loops)
        music_clip = safe_audio_subclip(music_clip, 0, duration)
        volume = float(segment.get("music_volume", 1.0))
        if hasattr(music_clip, "volumex"):
            music_clip = music_clip.volumex(volume)
        elif hasattr(music_clip, "fx"):
            from moviepy.audio.fx import volumex as volumex_fx
            music_clip = volumex_fx(music_clip, volume)
        else:
            music_clip = music_clip  # fallback: no volume adjustment
        if hasattr(music_clip, "set_start"):
            music_clip = music_clip.set_start(start_time)
        elif hasattr(music_clip, "with_start"):
            music_clip = music_clip.with_start(start_time)
        audio_layers.append(music_clip)

    final_clip = processed_clip
    if audio_layers:
        final_audio = mpy.CompositeAudioClip(audio_layers)
        if hasattr(final_audio, "set_duration"):
            final_audio = final_audio.set_duration(processed_clip.duration)
        elif hasattr(final_audio, "with_duration"):
            final_audio = final_audio.with_duration(processed_clip.duration)
        if hasattr(final_clip, "set_audio"):
            final_clip = final_clip.set_audio(final_audio)
        elif hasattr(final_clip, "with_audio"):
            final_clip = final_clip.with_audio(final_audio)
        else:
            raise AttributeError(
                "MoviePy VideoClip does not support set_audio/with_audio methods."
            )

    final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    final_clip.close()
    processed_clip.close()
    main_clip.close()
    if external_audio_clip is not None:
        external_audio_clip.close()


# --------------------------------------------------------------------------- #
# High level orchestration
# --------------------------------------------------------------------------- #


def render_project(config: ProjectConfig) -> Dict[str, object]:
    """Run the full pipeline and return metadata for inspection."""

    transcript = build_transcript(
        config.main_video_path,
        transcript_text=config.transcript_text,
        whisper_model=config.whisper_model,
    )
    highlight_segments = map_assignments_to_segments(
        transcript, config.highlight_assignments
    )

    any_segment_music = any(
        assignment.music_path for assignment in config.highlight_assignments
    )
    needs_audio_merge = HAVE_MOVIEPY and (
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

    process_video_with_overlays(
        config.main_video_path,
        transcript,
        highlight_segments,
        config.subtitle_design,
        silent_output_path,
        subtitle_segments=subtitle_segments,
        custom_subtitles=custom_subtitle_texts,
    )

    if needs_audio_merge:
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
        if (
            os.path.exists(silent_output_path)
            and silent_output_path != final_output_path
        ):
            os.remove(silent_output_path)

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
