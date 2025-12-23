from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from bisect import bisect_right
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


_NON_ALNUM = re.compile(r"[^a-z0-9]+")  # keep only letters/digits

def normalise_word(w: str) -> str:
    if not w:
        return ""
    w = w.replace("\ufeff", "")  # BOM
    w = unicodedata.normalize("NFKC", w).lower().strip()
    w = w.replace("’", "'")  # optional: unify apostrophe types
    w = _NON_ALNUM.sub("", w)  # strips commas/periods/etc.
    return w


_PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "called",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "some",
    "somebody",
    "someone",
    "something",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
    "yours",
}


_NUMBER_WORD_UNITS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_NUMBER_WORD_TEENS: Dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_NUMBER_WORD_TENS: Dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,  # common misspelling
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_NUMBER_WORD_VALUES: Dict[str, int] = {
    **_NUMBER_WORD_UNITS,
    **_NUMBER_WORD_TEENS,
    **_NUMBER_WORD_TENS,
}


def _parse_compound_number_word_token(token: str) -> Optional[int]:
    """Parse a normalised (alnum) English number word into an integer when possible.

    Supports common forms produced by ASR tokenisation:
    - Simple numbers ("fifty", "twelve")
    - Compounds without separators ("twentyone", "seventyseven")
    - Hundreds/thousands without separators ("onehundredfive", "twothousandandten")

    Returns ``None`` when the token does not look like a number word.
    """

    if not token:
        return None

    if token.isdigit():
        return int(token)

    direct = _NUMBER_WORD_VALUES.get(token)
    if direct is not None:
        return direct

    # Handle compounds like "twentyone" / "seventyseven".
    for tens_word, tens_value in _NUMBER_WORD_TENS.items():
        if token.startswith(tens_word):
            rest = token[len(tens_word) :]
            if not rest:
                return tens_value
            unit_val = _NUMBER_WORD_UNITS.get(rest)
            if unit_val is not None:
                return tens_value + unit_val

    # Handle hundreds/thousands, potentially with an "and" joiner.
    for magnitude_word, magnitude_value in (("thousand", 1000), ("hundred", 100)):
        if magnitude_word not in token:
            continue
        prefix, suffix = token.split(magnitude_word, 1)
        if not prefix:
            return None
        prefix_val = _parse_compound_number_word_token(prefix)
        if prefix_val is None:
            return None
        total = prefix_val * magnitude_value
        if not suffix:
            return total
        if suffix.startswith("and"):
            suffix = suffix[3:]
        if not suffix:
            return total
        suffix_val = _parse_compound_number_word_token(suffix)
        if suffix_val is None:
            return None
        return total + suffix_val

    return None


def normalise_numeric_token(token: str) -> str:
    """Canonicalise numeric words/digits to improve phrase matching.

    The pipeline uses token-based phrase matching; ASR often emits "fifty" vs "50",
    so mapping both to the same representation prevents spurious "not found" errors.
    """

    if not token:
        return token

    parsed = _parse_compound_number_word_token(token)
    if parsed is None:
        return token
    return str(parsed)


def slow_down_video(
    input_path: str, output_path: str, speed_factor: float, target_fps: float
) -> None:
    """
    Create an overlay-ready clip:
      - slowed down when speed_factor < 1.0
      - CFR at target_fps (avoids VFR timing that can break OpenCV seeking/reads)
      - yuv420p pixel format (most compatible)
      - no audio (overlay audio is never used in the pipeline)
    """
    speed_factor = float(speed_factor)
    target_fps = float(target_fps)

    if speed_factor <= 1e-6:
        raise ValueError(f"Invalid speed_factor={speed_factor}")

    # Build a CFR filter chain.
    # NOTE: fps filter duplicates/drops frames to make CFR, which OpenCV likes.
    vf_parts = []
    if speed_factor < 0.999:
        vf_parts.append(f"setpts={1.0/speed_factor}*PTS")
    vf_parts.append(f"fps={target_fps:.06f}")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-an",  # strip audio (not used for overlays)
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(
            f"  [CLIP PROCESSING] Prepared overlay clip: {os.path.basename(input_path)} "
            f"(speed_factor={speed_factor:.4f}, fps={target_fps:.2f})"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"  [CLIP PROCESSING] ffmpeg error preparing clip: {e.stderr}")
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


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    """Return media duration in seconds using ffprobe when available."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float((result.stdout or "").strip())
        return duration if duration > 0 else None
    except Exception:
        return None


def _parse_ffprobe_fraction(value: Optional[str]) -> Optional[float]:
    """Parse ffprobe frame-rate strings like '30000/1001' into a float."""

    if not value:
        return None
    value = value.strip()
    if not value or value in {"0/0", "N/A"}:
        return None

    if "/" in value:
        try:
            num_s, den_s = value.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den <= 0:
                return None
            out = num / den
            return out if out > 1e-9 and math.isfinite(out) else None
        except Exception:
            return None

    try:
        out = float(value)
        return out if out > 1e-9 and math.isfinite(out) else None
    except Exception:
        return None


def ffprobe_video_stream_stats(path: str) -> Dict[str, Optional[object]]:
    """Return basic video-stream metadata from ffprobe (best-effort).

    Keys:
      - fps: float | None (prefers avg_frame_rate, else r_frame_rate)
      - avg_fps: float | None
      - r_fps: float | None
      - nb_frames: int | None
    """

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return {}
        stream = streams[0] or {}

        avg_fps = _parse_ffprobe_fraction(stream.get("avg_frame_rate"))
        r_fps = _parse_ffprobe_fraction(stream.get("r_frame_rate"))
        fps = avg_fps or r_fps

        nb_frames_raw = stream.get("nb_frames")
        nb_frames: Optional[int] = None
        if isinstance(nb_frames_raw, str) and nb_frames_raw.isdigit():
            nb_frames = int(nb_frames_raw)
        elif isinstance(nb_frames_raw, (int, float)):
            nb_frames_int = int(nb_frames_raw)
            nb_frames = nb_frames_int if nb_frames_int > 0 else None

        return {
            "fps": fps,
            "avg_fps": avg_fps,
            "r_fps": r_fps,
            "nb_frames": nb_frames,
        }
    except Exception:
        return {}


def compute_effective_fps(
    cap: cv2.VideoCapture, path: str, fallback_fps: float
) -> float:
    """Compute an effective FPS for VFR/metadata-mismatched clips.

    Uses `total_frames / ffprobe_duration` when possible, falling back to
    OpenCV-reported FPS.
    """

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = ffprobe_duration_seconds(path)
    cv2_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    stream_stats = ffprobe_video_stream_stats(path)
    ffprobe_fps_raw = stream_stats.get("fps")
    ffprobe_fps = (
        float(ffprobe_fps_raw)
        if isinstance(ffprobe_fps_raw, (int, float))
        and float(ffprobe_fps_raw) > 1e-9
        and math.isfinite(float(ffprobe_fps_raw))
        else None
    )
    ffprobe_nb_frames_raw = stream_stats.get("nb_frames")
    ffprobe_nb_frames = (
        int(ffprobe_nb_frames_raw)
        if isinstance(ffprobe_nb_frames_raw, int) and ffprobe_nb_frames_raw > 0
        else None
    )

    fallback_fps = float(fallback_fps) if fallback_fps else 25.0

    # Candidate "effective fps" computed from frames/duration (can be wrong if
    # either side of the ratio is bad). Prefer ffprobe's nb_frames when present.
    candidate_eff: Optional[float] = None
    candidate_src: Optional[str] = None
    if duration and duration > 0:
        if ffprobe_nb_frames and ffprobe_nb_frames > 0:
            candidate_eff = float(ffprobe_nb_frames) / float(duration)
            candidate_src = "ffprobe_nb_frames/format_duration"
        elif total_frames > 0:
            candidate_eff = float(total_frames) / float(duration)
            candidate_src = "cv2_frame_count/format_duration"

    # Preferred fallbacks (never raise): ffprobe stream fps -> cv2 fps -> provided fallback.
    fallback_choices: List[Tuple[str, float]] = []
    if ffprobe_fps is not None:
        fallback_choices.append(("ffprobe_stream_fps", float(ffprobe_fps)))
    if cv2_fps > 1e-9 and math.isfinite(cv2_fps):
        fallback_choices.append(("cv2_fps", float(cv2_fps)))
    fallback_choices.append(("fallback_fps", float(fallback_fps)))

    def _pick_fallback() -> Tuple[str, float]:
        return fallback_choices[0]

    # Guard against bad metadata inflating effective fps (causes overlay frames to
    # advance too quickly and "look sped up").
    if candidate_eff is not None and candidate_eff > 1e-9 and math.isfinite(candidate_eff):
        suspicious_reason: Optional[str] = None

        # Hard sanity bounds (ultra-high values are almost always metadata bugs).
        if candidate_eff < 1.0 or candidate_eff > 240.0:
            suspicious_reason = "candidate_out_of_range"
        else:
            # Compare against ffprobe stream fps (most reliable reference) when available.
            if ffprobe_fps is not None and ffprobe_fps > 1e-9 and math.isfinite(ffprobe_fps):
                ratio = candidate_eff / float(ffprobe_fps)
                if ratio < 0.67 or ratio > 1.50:
                    suspicious_reason = f"candidate_vs_ffprobe_ratio={ratio:.2f}"
            # If ffprobe stream fps unavailable, compare against OpenCV FPS with wider tolerance.
            elif cv2_fps > 1e-9 and math.isfinite(cv2_fps):
                ratio = candidate_eff / float(cv2_fps)
                if ratio < 0.40 or ratio > 2.50:
                    suspicious_reason = f"candidate_vs_cv2_ratio={ratio:.2f}"

        if suspicious_reason:
            chosen_src, chosen_fps = _pick_fallback()
            try:
                logger.warning(
                    "  [FPS GUARD] Suspicious effective FPS for %s (%s): "
                    "candidate=%.3f (%s), ffprobe_stream_fps=%s, cv2_fps=%s, "
                    "cv2_frames=%s, ffprobe_nb_frames=%s, ffprobe_duration=%s, fallback_fps=%.3f "
                    "-> using %s=%.3f",
                    os.path.basename(path) if path else path,
                    suspicious_reason,
                    candidate_eff,
                    candidate_src,
                    f"{ffprobe_fps:.3f}" if ffprobe_fps is not None else None,
                    f"{cv2_fps:.3f}" if (cv2_fps > 1e-9 and math.isfinite(cv2_fps)) else None,
                    total_frames if total_frames > 0 else None,
                    ffprobe_nb_frames,
                    f"{duration:.6f}" if duration is not None else None,
                    float(fallback_fps),
                    chosen_src,
                    float(chosen_fps),
                )
            except Exception:
                # Never allow logging format issues to break the pipeline.
                pass

            return float(chosen_fps) if float(chosen_fps) > 1e-9 else float(fallback_fps)

        return float(candidate_eff)

    chosen_src, chosen_fps = _pick_fallback()
    return float(chosen_fps) if float(chosen_fps) > 1e-9 else float(fallback_fps)


def _adaptive_cluster_gap(gaps: List[float], default: float = 1.25) -> float:
    # keep only sane, positive gaps
    gaps = [g for g in gaps if g > 1e-3 and math.isfinite(g)]
    if len(gaps) < 2:
        return default

    gaps.sort()

    # find biggest multiplicative jump (often separates “small pauses” vs “real breaks”)
    best_i = None
    best_ratio = 1.0
    for i in range(len(gaps) - 1):
        a = gaps[i]
        b = gaps[i + 1]
        ratio = b / max(a, 1e-6)
        if ratio > best_ratio:
            best_ratio = ratio
            best_i = i

    if best_i is not None and best_ratio >= 2.5:
        thr = 0.5 * (gaps[best_i] + gaps[best_i + 1])
    else:
        # fallback: “80th percentile” style threshold
        thr = gaps[int(0.80 * (len(gaps) - 1))]

    # clamp + tiny cushion so we kill 1-frame flashes
    thr = min(max(thr + 0.05, 0.25), 1.75)
    return thr


def build_overlay_schedule_times(
    highlight_segments,
    transcript,
    subtitle_segments=None,
    use_subtitle_bounds_for_overlay=True,
    cluster_gap_seconds=None,
    lead_in_seconds=0.15,
    tail_out_seconds=0.28,
    min_on_seconds=0.001,
    logger=None,
):
    """
    - Adds lead-in / tail-out ONLY inside real pauses (bounded by transcript word timings).
    - Guarantees schedule intervals NEVER overlap.
    - Works with optional subtitle segment expansion (use_subtitle_bounds_for_overlay).
    - Returns:
        schedule_times: List[(start_s, end_s, seg_idx)] sorted by start_s
        highlight_subtitle_indices: List[Optional[int]]
        highlight_subtitle_spans:   List[Optional[(s0, s1)]]
        segment_on_durations_sec:   List[float] per seg_idx
    """

    n = len(highlight_segments)
    if n == 0:
        return [], [], [], []

    if not transcript:
        raise ValueError("Transcript is empty; cannot build overlay schedule.")

    # ---- Map highlight segments to subtitle spans (optional) ----
    highlight_subtitle_indices = [None] * n
    highlight_subtitle_spans = [None] * n

    def _overlapping_subtitle_span(sw, ew):
        if not subtitle_segments:
            return None
        overlaps = []
        for si, (ssw, sew) in enumerate(subtitle_segments):
            if ssw <= ew and sew >= sw:  # overlap
                overlaps.append(si)
        if not overlaps:
            return None
        return (overlaps[0], overlaps[-1])

    phrases = []
    for seg_idx, seg in enumerate(highlight_segments):
        sw = int(seg["start_word"])
        ew = int(seg["end_word"])

        span = _overlapping_subtitle_span(sw, ew)
        if span is not None:
            highlight_subtitle_indices[seg_idx] = span[0]
            highlight_subtitle_spans[seg_idx] = span
            if use_subtitle_bounds_for_overlay:
                s0, s1 = span
                sw = int(subtitle_segments[s0][0])
                ew = int(subtitle_segments[s1][1])

        sw = max(0, min(sw, len(transcript) - 1))
        ew = max(0, min(ew, len(transcript) - 1))
        if ew < sw:
            sw, ew = ew, sw

        t0 = float(transcript[sw]["start_time"])
        t1 = float(transcript[ew]["end_time"])
        if t1 < t0:
            t1 = t0  # safety

        phrases.append(
            {
                "seg_idx": seg_idx,
                "sw": sw,
                "ew": ew,
                "t0": t0,
                "t1": t1,
            }
        )

    # Sort by actual time
    phrases.sort(key=lambda p: p["t0"])

    # ---- Adaptive cluster gap (if not provided) ----
    if cluster_gap_seconds is None:
        gaps = []
        for i in range(1, len(phrases)):
            gaps.append(max(0.0, phrases[i]["t0"] - phrases[i - 1]["t1"]))
        cluster_gap_seconds = _adaptive_cluster_gap(gaps)

    if logger:
        logger.info(f"[SCHED_SETUP] adaptive cluster_gap_seconds={cluster_gap_seconds:.3f}s")

    transcript_end = float(transcript[-1]["end_time"])

    # ---- Helper: compute "pause-only" guardrails around the chosen sw/ew ----
    def _guardrails(sw, ew):
        t0 = float(transcript[sw]["start_time"])
        t1 = float(transcript[ew]["end_time"])

        prev_end = 0.0
        if sw > 0:
            prev_end = float(transcript[sw - 1]["end_time"])

        next_start = transcript_end
        if ew + 1 < len(transcript):
            next_start = float(transcript[ew + 1]["start_time"])

        # ASR can overlap timings; never let guardrails chop into the phrase itself.
        if prev_end > t0:
            prev_end = t0
        if next_start < t1:
            next_start = t1

        return prev_end, next_start, t0, t1

    m = len(phrases)
    starts = [0.0] * m
    ends = [0.0] * m

    # ---- First pass: compute starts (lead-in), clamped into the pause after prev word ----
    for i, p in enumerate(phrases):
        prev_end, next_start, t0, t1 = _guardrails(p["sw"], p["ew"])
        p["_prev_end"] = prev_end
        p["_next_start"] = next_start
        p["_t0"] = t0
        p["_t1"] = t1

        if i == 0:
            st = t0 - lead_in_seconds
        else:
            gap = max(0.0, t0 - phrases[i - 1]["t1"])
            st = t0 if gap <= cluster_gap_seconds else (t0 - lead_in_seconds)

        # Clamp: lead-in can ONLY live in the pause (>= prev_end) and never after t0.
        st = max(0.0, st, prev_end)
        st = min(st, t0)
        starts[i] = st

    # ---- Second pass: compute ends (tail-out), clamped into the pause before next word ----
    for i, p in enumerate(phrases):
        t1 = p["_t1"]
        next_start = p["_next_start"]

        if i < m - 1:
            gap_to_next = max(0.0, phrases[i + 1]["t0"] - t1)
            if gap_to_next <= cluster_gap_seconds:
                et = starts[i + 1]  # butt to next start within the run
            else:
                et = t1 + tail_out_seconds
        else:
            et = t1 + tail_out_seconds

        # Clamp: tail-out can ONLY live in the pause (<= next_start) and never before t1.
        et = min(et, next_start)
        et = max(et, t1)
        ends[i] = et

    # ---- Hard guarantee: no overlaps (even if lead/tail would collide) ----
    for i in range(m - 1):
        if ends[i] > starts[i + 1]:
            ends[i] = starts[i + 1]

    # ---- Final clamp to transcript end + min duration where possible (without re-overlapping) ----
    segment_on_durations_sec = [0.0] * n
    schedule_times = []

    for i, p in enumerate(phrases):
        st = max(0.0, min(starts[i], transcript_end))
        et = max(0.0, min(ends[i], transcript_end))

        # Ensure not inverted
        if et < st:
            et = st

        # Minimal duration, but never force an overlap
        if i < m - 1:
            max_len = max(0.0, starts[i + 1] - st)
            min_len = min(min_on_seconds, max_len)
        else:
            min_len = min_on_seconds

        if et < st + min_len:
            et = st + min_len

        if i < m - 1 and et > starts[i + 1]:
            et = starts[i + 1]  # re-snap if needed

        seg_idx = p["seg_idx"]
        schedule_times.append((st, et, seg_idx))
        segment_on_durations_sec[seg_idx] = max(0.0, et - st)

    schedule_times.sort(key=lambda x: x[0])

    # Absolute last safety net (should never trigger now)
    for i in range(len(schedule_times) - 1):
        st, et, idx = schedule_times[i]
        nst = schedule_times[i + 1][0]
        if et > nst:
            schedule_times[i] = (st, nst, idx)

    # ---------------------------------------------------------------------
    # Bridge small gaps between consecutive overlays.
    #
    # Rationale:
    # - The schedule builder intentionally limits lead-in/tail-out to short
    #   amounts (and uses an adaptive "cluster gap") so main video can show
    #   during real pauses.
    # - In practice, short mid-gap pauses (hundreds of ms) between two B-roll
    #   overlays look like a distracting "flash" of the talking head.
    #
    # So we "stitch" consecutive overlays by extending ONLY the previous
    # overlay's scheduled end up to the next overlay's start when the *remaining*
    # schedule gap is small enough.
    # ---------------------------------------------------------------------
    bridge_gap_seconds = 0.75
    if bridge_gap_seconds > 1e-6 and len(schedule_times) >= 2:
        for i in range(len(schedule_times) - 1):
            st_i, et_i, idx_i = schedule_times[i]
            st_n = schedule_times[i + 1][0]
            gap = float(st_n) - float(et_i)

            if gap > 0.0 and gap <= bridge_gap_seconds:
                if logger:
                    logger.info(
                        "  [SCHED] Bridging small gap: seg=%d end %.3f -> %.3f (gap %.3fs)",
                        int(idx_i),
                        float(et_i),
                        float(st_n),
                        gap,
                    )
                schedule_times[i] = (float(st_i), float(st_n), int(idx_i))
                segment_on_durations_sec[int(idx_i)] = max(0.0, float(st_n) - float(st_i))

    return schedule_times, highlight_subtitle_indices, highlight_subtitle_spans, segment_on_durations_sec


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

    # 0) Prefer cached transcript if it exists and is newer than the video.
    try:
        cached_path = os.path.splitext(video_path)[0] + "_subtitle.json"
        if os.path.exists(cached_path):
            video_mtime = os.path.getmtime(video_path)
            cached_mtime = os.path.getmtime(cached_path)
            if cached_mtime >= video_mtime:
                with open(cached_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    isinstance(cached, list)
                    and cached
                    and all(
                        isinstance(item, dict)
                        and "word" in item
                        and "start_time" in item
                        and "end_time" in item
                        for item in cached
                    )
                ):
                    logger.info(
                        f"  [TRANSCRIPT] Using cached transcript from {cached_path}"
                    )
                    return cached
    except Exception as exc:
        logger.warning(
            f"  [TRANSCRIPT] Failed to load cached transcript; re-transcribing ({exc})"
        )

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

    Returns the ``occurrence``-th acceptable match in time order after
    ``start_index``. This preserves the intended meaning of ``occurrence`` as
    "nth occurrence in the transcript", and avoids jumping to a later (slightly
    higher-similarity) match that would break chronological alignment.
    """
    token_list = list(tokens)
    if not token_list or start_index >= len(transcript_words):
        return None

    min_window = max(1, len(token_list) - max_window_delta)
    max_window = max(min_window, len(token_list) + max_window_delta)
    phrase_str = " ".join(token_list)

    anchors = [a for a in _anchor_tokens_for_phrase(token_list) if a != token_list[0]]
    required_anchor_hits = 0
    if token_list[0] in _PHRASE_STOPWORDS or len(token_list[0]) < 3:
        required_anchor_hits = 1 if len(token_list) <= 6 else min(2, len(anchors))

    match_count = 0

    for idx in range(start_index, len(transcript_words) - min_window + 1):
        # First word must still match to keep things sane
        if transcript_words[idx] != token_list[0]:
            continue

        remaining = len(transcript_words) - idx
        window_max = min(max_window, remaining)

        if required_anchor_hits > 0 and anchors:
            span_set = set(transcript_words[idx : idx + window_max])
            hits = sum(1 for a in anchors if a in span_set)
            if hits < required_anchor_hits:
                continue

        best_ratio = 0.0
        best_end: Optional[int] = None

        for window_len in range(min_window, window_max + 1):
            window_tokens = transcript_words[idx : idx + window_len]
            window_token_list = list(window_tokens)
            if required_anchor_hits > 0 and anchors:
                hits = sum(1 for a in anchors if a in window_token_list)
                if hits < required_anchor_hits:
                    continue
            ratio_chars = SequenceMatcher(
                None, phrase_str, " ".join(window_token_list)
            ).ratio()
            ratio_tokens = SequenceMatcher(None, token_list, window_token_list).ratio()
            ratio = max(ratio_chars, ratio_tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_end = idx + window_len - 1

        if best_end is not None and best_ratio >= min_similarity:
            match_count += 1
            if match_count == occurrence:
                return idx, best_end

    return None


def _anchor_tokens_for_phrase(tokens: Sequence[str], max_anchors: int = 6) -> List[str]:
    """Pick stable anchor tokens for fuzzy matching.

    We avoid common stopwords/pronouns so phrases like "He developed ..." can
    still match transcripts like "It uses ..." via the distinctive content words.
    """

    candidates = [
        t
        for t in tokens
        if t and t not in _PHRASE_STOPWORDS and len(t) >= 3
    ]
    if not candidates:
        candidates = [t for t in tokens if t]

    # Deduplicate while preferring longer tokens first.
    ordered: List[str] = []
    seen = set()
    for tok in sorted(candidates, key=len, reverse=True):
        if tok in seen:
            continue
        seen.add(tok)
        ordered.append(tok)
        if len(ordered) >= max_anchors:
            break
    return ordered


def _find_anchor_fuzzy_phrase_match(
    transcript_words: Sequence[str],
    tokens: Sequence[str],
    occurrence: int,
    start_index: int,
    max_window_delta: int = 3,
    min_similarity: float = 0.65,
) -> Optional[Tuple[int, int]]:
    """Fuzzy match that doesn't require the first word to match.

    This is a last-resort matcher for WhisperX drift where the phrase begins with
    mismatched pronouns/filler (e.g. "He developed ..." vs "It uses ...") but the
    core content words still appear in order.
    """

    token_list = list(tokens)
    if not token_list or start_index >= len(transcript_words):
        return None

    anchors = _anchor_tokens_for_phrase(token_list)
    required_anchor_hits = 1 if len(token_list) <= 3 else min(2, len(anchors))

    min_window = max(1, len(token_list) - max_window_delta)
    max_window = max(min_window, len(token_list) + max_window_delta)
    phrase_str = " ".join(token_list)
    match_count = 0

    for idx in range(start_index, len(transcript_words) - min_window + 1):
        remaining = len(transcript_words) - idx
        window_max = min(max_window, remaining)
        if window_max <= 0:
            continue

        # Quick anchor check on the largest window span.
        if required_anchor_hits > 0 and anchors:
            span_set = set(transcript_words[idx : idx + window_max])
            hits = sum(1 for a in anchors if a in span_set)
            if hits < required_anchor_hits:
                continue

        best_ratio = 0.0
        best_end: Optional[int] = None

        for window_len in range(min_window, window_max + 1):
            window_tokens = transcript_words[idx : idx + window_len]
            if required_anchor_hits > 0 and anchors:
                hits = sum(1 for a in anchors if a in window_tokens)
                if hits < required_anchor_hits:
                    continue

            window_token_list = list(window_tokens)
            ratio_chars = SequenceMatcher(None, phrase_str, " ".join(window_token_list)).ratio()
            ratio_tokens = SequenceMatcher(None, token_list, window_token_list).ratio()
            ratio = max(ratio_chars, ratio_tokens)

            if ratio > best_ratio:
                best_ratio = ratio
                best_end = idx + window_len - 1

        if best_end is not None and best_ratio >= min_similarity:
            match_count += 1
            if match_count == occurrence:
                return idx, best_end

    return None


def _drop_leading_stopwords(tokens: Sequence[str], max_drop: int = 3) -> List[str]:
    out = [t for t in tokens if t]
    dropped = 0
    while out and out[0] in _PHRASE_STOPWORDS and dropped < max_drop:
        out.pop(0)
        dropped += 1
    return out


def _remove_stopwords(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t and t not in _PHRASE_STOPWORDS]


def _generate_token_merge_variants(
    tokens: Sequence[str],
    transcript_vocab: set,
) -> List[List[str]]:
    """Generate token lists with adjacent merges that exist in the transcript.

    WhisperX occasionally emits fused alnum tokens (e.g. "3-step" -> "3step",
    "67 year old" -> "67yearold"). As a last-resort fallback, merge adjacent
    phrase tokens when the merged form is present in the transcript vocabulary.
    """

    token_list = [t for t in tokens if t]
    if len(token_list) < 2:
        return []

    variants: List[List[str]] = []
    seen: set = set()

    def _add(candidate: List[str]) -> None:
        key = tuple(candidate)
        if key in seen:
            return
        seen.add(key)
        variants.append(candidate)

    n = len(token_list)

    for i in range(n - 1):
        merged = token_list[i] + token_list[i + 1]
        if merged in transcript_vocab:
            _add(token_list[:i] + [merged] + token_list[i + 2 :])

        second = token_list[i + 1]
        if second.endswith("s") and len(second) > 1:
            merged_singular = token_list[i] + second[:-1]
            if merged_singular in transcript_vocab:
                _add(token_list[:i] + [merged_singular] + token_list[i + 2 :])

    for i in range(n - 2):
        merged = token_list[i] + token_list[i + 1] + token_list[i + 2]
        if merged in transcript_vocab:
            _add(token_list[:i] + [merged] + token_list[i + 3 :])

        third = token_list[i + 2]
        if third.endswith("s") and len(third) > 1:
            merged_singular = token_list[i] + token_list[i + 1] + third[:-1]
            if merged_singular in transcript_vocab:
                _add(token_list[:i] + [merged_singular] + token_list[i + 3 :])

    return variants



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

    # Numeric word canonicalisation (e.g. "fifty" vs "50") is attempted only after
    # the more conservative matchers above, so existing behaviour remains stable
    # unless we would otherwise fail.
    canonical_transcript = [normalise_numeric_token(tok) for tok in transcript_words]
    canonical_tokens = [normalise_numeric_token(tok) for tok in target_tokens]

    match = _find_exact_phrase_match(
        canonical_transcript, canonical_tokens, target_occurrence, start_index
    )
    if match:
        return match

    match = _find_fuzzy_phrase_match(
        canonical_transcript, canonical_tokens, target_occurrence, start_index
    )
    if match:
        return match

    # Hyphenated tokens sometimes appear as a single word ("67-year-old") or split
    # into multiple tokens ("67 year old") depending on ASR output. If we still
    # failed, retry with a tokenisation variant that splits common joiners.
    split_phrase = (
        phrase.replace("-", " ")
        .replace("–", " ")
        .replace("—", " ")
        .replace("/", " ")
    )
    split_tokens = [normalise_word(tok) for tok in split_phrase.split()]
    split_tokens = [tok for tok in split_tokens if tok]
    canonical_split_tokens = [normalise_numeric_token(tok) for tok in split_tokens]
    if split_tokens and split_tokens != target_tokens:
        match = _find_exact_phrase_match(
            transcript_words, split_tokens, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_fuzzy_phrase_match(
            transcript_words, split_tokens, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_exact_phrase_match(
            canonical_transcript,
            canonical_split_tokens,
            target_occurrence,
            start_index,
        )
        if match:
            return match

        match = _find_fuzzy_phrase_match(
            canonical_transcript,
            canonical_split_tokens,
            target_occurrence,
            start_index,
        )
        if match:
            return match

    # If we still failed, try a less conservative fuzzy matcher that does not
    # require the first token to match (useful when Whisper drifts on pronouns or
    # filler words at the start of a sentence).
    match = _find_anchor_fuzzy_phrase_match(
        transcript_words, target_tokens, target_occurrence, start_index
    )
    if match:
        return match

    match = _find_anchor_fuzzy_phrase_match(
        canonical_transcript, canonical_tokens, target_occurrence, start_index
    )
    if match:
        return match

    # Stopword-aware variants can rescue phrases like:
    #   "He developed something called Airlift Technology"
    # when the transcript contains:
    #   "It uses airlift technology"
    variant_tokens: List[List[str]] = []
    for base in (target_tokens, split_tokens):
        if not base:
            continue
        variant_tokens.append(_drop_leading_stopwords(base))
        variant_tokens.append(_remove_stopwords(base))

    seen_variants = set()
    for variant in variant_tokens:
        if not variant:
            continue
        key = tuple(variant)
        if key in seen_variants:
            continue
        seen_variants.add(key)
        if variant == target_tokens:
            continue

        match = _find_exact_phrase_match(
            transcript_words, variant, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_fuzzy_phrase_match(
            transcript_words, variant, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_anchor_fuzzy_phrase_match(
            transcript_words, variant, target_occurrence, start_index, min_similarity=0.70
        )
        if match:
            return match

        canonical_variant = [normalise_numeric_token(tok) for tok in variant]

        match = _find_exact_phrase_match(
            canonical_transcript, canonical_variant, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_fuzzy_phrase_match(
            canonical_transcript, canonical_variant, target_occurrence, start_index
        )
        if match:
            return match

        match = _find_anchor_fuzzy_phrase_match(
            canonical_transcript,
            canonical_variant,
            target_occurrence,
            start_index,
            min_similarity=0.70,
        )
        if match:
            return match

    # Last-resort: handle fused ASR tokens like "3step" / "67yearold" by merging
    # adjacent phrase tokens when the merged form exists in the transcript.
    transcript_vocab = set(transcript_words)
    canonical_vocab = set(canonical_transcript)

    merge_attempts = [
        ("raw", transcript_words, transcript_vocab, target_tokens),
        ("raw", transcript_words, transcript_vocab, split_tokens),
        ("canon", canonical_transcript, canonical_vocab, canonical_tokens),
        ("canon", canonical_transcript, canonical_vocab, canonical_split_tokens),
    ]

    seen_merge_variants: set = set()
    for tag, words, vocab, base_tokens in merge_attempts:
        for merged_tokens in _generate_token_merge_variants(base_tokens, vocab):
            key = (tag, tuple(merged_tokens))
            if key in seen_merge_variants:
                continue
            seen_merge_variants.add(key)

            match = _find_exact_phrase_match(
                words, merged_tokens, target_occurrence, start_index
            )
            if match:
                return match

            match = _find_fuzzy_phrase_match(
                words, merged_tokens, target_occurrence, start_index
            )
            if match:
                return match

            match = _find_anchor_fuzzy_phrase_match(
                words,
                merged_tokens,
                target_occurrence,
                start_index,
                min_similarity=0.70,
            )
            if match:
                return match

    raise ValueError(
        f"Phrase '{phrase}' occurrence {target_occurrence} not found in transcript."
    )


# --------------------------------------------------------------------------- #
# Robust phrase matcher (alignment-based, phrase-aware)
# --------------------------------------------------------------------------- #

import bisect
from typing import Union

# Treat these as “cheap to drop” because ASR often omits/changes them.
_MATCH_STOPWORDS = {
    "a","an","the","to","of","and","or","but","if","then","so","for","in","on","at","by","with","from",
    "is","are","was","were","be","been","being","am",
    "it","its","that","this","these","those","as","about","into","over","under","up","out",
    "you","your","yours","we","our","i","me","my",
    "im","ive","youre","thats","heres",
    # numeric fluff
    "percent","dollars",
}

_NUM_WORD_TO_DIGIT = {
    "zero":"0","oh":"0","o":"0",
    "one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "ten":"10","eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19",
    "twenty":"20","thirty":"30","forty":"40","fifty":"50","sixty":"60","seventy":"70","eighty":"80","ninety":"90",
}

_ALNUM_SPLIT_RE = re.compile(r"[a-z]+|\d+", re.I)
_HYPHENS_RE = re.compile(r"[-–—]+")


@dataclass(frozen=True)
class TranscriptIndex:
    """
    Preprocessed transcript for robust matching.

    - word_tokens: original per-word tokens you pass in (already normalised_word()’d)
    - tokens: expanded tokens (e.g., "3step" -> ["3","step"])
    - token_to_word: token index -> original word index mapping
    """
    word_tokens: Tuple[str, ...]
    tokens: Tuple[str, ...]
    token_to_word: Tuple[int, ...]


def _pm_basic_norm(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _pm_split_alnum(token: str) -> List[str]:
    parts = _ALNUM_SPLIT_RE.findall(token.lower())
    return parts if parts else []


def _pm_canon_single(tok: str) -> str:
    return _NUM_WORD_TO_DIGIT.get(tok.lower(), tok.lower())


def _pm_stem(tok: str) -> str:
    # tiny stemmer: fixes need/needs, step/steps, etc.
    t = tok
    for suf in ("'s", "s", "es", "ed", "ing"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[: -len(suf)]
    return t


_DIGIT_TO_NUM_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen",
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty",
    "70": "seventy", "80": "eighty", "90": "ninety",
}

_VOWELS = set("aeiouy")

def _pm_consonant_skeleton(tok: str) -> str:
    """
    Lightweight phonetic-ish key:
      - strips vowels
      - collapses repeats
    Helps with ASR confusions like:
      force ~ four, entry ~ century
    """
    t = "".join(ch for ch in tok.lower() if ch.isalnum())
    if not t:
        return ""
    if t.isdigit():
        return t
    sk = "".join(ch for ch in t if ch not in _VOWELS)
    if not sk:
        sk = t[0]
    out = []
    for ch in sk:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)

def _pm_token_sim_basic(a: str, b: str) -> float:
    """Similarity in [0,1] without digit<->word bridging."""
    if a == b:
        return 1.0

    sa, sb = _pm_stem(a), _pm_stem(b)
    if sa == sb:
        return 0.95

    # numeric substring: "50" matches "5950"
    if a.isdigit() and b.isdigit() and (a in b or b in a):
        return 0.90

    # merged word substring: "heimlich" matches "selfheimlich"
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return 0.88

    # NEW: consonant-skeleton similarity (handles "force"~"four", "entry"~"century")
    ska = _pm_consonant_skeleton(a)
    skb = _pm_consonant_skeleton(b)
    if len(ska) >= 2 and len(skb) >= 2:
        if ska == skb:
            return 0.88
        if (ska in skb or skb in ska) and (len(a) >= 4 or len(b) >= 4):
            return 0.86
        r2 = SequenceMatcher(None, ska, skb).ratio()
        if r2 >= 0.84 and (len(ska) >= 3 or len(skb) >= 3):
            return max(0.86, r2)

    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.84 else 0.0

def _pm_token_sim(a: str, b: str) -> float:
    """
    Similarity in [0,1], with digit<->word bridging.
    This fixes cases where transcript canonicalizes number-words to digits:
      transcript: "four" -> "4"
      phrase: "force" (ASR confusion) needs to compare against "four"
    """
    a_vars = [a]
    b_vars = [b]

    if a.isdigit():
        w = _DIGIT_TO_NUM_WORD.get(a)
        if w:
            a_vars.append(w)

    if b.isdigit():
        w = _DIGIT_TO_NUM_WORD.get(b)
        if w:
            b_vars.append(w)

    best = 0.0
    for av in a_vars:
        for bv in b_vars:
            s = _pm_token_sim_basic(av, bv)
            if s > best:
                best = s
                if best >= 1.0:
                    return 1.0
    return best



def _pm_canon_seq(tokens: List[str], idxs: Optional[List[int]] = None) -> Tuple[List[str], Optional[List[int]]]:
    """
    Canonicalize sequences:
      - number words -> digits
      - "triple zero"/"double zero" -> 0 0 0 / 0 0
      - "000" -> 0 0 0  (ONLY all-zeros; avoids messing with 911 etc.)
    """
    out_t: List[str] = []
    out_i: Optional[List[int]] = [] if idxs is not None else None

    i = 0
    while i < len(tokens):
        t = _pm_canon_single(tokens[i])
        wi = idxs[i] if idxs is not None else None

        if t in ("triple", "double") and i + 1 < len(tokens):
            nxt = _pm_canon_single(tokens[i + 1])
            nxt_wi = idxs[i + 1] if idxs is not None else None
            if nxt in ("0", "zero", "oh", "o"):
                count = 3 if t == "triple" else 2
                if out_i is not None:
                    # spread indices so span includes both words
                    out_t.append("0"); out_i.append(wi if wi is not None else 0)
                    for _ in range(count - 1):
                        out_t.append("0"); out_i.append(nxt_wi if nxt_wi is not None else (wi if wi is not None else 0))
                else:
                    out_t.extend(["0"] * count)
                i += 2
                continue

        if re.fullmatch(r"0{2,}", t):
            count = len(t)
            if out_i is not None:
                out_t.extend(["0"] * count)
                out_i.extend([wi] * count)  # same original word idx
            else:
                out_t.extend(["0"] * count)
            i += 1
            continue

        if out_i is not None:
            out_t.append(t); out_i.append(wi)
        else:
            out_t.append(t)
        i += 1

    return out_t, out_i


def build_transcript_index(transcript_words: Sequence[str]) -> TranscriptIndex:
    expanded_tokens: List[str] = []
    token_to_word: List[int] = []

    for wi, w in enumerate(transcript_words):
        w = normalise_word(w)
        if not w:
            continue
        parts = _pm_split_alnum(w)  # "3step" -> ["3","step"]
        if not parts:
            continue
        for p in parts:
            expanded_tokens.append(_pm_canon_single(p))
            token_to_word.append(wi)

    expanded_tokens, token_to_word = _pm_canon_seq(expanded_tokens, token_to_word)
    return TranscriptIndex(tuple(transcript_words), tuple(expanded_tokens), tuple(token_to_word))


def _pm_phrase_to_tokens(phrase: str) -> List[str]:
    text = phrase.lower()
    text = _HYPHENS_RE.sub(" ", text)                  # "self-heimlich" -> "self heimlich"
    text = re.sub(r"\bper\s+cent\b", "percent", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)  # fixes your old \b issue
    text = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)", r"\1 dollars", text)

    raw = [_pm_basic_norm(t) for t in text.split()]
    raw = [t for t in raw if t]

    expanded: List[str] = []
    for t in raw:
        parts = _pm_split_alnum(t)
        expanded.extend(parts if parts else [t])

    expanded = [_pm_canon_single(t) for t in expanded]
    expanded, _ = _pm_canon_seq(expanded, None)
    return expanded


def _pm_apply_bigram_splits(
    tokens: Sequence[str],
    token_to_word: Sequence[int],
    phrase_tokens: Sequence[str],
) -> Tuple[List[str], List[int]]:
    """
    Phrase-aware un-merge:
      if transcript token == concat(phrase_token[i] + phrase_token[i+1]),
      split transcript token into those two tokens.

    Examples:
      "selfheimlich" -> "self" "heimlich"
      "welfarecheck" -> "welfare" "check"
    """
    join_map: Dict[str, Tuple[str, str]] = {}
    for i in range(len(phrase_tokens) - 1):
        a, b = phrase_tokens[i], phrase_tokens[i + 1]
        if len(a) >= 3 and len(b) >= 3 and a.isalpha() and b.isalpha():
            join_map[a + b] = (a, b)

    if not join_map:
        return list(tokens), list(token_to_word)

    out_t: List[str] = []
    out_i: List[int] = []
    for t, wi in zip(tokens, token_to_word):
        split = join_map.get(t)
        if split:
            out_t.extend(split)
            out_i.extend([wi, wi])
        else:
            out_t.append(t)
            out_i.append(wi)
    return out_t, out_i


def _pm_align_substring(
    phrase_tokens: List[str],
    transcript_tokens: Sequence[str],
    *,
    start_pos: int = 0,
    good_score: float,
    panic_score: float,
    panic_requires_anchors: bool,
) -> Optional[Tuple[int, int, float, int]]:
    """
    Token-level edit-distance substring match.
    Returns: (tok_start, tok_end, score, anchor_hits)

    Selection policy:
      1) earliest match with score >= good_score
      2) else earliest match with score >= panic_score (and anchors if required)
      3) else best-scoring match (may fail later in thresholding)
    """
    m = len(phrase_tokens)
    if m == 0 or start_pos >= len(transcript_tokens):
        return None

    prefer_anchor_hits = (
        m <= 6
        and any(pt not in _MATCH_STOPWORDS and len(pt) >= 3 for pt in phrase_tokens)
    )
    anchor_tiebreak_cost_eps = 0.10

    T = transcript_tokens[start_pos:]
    n = len(T)
    if n == 0:
        return None

    INS_BASE = 0.80  # skipping transcript tokens
    DEL_BASE = 0.90  # dropping phrase tokens

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    ptr = [[0] * (n + 1) for _ in range(m + 1)]  # 1=diag,2=left,3=up

    for i in range(1, m + 1):
        pt = phrase_tokens[i - 1]
        del_cost = 0.20 if pt in _MATCH_STOPWORDS else DEL_BASE
        dp[i][0] = dp[i - 1][0] + del_cost
        ptr[i][0] = 3

    for j in range(1, n + 1):
        dp[0][j] = 0.0  # substring can start anywhere “for free”

    for i in range(1, m + 1):
        pt = phrase_tokens[i - 1]
        del_cost = 0.20 if pt in _MATCH_STOPWORDS else DEL_BASE
        for j in range(1, n + 1):
            tt = T[j - 1]
            sim = _pm_token_sim(pt, tt)
            sub_cost = 1.0 - sim
            ins_cost = 0.30 if tt in _MATCH_STOPWORDS else INS_BASE

            c_diag = dp[i - 1][j - 1] + sub_cost
            c_left = dp[i][j - 1] + ins_cost
            c_up   = dp[i - 1][j] + del_cost

            best, move = c_diag, 1
            if c_left < best:
                best, move = c_left, 2
            if c_up < best:
                best, move = c_up, 3

            dp[i][j] = best
            ptr[i][j] = move

    def _backtrack(end_j: int) -> Optional[Tuple[int, int, int]]:
        i, j = m, end_j
        anchors = 0
        while i > 0 and j > 0:
            move = ptr[i][j]
            if move == 1:
                pt = phrase_tokens[i - 1]
                tt = T[j - 1]
                sim = _pm_token_sim(pt, tt)
                if pt not in _MATCH_STOPWORDS and sim >= 0.90 and len(pt) >= 3:
                    anchors += 1
                i -= 1
                j -= 1
            elif move == 2:
                j -= 1
            else:
                i -= 1

        tok_start = j
        tok_end = end_j - 1

        # IMPORTANT: prevent empty-span “matches” (was causing min() empty sequence)
        if tok_end < tok_start:
            return None

        return tok_start, tok_end, anchors

    def _score_for_end(end_j: int) -> float:
        cost = dp[m][end_j]
        return max(0.0, 1.0 - cost / float(m))

    good_cost_max = float(m) * (1.0 - float(good_score))
    panic_cost_max = float(m) * (1.0 - float(panic_score))

    def _best_under(cost_max: float, require_anchors: bool) -> Optional[Tuple[int,int,float,int]]:
        best = None  # (cost, end_j, tok0, tok1, anchors)
        for end_j in range(1, n + 1):
            cost = dp[m][end_j]
            if cost > cost_max:
                continue
            bt = _backtrack(end_j)
            if bt is None:
                continue
            tok0, tok1, anchors = bt
            if require_anchors and anchors == 0:
                continue
            cand = (cost, end_j, tok0, tok1, anchors)
            if best is None:
                best = cand
                continue

            if not prefer_anchor_hits:
                if cand < best:
                    best = cand
                continue

            best_cost, best_end_j, best_tok0, best_tok1, best_anchors = best
            if cost < best_cost - anchor_tiebreak_cost_eps:
                best = cand
                continue

            if abs(cost - best_cost) <= anchor_tiebreak_cost_eps:
                if anchors > best_anchors:
                    best = cand
                elif anchors == best_anchors and (end_j, tok0, tok1) < (
                    best_end_j,
                    best_tok0,
                    best_tok1,
                ):
                    best = cand
        if best is None:
            return None
        cost, end_j, tok0, tok1, anchors = best
        return tok0 + start_pos, tok1 + start_pos, _score_for_end(end_j), anchors

    res = _best_under(good_cost_max, require_anchors=False)
    if res is not None:
        return res

    res = _best_under(panic_cost_max, require_anchors=panic_requires_anchors)
    if res is not None:
        return res
    # Pass 3: best overall (for error reporting / strict reject)
    best_j = min(range(1, n + 1), key=lambda j: (dp[m][j], j))
    bt = _backtrack(best_j)
    if bt is None:
        return None
    tok0, tok1, anchors = bt
    return tok0 + start_pos, tok1 + start_pos, _score_for_end(best_j), anchors

def find_phrase_indices_windowed(
    tindex: TranscriptIndex,
    phrase: str,
    *,
    occurrence: int,
    start_index: int,
    backtrack_words: int = 12,
    lookahead_words: int = 40,
) -> tuple[int, int]:
    w0 = max(0, int(start_index) - backtrack_words)
    w1 = min(len(tindex.word_tokens), int(start_index) + lookahead_words)

    sub_words = list(tindex.word_tokens[w0:w1])          # already normalised
    sub_index = build_transcript_index(sub_words)

    sw, ew = find_phrase_indices(
        sub_index,
        phrase,
        occurrence=occurrence,
        start_index=max(0, int(start_index) - w0),
        strict=False,
    )
    return w0 + sw, w0 + ew



def find_phrase_indices(
    transcript_words_or_index: Union[Sequence[str], TranscriptIndex],
    phrase: str,
    occurrence: int = 1,
    start_index: int = 0,
    *,
    strict: bool = False,
) -> Tuple[int, int]:
    """
    Robust phrase-to-transcript matcher.

    strict=False:
      - accepts “good” matches
      - also accepts “panic” matches if at least one anchor token matched strongly (or phrase is numeric)
      - avoids pipeline-killing crashes on small ASR deviations

    strict=True:
      - only accepts “good” matches; otherwise raises ValueError
    """
    if not phrase:
        raise ValueError("Phrase must be provided when start/end indices are omitted.")

    tindex = transcript_words_or_index if isinstance(transcript_words_or_index, TranscriptIndex) else build_transcript_index(transcript_words_or_index)

    phrase_tokens = _pm_phrase_to_tokens(phrase)
    if not phrase_tokens:
        raise ValueError("Phrase must contain at least one word.")

    # Phrase-aware unmerge
    T, M = _pm_apply_bigram_splits(tindex.tokens, tindex.token_to_word, phrase_tokens)

    has_numeric = any(t.isdigit() for t in phrase_tokens)
    m = len(phrase_tokens)
    good = 0.72 if m >= 4 else 0.82
    panic = 0.55 if m >= 4 else 0.65
    # 2-word phrases often lose 1 token in ASR ("Airway clear" -> just "airway")
    if m <= 2:
        panic = min(panic, 0.54)  # allow the ~0.55 one-token-hit case
    panic_requires_anchors = not has_numeric
    lookback = 10 if not has_numeric else 30
    word_start_limit = max(0, int(start_index) - lookback)
    start_pos = bisect.bisect_left(M, word_start_limit)

    want = max(1, int(occurrence))
    cur_pos = start_pos
    last: Optional[Tuple[int, int, float, int]] = None

    for _ in range(want):
        res = _pm_align_substring(
            phrase_tokens,
            T,
            start_pos=cur_pos,
            good_score=good,
            panic_score=panic,
            panic_requires_anchors=panic_requires_anchors,
        )
        if res is None:
            last = None
            break
        tok0, tok1, score, anchors = res
        last = (tok0, tok1, score, anchors)
        cur_pos = tok1 + 1
        if cur_pos >= len(T):
            break

    if last is None:
        raise ValueError(
            f"Unable to align phrase {phrase!r} (occurrence {want}) starting at start_index={start_index}. "
            f"No candidate window was available (word_start_limit={word_start_limit}, start_pos={start_pos}, "
            f"transcript_tokens={len(T)}, transcript_words={len(tindex.word_tokens)}). "
            "This usually means start_index points past the phrase (it may occur earlier), or there are no tokens left to search."
        )


    tok0, tok1, score, anchors = last
    word_span = M[tok0 : tok1 + 1]
    w0, w1 = min(word_span), max(word_span)

    if score < good:
        if (not strict) and (score >= panic) and (anchors > 0 or has_numeric):
            logger.warning(
                "[MATCHING][ALIGN] best-effort accept: phrase=%r score=%.3f anchors=%d words=(%d,%d) tokens=%s",
                phrase, score, anchors, w0, w1, phrase_tokens
            )
            return w0, w1

        candidate_snippet = " ".join(
            tindex.word_tokens[max(0, w0 - 3) : min(len(tindex.word_tokens), w1 + 4)]
        )

        raise ValueError(
            f"Unable to align phrase {phrase!r} (occurrence {want}) to transcript with sufficient confidence "
            f"(best_score={score:.3f} < required={good:.2f}, anchors={anchors}, start_index={start_index}). "
            "The phrase may exist in the audio, but the ASR transcript/normalization differs (missing/extra words, "
            "pluralization, merged/split tokens), or an earlier ambiguous match moved start_index past the right location. "
            f"Best_candidate_words=({w0},{w1}) candidate_snippet={candidate_snippet!r} phrase_tokens={phrase_tokens}"
        )

    logger.info(
        "[MATCHING][ALIGN] phrase=%r score=%.3f anchors=%d -> words=(%d,%d) tokens=%s",
        phrase, score, anchors, w0, w1, phrase_tokens
    )
    return w0, w1


def map_assignments_to_segments(
    transcript: List[Dict[str, float]],
    assignments: Sequence[HighlightAssignment],
) -> List[Dict[str, Optional[object]]]:
    """Convert user highlight selections into rendering segments.

    We want robust behaviour with WhisperX drift and trimmed uploads:
    - Prefer phrase-driven matching into the WhisperX transcript (best timing).
    - Never abort the whole render if a phrase can't be found.
    - If indices are provided but appear to be in a longer "script" coordinate
      system (common in batch/templated workflows), repair them using anchors from
      phrases that *did* match, falling back to a global scale+clamp.
    """

    if not transcript or not assignments:
        return []

    transcript_len = len(transcript)
    last_idx = transcript_len - 1
    if last_idx < 0:
        return []

    normalised_transcript = [normalise_word(entry["word"]) for entry in transcript]
    transcript_index = build_transcript_index(normalised_transcript)

    def _lookahead_for_phrase(text: str) -> int:
        tokens = _pm_phrase_to_tokens(text or "")
        n_tokens = len(tokens)
        if n_tokens <= 2:
            return 70
        return max(90, min(240, n_tokens * 6))

    def _safe_int(value: Optional[object]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _phrase_token_count(text: str) -> int:
        tokens = [normalise_word(tok) for tok in (text or "").split()]
        tokens = [tok for tok in tokens if tok]
        return max(1, len(tokens))

    def _build_hint_mapper(
        anchors: Sequence[Tuple[int, int]],
        max_hint_index: Optional[int],
    ):
        """Return a monotonic mapper from hint-index space -> transcript-index space."""

        points: List[Tuple[int, int]] = []
        for hint_idx, trans_idx in anchors:
            if hint_idx < 0:
                continue
            if trans_idx < 0:
                continue
            points.append((int(hint_idx), int(trans_idx)))

        if max_hint_index is not None and max_hint_index > 0:
            points.append((0, 0))
            points.append((max_hint_index, last_idx))

        if not points:
            return None

        # Sort and dedupe.
        points = sorted(set(points), key=lambda p: (p[0], p[1]))

        # Enforce monotonicity in transcript indices.
        filtered: List[Tuple[int, int]] = []
        last_y = -1
        for x, y in points:
            if y < last_y:
                continue
            filtered.append((x, y))
            last_y = y

        if len(filtered) < 2:
            return None

        xs = [p[0] for p in filtered]

        def _map(hint_index: int) -> int:
            x = int(hint_index)
            if x <= filtered[0][0]:
                x0, y0 = filtered[0]
                x1, y1 = filtered[1]
                if x1 == x0:
                    return y0
                slope = (y1 - y0) / float(x1 - x0)
                return int(round(y0 + (x - x0) * slope))

            if x >= filtered[-1][0]:
                x0, y0 = filtered[-2]
                x1, y1 = filtered[-1]
                if x1 == x0:
                    return y1
                slope = (y1 - y0) / float(x1 - x0)
                return int(round(y1 + (x - x1) * slope))

            j = bisect_right(xs, x) - 1
            x0, y0 = filtered[j]
            x1, y1 = filtered[j + 1]
            if x1 == x0:
                return y0
            t = (x - x0) / float(x1 - x0)
            return int(round(y0 + t * (y1 - y0)))

        return _map

    def _repair_fallback_indices(
        start_word_hint: Optional[object],
        end_word_hint: Optional[object],
        phrase_text: str,
        *,
        min_start: int,
        max_end: Optional[int],
        max_hint_index: Optional[int],
        hint_mapper,
    ) -> Optional[Tuple[int, int]]:
        start_idx = _safe_int(start_word_hint)
        end_idx = _safe_int(end_word_hint)
        if start_idx is None or end_idx is None:
            return None

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        approx_words = _phrase_token_count(phrase_text)

        if hint_mapper is not None:
            start_idx = hint_mapper(start_idx)
            end_idx = hint_mapper(end_idx)
        elif (
            max_hint_index is not None
            and max_hint_index > 0
            and max_hint_index >= transcript_len
            and (start_idx >= transcript_len or end_idx >= transcript_len)
        ):
            scale = last_idx / float(max_hint_index)
            start_idx = int(round(start_idx * scale))
            end_idx = int(round(end_idx * scale))

        # Clamp into range.
        start_idx = max(0, min(start_idx, last_idx))
        end_idx = max(0, min(end_idx, last_idx))
        if end_idx < start_idx:
            end_idx = start_idx

        if max_end is not None:
            max_cap = int(max_end)
            if max_cap < 0:
                return None
            max_cap = max(0, min(max_cap, last_idx))
            if start_idx > max_cap:
                return None
            if end_idx > max_cap:
                end_idx = max_cap
                if end_idx < start_idx:
                    end_idx = start_idx

        # If the mapped hint span collapsed (common when the hint indices come from a
        # different transcript), pad it to a reasonable minimum so overlays aren't
        # single-word blips.
        min_span_words = min(approx_words, 12)
        if min_span_words > 1 and (end_idx - start_idx + 1) < min_span_words:
            end_idx = min(last_idx, start_idx + min_span_words - 1)

        # If we collapsed onto the very end (common when the video cuts early),
        # keep an approximate span length instead of a 0-length overlay.
        if start_idx == last_idx and approx_words > 1:
            start_idx = max(0, last_idx - approx_words + 1)

        # Avoid jumping far backwards relative to the current cursor.
        if start_idx < min_start:
            start_idx = min_start
            if end_idx < start_idx:
                end_idx = start_idx
            if end_idx == start_idx and approx_words > 1:
                end_idx = min(last_idx, start_idx + approx_words - 1)

        return start_idx, end_idx

    # Compute max hint index (used for global scaling and endpoint anchoring).
    max_hint_index: Optional[int] = None
    for assignment in assignments:
        for hint in (assignment.start_word, assignment.end_word):
            hint_int = _safe_int(hint)
            if hint_int is None:
                continue
            max_hint_index = hint_int if max_hint_index is None else max(max_hint_index, hint_int)

    # Pass 1: phrase matching only. Missing phrases must NOT poison the cursor.
    provisional_matches: List[Optional[Tuple[int, int]]] = [None] * len(assignments)
    failure_reasons: List[Optional[Exception]] = [None] * len(assignments)
    search_start = 0

    for idx, assignment in enumerate(assignments):
        phrase_text = (assignment.phrase or "").strip()
        if not phrase_text:
            continue

        try:
            resolved = find_phrase_indices_windowed(
                transcript_index,
                phrase_text,
                occurrence=assignment.occurrence,
                start_index=search_start,
                backtrack_words=16,
                lookahead_words=_lookahead_for_phrase(phrase_text),
            )
        except ValueError as exc:
            resolved = None
            for backtrack_words, min_lookahead in ((32, 220), (48, 320)):
                try:
                    resolved = find_phrase_indices_windowed(
                        transcript_index,
                        phrase_text,
                        occurrence=assignment.occurrence,
                        start_index=search_start,
                        backtrack_words=backtrack_words,
                        lookahead_words=max(_lookahead_for_phrase(phrase_text), min_lookahead),
                    )
                    break
                except ValueError:
                    resolved = None

            if resolved is None:
                failure_reasons[idx] = exc
                continue

        provisional_matches[idx] = resolved
        search_start = max(search_start, int(resolved[1]) + 1)

    # Build anchors between (hint indices) and (resolved transcript indices).
    anchors: List[Tuple[int, int]] = []
    for idx, assignment in enumerate(assignments):
        resolved = provisional_matches[idx]
        if not resolved:
            continue
        hint_start = _safe_int(assignment.start_word)
        hint_end = _safe_int(assignment.end_word)
        if hint_start is not None:
            anchors.append((hint_start, int(resolved[0])))
        if hint_end is not None:
            anchors.append((hint_end, int(resolved[1])))

    hint_mapper = _build_hint_mapper(anchors, max_hint_index)

    # Pass 2: emit segments, using phrase matches when available and repairing indices otherwise.
    mapped: List[Dict[str, Optional[object]]] = []
    search_start = 0

    def _next_matched_start(min_start: int, start_idx: int) -> Optional[int]:
        """Return the next provisional match start at/after ``min_start`` if any."""
        anchor = int(min_start)
        for future_idx in range(start_idx, len(assignments)):
            resolved_future = provisional_matches[future_idx]
            if not resolved_future:
                continue
            future_start = int(resolved_future[0])
            if future_start >= anchor:
                return future_start
        return None

    for idx, assignment in enumerate(assignments):
        phrase_text = (assignment.phrase or "").strip()
        resolved = provisional_matches[idx]
        resolved_from_phrase = resolved is not None

        if resolved_from_phrase:
            start_word, end_word = int(resolved[0]), int(resolved[1])
        else:
            min_start = max(0, int(search_start) - 8)
            used_monotonic_fallback = False
            next_start = _next_matched_start(search_start, idx + 1)
            max_end = None
            if next_start is not None:
                max_end = int(next_start) - 1
                if max_end < min_start:
                    max_end = min_start
            repaired = _repair_fallback_indices(
                assignment.start_word,
                assignment.end_word,
                phrase_text,
                min_start=min_start,
                max_end=max_end,
                max_hint_index=max_hint_index,
                hint_mapper=hint_mapper,
            )

            if repaired is None:
                if phrase_text:
                    # Final fallback: keep overlays monotonic even if the phrase isn't present
                    # in the ASR transcript by allocating a small span before the next known
                    # segment (if any). This avoids long "main video" gaps and stuck subtitles.
                    fallback_start = min(max(int(search_start), 0), last_idx)
                    fallback_end_cap = last_idx
                    if next_start is not None:
                        if next_start > fallback_start:
                            fallback_end_cap = min(fallback_end_cap, next_start - 1)
                        else:
                            fallback_end_cap = min(fallback_end_cap, fallback_start)

                    approx_words = _phrase_token_count(phrase_text)
                    min_span_words = min(approx_words, 12)
                    fallback_end = min(
                        fallback_end_cap, fallback_start + min_span_words - 1
                    )

                    min_duration_s = 0.75
                    while (
                        fallback_end + 1 < len(transcript)
                        and fallback_end < fallback_end_cap
                        and transcript[fallback_end]["end_time"]
                        - transcript[fallback_start]["start_time"]
                        < min_duration_s
                    ):
                        fallback_end += 1

                    logger.warning(
                        "[MAP] %s; falling back to monotonic span words[%d:%d] for clip=%s phrase=%r",
                        failure_reasons[idx]
                        or ValueError(
                            f"Phrase '{phrase_text}' occurrence {assignment.occurrence} not found in transcript."
                        ),
                        fallback_start,
                        fallback_end,
                        getattr(assignment, "clip_path", None),
                        phrase_text,
                    )
                    start_word, end_word = fallback_start, fallback_end
                    repaired = (start_word, end_word)
                    used_monotonic_fallback = True
                else:
                    logger.warning(
                        "[MAP] Skipping assignment with no phrase/indices for clip=%s",
                        getattr(assignment, "clip_path", None),
                    )
                    continue

            if phrase_text and not used_monotonic_fallback:
                logger.warning(
                    "[MAP] %s; falling back to repaired explicit indices for clip=%s phrase=%r",
                    failure_reasons[idx]
                    or ValueError(
                        f"Phrase '{phrase_text}' occurrence {assignment.occurrence} not found in transcript."
                    ),
                    getattr(assignment, "clip_path", None),
                    phrase_text,
                )
            start_word, end_word = repaired

        # Clamp into range (never crash).
        start_word = max(0, min(int(start_word), last_idx))
        end_word = max(0, min(int(end_word), last_idx))
        if end_word < start_word:
            end_word = start_word

        # Tighten the span only when we truly matched by phrase.
        if phrase_text and resolved_from_phrase:
            phrase_tokens = [
                normalise_word(tok) for tok in phrase_text.split()
                if normalise_word(tok)
            ]

            if phrase_tokens:
                first_token = phrase_tokens[0]
                last_token = phrase_tokens[-1]

                segment_tokens = normalised_transcript[start_word : end_word + 1]
                try:
                    offset = segment_tokens.index(first_token)
                except ValueError:
                    offset = None

                if offset is not None:
                    new_start = start_word + offset
                    if new_start <= end_word:
                        start_word = new_start

                segment_tokens = normalised_transcript[start_word : end_word + 1]
                if last_token not in segment_tokens:
                    if last_token in _PHRASE_STOPWORDS or len(last_token) < 3:
                        j_limit = end_word + 1
                    else:
                        j_limit = min(len(normalised_transcript), end_word + 1 + 14)

                    for j in range(end_word + 1, j_limit):
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

                # Targeted boundary-integrity repair for short spans.
                #
                # For short (1–2s-ish) highlight segments, even a 1-word boundary
                # slip is very noticeable. The robust aligner is intentionally
                # permissive (stopwords are cheap to drop) to avoid pipeline-killing
                # crashes, so for short spans we add a conservative, warning-only
                # attempt to pull the span outward when we see strong evidence that
                # the boundary token(s) sit immediately adjacent to the match.
                try:
                    seg_duration_s = float(transcript[end_word]["end_time"]) - float(
                        transcript[start_word]["start_time"]
                    )
                except Exception:
                    seg_duration_s = None

                if (
                    seg_duration_s is not None
                    and seg_duration_s <= 2.5
                    and len(phrase_tokens) <= 8
                ):
                    segment_tokens = normalised_transcript[start_word : end_word + 1]

                    def _first_non_stop(tokens: List[str]) -> Optional[str]:
                        for t in tokens:
                            if t and t not in _PHRASE_STOPWORDS and len(t) >= 3:
                                return t
                        return None

                    def _last_non_stop(tokens: List[str]) -> Optional[str]:
                        for t in reversed(tokens):
                            if t and t not in _PHRASE_STOPWORDS and len(t) >= 3:
                                return t
                        return None

                    first_sig = _first_non_stop(phrase_tokens)
                    last_sig = _last_non_stop(phrase_tokens)
                    first_long = next((t for t in phrase_tokens if len(t) >= 3), None)
                    last_long = next(
                        (t for t in reversed(phrase_tokens) if len(t) >= 3), None
                    )

                    missing = [
                        t
                        for t in (first_sig, last_sig, first_long, last_long)
                        if t and t not in segment_tokens
                    ]

                    if missing:
                        orig_start_word, orig_end_word = start_word, end_word
                        logger.warning(
                            "[MAP][BOUNDARY] short-span mismatch; phrase=%r words[%d:%d] dur=%.3fs missing=%s snippet=%r",
                            phrase_text,
                            orig_start_word,
                            orig_end_word,
                            seg_duration_s,
                            missing,
                            " ".join(
                                entry["word"]
                                for entry in transcript[
                                    orig_start_word : orig_end_word + 1
                                ]
                            ),
                        )

                        # Expand backward by matching the contiguous leading
                        # long-token prefix immediately before start_word.
                        if first_long and first_long not in segment_tokens:
                            prefix_long: List[str] = []
                            for t in phrase_tokens:
                                if len(t) >= 3:
                                    prefix_long.append(t)
                                    if len(prefix_long) >= 2:
                                        break
                                else:
                                    break

                            if prefix_long and start_word > 0:
                                for delta in range(len(prefix_long), 0, -1):
                                    cand_start = start_word - delta
                                    if cand_start < 0:
                                        continue
                                    if normalised_transcript[cand_start:start_word] == prefix_long[
                                        :delta
                                    ]:
                                        start_word = cand_start
                                        break

                        # Expand forward by matching the contiguous trailing
                        # long-token suffix immediately after end_word.
                        segment_tokens = normalised_transcript[start_word : end_word + 1]
                        if last_long and last_long not in segment_tokens:
                            suffix_long: List[str] = []
                            for t in reversed(phrase_tokens):
                                if len(t) >= 3:
                                    suffix_long.append(t)
                                    if len(suffix_long) >= 2:
                                        break
                                else:
                                    break
                            suffix_long = list(reversed(suffix_long))

                            if suffix_long and end_word + 1 < len(normalised_transcript):
                                for delta in range(len(suffix_long), 0, -1):
                                    cand_end = end_word + delta
                                    if cand_end >= len(normalised_transcript):
                                        continue
                                    expected = suffix_long[-delta:]
                                    if (
                                        normalised_transcript[end_word + 1 : cand_end + 1]
                                        == expected
                                    ):
                                        end_word = cand_end
                                        break

                        # Final (strict) rescue: if the full phrase appears as an
                        # exact contiguous sequence nearby, snap to it.
                        segment_tokens = normalised_transcript[start_word : end_word + 1]
                        still_missing_sig = (
                            (first_sig and first_sig not in segment_tokens)
                            or (last_sig and last_sig not in segment_tokens)
                        )
                        if still_missing_sig and phrase_tokens:
                            window_pad = 24
                            win_lo = max(0, orig_start_word - window_pad)
                            win_hi = min(
                                len(normalised_transcript) - len(phrase_tokens),
                                orig_end_word + window_pad,
                            )
                            best_exact = None
                            for j in range(win_lo, win_hi + 1):
                                if (
                                    normalised_transcript[j : j + len(phrase_tokens)]
                                    == phrase_tokens
                                ):
                                    cand = (abs(j - orig_start_word), j)
                                    if best_exact is None or cand < best_exact:
                                        best_exact = cand
                            if best_exact is not None:
                                _, j = best_exact
                                start_word = j
                                end_word = j + len(phrase_tokens) - 1

                        if (start_word, end_word) != (orig_start_word, orig_end_word):
                            logger.warning(
                                "[MAP][BOUNDARY] adjusted short-span; phrase=%r words[%d:%d] -> words[%d:%d]",
                                phrase_text,
                                orig_start_word,
                                orig_end_word,
                                start_word,
                                end_word,
                            )

        start_word = max(0, min(start_word, last_idx))
        end_word = max(0, min(end_word, last_idx))
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

        # Keep a monotonic cursor for later repairs.
        search_start = max(search_start, end_word + 1)

    # Ensure segments are returned in time order. This makes overlay-to-overlay
    # continuation logic stable even when the clip mapping file is out of order or
    # when a phrase match required a small backtrack.
    mapped.sort(
        key=lambda seg: (
            int(seg.get("start_word", 0) or 0),
            int(seg.get("end_word", 0) or 0),
            str(seg.get("clip_path") or ""),
        )
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
    transcript_index = build_transcript_index(normalised_transcript)
    mapped: List[Dict[str, object]] = []
    search_start = 0

    def _lookahead_for_phrase(text: str) -> int:
        tokens = _pm_phrase_to_tokens(text or "")
        n_tokens = len(tokens)
        if n_tokens <= 2:
            return 80
        return max(110, min(260, n_tokens * 6))

    def _find_next_sentence_start(anchor_start: int, start_idx: int) -> Optional[int]:
        """Return the next sentence start index at/after ``anchor_start`` if any.

        Used to avoid consuming transcript words for subtitle sentences that can't
        be matched (e.g. non-spoken callouts like "50% off."). When we know where
        the next real match begins, we can keep fallbacks within that gap so we
        don't break alignment for later lines.
        """

        max_lookahead = 15
        end_idx = min(len(sentences), start_idx + max_lookahead)
        for future in sentences[start_idx:end_idx]:
            future_text = (future.text or "").strip()
            if not future_text:
                continue

            if future.start_word is not None and future.end_word is not None:
                try:
                    future_start = int(future.start_word)
                except (TypeError, ValueError):
                    future_start = None
                if future_start is not None and future_start >= anchor_start:
                    return future_start
                continue

            future_phrase = (future.phrase or future.text).strip()
            if not future_phrase:
                continue

            future_occurrence = max(1, int(future.occurrence or 1))
            try:
                future_match = find_phrase_indices_windowed(
                    transcript_index,
                    future_phrase,
                    occurrence=future_occurrence,
                    start_index=anchor_start,
                    backtrack_words=12,
                    lookahead_words=240,
                )
            except ValueError:
                continue

            if future_match:
                return int(future_match[0])

        return None

    for sentence_idx, sentence in enumerate(sentences):
        original_search_start = search_start
        text = sentence.text.strip()
        if not text:
            continue
        phrase = (sentence.phrase or sentence.text).strip()
        start_word = sentence.start_word
        end_word = sentence.end_word
        advance_search = True

        if start_word is not None and end_word is not None:
            start_word = int(start_word)
            end_word = int(end_word)
        else:
            target_occurrence = max(1, int(sentence.occurrence or 1))
            try:
                start_word, end_word = find_phrase_indices_windowed(
                    transcript_index,
                    phrase,
                    occurrence=target_occurrence,
                    start_index=search_start,
                    backtrack_words=20,
                    lookahead_words=_lookahead_for_phrase(phrase),
                )
            except ValueError as exc:
                # When the requested subtitle sentence does not appear in the transcript
                # (common with ASR drift or when subtitles include non-spoken text),
                # fall back to a small monotonic span instead of aborting the render.
                logger.warning("[SUBTITLE MAP] %s", exc)

                fallback_start = min(max(search_start, 0), len(transcript) - 1)
                fallback_end_cap = len(transcript) - 1

                next_start = _find_next_sentence_start(search_start, sentence_idx + 1)
                if next_start is not None:
                    if next_start > fallback_start:
                        fallback_end_cap = min(fallback_end_cap, next_start - 1)
                    else:
                        # No gap before the next sentence. Keep the cursor in place
                        # so later sentences can still match correctly.
                        fallback_end_cap = min(fallback_end_cap, max(fallback_start - 1, 0))
                        if fallback_start > 0:
                            fallback_start = fallback_end_cap
                        else:
                            advance_search = False

                approx_tokens = [normalise_word(tok) for tok in phrase.split()]
                approx_tokens = [tok for tok in approx_tokens if tok]
                approx_words = max(1, len(approx_tokens))
                fallback_end = min(fallback_end_cap, fallback_start + approx_words - 1)

                min_duration_s = 0.85
                while (
                    fallback_end + 1 < len(transcript)
                    and fallback_end < fallback_end_cap
                    and transcript[fallback_end]["end_time"]
                    - transcript[fallback_start]["start_time"]
                    < min_duration_s
                ):
                    fallback_end += 1

                start_word, end_word = fallback_start, fallback_end

        if start_word < 0 or end_word >= len(transcript) or start_word > end_word:
            logger.warning(
                "[SUBTITLE MAP] Invalid indices resolved for %r: (%s, %s). Clamping.",
                sentence.text,
                start_word,
                end_word,
            )
            last_idx = len(transcript) - 1
            start_word = max(0, min(int(start_word), last_idx))
            end_word = max(0, min(int(end_word), last_idx))
            if end_word < start_word:
                end_word = start_word

        mapped.append(
            {
                "start_word": start_word,
                "end_word": end_word,
                "text": text,
            }
        )

        if advance_search:
            search_start = max(search_start, end_word + 1)
        else:
            search_start = original_search_start

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

    # Scale subtitle sizing relative to the output resolution.
    #
    # The SubtitleDesign defaults are tuned for "standard" exports:
    # - 4:5  -> 1080x1350
    # - 9:16 -> 1080x1920
    #
    # This renderer preserves relative sizing when the input video is smaller/larger
    # by scaling pixel-based parameters against the current frame height.
    is_9_16 = False
    if aspect_ratio in ("4:5", "9:16"):
        is_9_16 = aspect_ratio == "9:16"
    else:
        # Backward-compatible heuristic (call sites historically didn't pass aspect_ratio).
        is_9_16 = design.bottom_margin >= 300

    reference_height = 1920 if is_9_16 else 1350
    scale = 1.0
    if reference_height > 0 and height > 0:
        scale = height / float(reference_height)
    # Guard against extreme values on tiny/huge videos.
    scale = max(0.25, min(3.0, float(scale)))

    def _scale_int(value: int, *, min_value: int = 0) -> int:
        try:
            scaled = int(round(float(value) * scale))
        except Exception:
            scaled = int(value)
        return max(min_value, scaled)

    def _scale_signed_int(value: int) -> int:
        try:
            return int(round(float(value) * scale))
        except Exception:
            return int(value)

    def _scale_tuple_int(values: Tuple[int, int], *, min_value: int = 0) -> Tuple[int, int]:
        return (
            _scale_int(int(values[0]), min_value=min_value),
            _scale_int(int(values[1]), min_value=min_value),
        )

    def _scale_tuple_signed(values: Tuple[int, int]) -> Tuple[int, int]:
        return (
            _scale_signed_int(int(values[0])),
            _scale_signed_int(int(values[1])),
        )

    effective_font_size_px = _scale_int(int(design.font_size_px), min_value=14)
    effective_text_scale = max(0.3, float(design.text_scale) * float(scale))
    effective_text_thickness = _scale_int(int(design.text_thickness), min_value=1)
    effective_outline_thickness = (
        _scale_int(int(design.outline_thickness), min_value=1)
        if int(design.outline_thickness) > 0
        else 0
    )
    effective_line_spacing = _scale_int(int(design.line_spacing), min_value=0)
    effective_bottom_margin = _scale_int(int(design.bottom_margin), min_value=0)
    effective_corner_radius = _scale_int(int(design.corner_radius), min_value=0)
    effective_margin_x = _scale_int(int(getattr(design, "margin_x", design.margin)), min_value=0)
    effective_margin_y = _scale_int(int(getattr(design, "margin_y", design.margin)), min_value=0)
    effective_highlight_padding = _scale_tuple_int(design.highlight_padding, min_value=0)
    effective_box_shadow_offset = _scale_tuple_signed(
        getattr(design, "box_shadow_offset", (0, 0))
    )
    effective_box_shadow_blur = _scale_int(
        int(getattr(design, "box_shadow_blur", 0)), min_value=0
    )

    active_segment_index: Optional[int] = None
    if subtitle_segments:
        previous_candidate: Optional[int] = None
        for idx, (seg_start, seg_end) in enumerate(subtitle_segments):
            start_t = float(transcript[seg_start]["start_time"])
            end_t = float(transcript[seg_end]["end_time"])

            if current_time < start_t:
                break
            previous_candidate = idx

            # NOTE: don't break; later overlapping segments should win
            if current_time <= end_t:
                active_segment_index = idx

        # keep previous subtitle until the next starts (your existing behavior)
        if active_segment_index is None and previous_candidate is not None:
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
        pil_font = get_pil_font(design.font_path, int(effective_font_size_px))
        pil_ascent, pil_descent = pil_font.getmetrics()
        default_line_height = pil_ascent + pil_descent
        stroke_px = int(effective_outline_thickness) if (is_9_16 and effective_outline_thickness > 0) else 0

        def measure_word(text: str) -> Tuple[int, int, int]:
            render_text = text if text else " "
            try:
                bbox = pil_font.getbbox(render_text, stroke_width=stroke_px)
            except TypeError:
                bbox = pil_font.getbbox(render_text)
            width = int(math.ceil(bbox[2] - bbox[0]))
            height = int(math.ceil(bbox[3] - bbox[1]))
            if width <= 0:
                width = int(math.ceil(pil_font.getlength(render_text)))
                if stroke_px > 0:
                    width += stroke_px * 2
            height = max(height, default_line_height + (stroke_px * 2))
            ascent = pil_ascent + stroke_px
            return max(width, 1), height, ascent

        space_width = int(math.ceil(pil_font.getlength(" "))) or 6
    else:

        def measure_word(text: str) -> Tuple[int, int, int]:
            thickness_for_measure = max(effective_text_thickness, effective_outline_thickness)
            ((word_w, word_h), baseline) = cv2.getTextSize(
                text if text else " ",
                design.font,
                effective_text_scale,
                thickness_for_measure,
            )
            word_w = max(word_w, 1)
            word_h = max(word_h, 1)
            ascent = word_h - baseline
            if ascent <= 0:
                ascent = word_h
            return word_w, word_h, ascent

        thickness_for_measure = max(effective_text_thickness, effective_outline_thickness)
        space_width = cv2.getTextSize(
            " ", design.font, effective_text_scale, thickness_for_measure
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
    line_spacing = int(effective_line_spacing)
    if line_heights:
        text_block_height = (
            sum(line_heights) + max(0, len(line_heights) - 1) * line_spacing
        )
    else:
        text_block_height = 0

    padding_x = int(effective_margin_x)
    padding_y = int(effective_margin_y)
    box_width = int(text_block_width + 2 * padding_x)
    box_height = int(text_block_height + 2 * padding_y)
    # Center the box (same as original)
    box_left = int(max(0, (width - box_width) / 2))
    box_right = int(min(width, box_left + box_width))
    line_count = len(lines)
    bottom_margin_dynamic = int(effective_bottom_margin)
    if line_count == 1:
        bottom_margin_dynamic = max(0, int(effective_bottom_margin * 0.85))
    elif line_count >= 2:
        bottom_margin_dynamic = int(effective_bottom_margin) + _scale_int(8, min_value=0)
    box_bottom = height - max(0, bottom_margin_dynamic)
    box_top = box_bottom - box_height
    if box_top < 0:
        box_top = 0
        box_bottom = min(height, box_height)

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
            shadow_offset=effective_box_shadow_offset,
            shadow_blur=effective_box_shadow_blur,
            shadow_alpha=getattr(design, "box_shadow_alpha", 0.0),
            radius=effective_corner_radius,
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
                shadow_offset=effective_box_shadow_offset,
                shadow_blur=effective_box_shadow_blur,
                shadow_alpha=0.0,  # No shadow for 9:16
                radius=effective_corner_radius,
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
                padding_word_x, padding_word_y = effective_highlight_padding
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
                            stroke_width=effective_outline_thickness,
                            stroke_fill=outline_rgb,
                        )
                    except TypeError:
                        # Fallback for older PIL versions - draw outline in 8 directions
                        outline_thickness = effective_outline_thickness
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
                        effective_text_scale,
                        outline_color,
                        thickness=effective_outline_thickness,
                        lineType=cv2.LINE_AA,
                    )

                cv2.putText(
                    annotated,
                    word,
                    (x_cursor, baseline_y),
                    design.font,
                    effective_text_scale,
                    text_color,
                    thickness=effective_text_thickness,
                    lineType=cv2.LINE_AA,
                )
            x_cursor += word_width
        y_cursor = baseline_y + line_descent + line_spacing

    if pil_image is not None:
        annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return annotated


@dataclass(frozen=True)
class OverlayTiming:
    sched_start: float
    sched_end: float
    words_start: float
    words_end: float


def overlay_alpha(
    t: float,
    seg: OverlayTiming,
    fade_seconds: float,
    *,
    is_cluster_start: bool = False,
    is_cluster_end: bool = False,
) -> float:
    # Never draw outside the scheduled window.
    if t < seg.sched_start or t >= seg.sched_end:
        return 0.0

    lead = max(0.0, seg.words_start - seg.sched_start)
    tail = max(0.0, seg.sched_end - seg.words_end)

    fade_in = min(float(fade_seconds), lead)
    fade_out = min(float(fade_seconds), tail)

    a = 1.0

    # Fade-in ONLY when coming from MAIN (not when coming from another overlay)
    if (not is_cluster_start) and fade_in > 1e-6 and t < seg.words_start:
        # If fade is shorter than ~a frame, just snap to 1 to avoid flicker
        if fade_in < 0.04:
            a = 1.0
        else:
            a = (t - seg.sched_start) / fade_in

    # Fade-out ONLY when going back to MAIN (not when another overlay follows)
    if (not is_cluster_end) and fade_out > 1e-6 and t > seg.words_end:
        a = min(a, (seg.sched_end - t) / fade_out)

    # Clamp
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    return float(a)


def _process_video_with_overlays_legacy(
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

    def _resolve_overlay_clip_path(path: str) -> str:
        if not path:
            return path
        if os.path.exists(path):
            return path

        candidates: List[str] = []
        norm = path.replace("\\", "/")
        if norm.startswith("clips/"):
            candidates.append(os.path.join("data", norm))
            candidates.append(os.path.join("data", "clips", os.path.basename(norm)))
        elif not os.path.isabs(path) and norm.endswith(".mp4"):
            candidates.append(os.path.join("clips", norm))
            candidates.append(os.path.join("data", "clips", os.path.basename(norm)))

        for candidate in candidates:
            if os.path.exists(candidate):
                logger.info(
                    "  [CLIP] Resolved missing overlay clip %s -> %s", path, candidate
                )
                return candidate

        return path

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
        clip_path = _resolve_overlay_clip_path(str(clip_path))
        segment["clip_path"] = clip_path
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

        clip_path = _resolve_overlay_clip_path(str(clip_path))
        segment["clip_path"] = clip_path
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
                    # segment that uses it *including*:
                    #   - the early lead-in used for MAIN→OVERLAY transitions
                    #   - the small tail margin after the last word
                    #   - a bit of slack for float / frame rounding
                    #
                    # In practice, from the logs, we were ending up about
                    # 0.1–0.23s short, which is exactly when the overlay runs
                    # out of frames right before the segment finishes.
                    #
                    # So we are gonna add a slightly larger fixed cushion instead of just
                    # one frame.
                    extra_margin = 0.30  # seconds; tweak if ever needed
                    required_duration = target_duration + extra_margin

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
                        slow_down_video(clip_path, tmp_path, speed_factor, target_fps=fps)
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
        effective_fps = compute_effective_fps(
            overlay_capture, processed_clip_path, fallback_fps=float(fps)
        )
        clip_state[processed_clip_path] = {
            "capture": overlay_capture,
            "total_frames": clip_total_frames,
            "fps": overlay_capture.get(cv2.CAP_PROP_FPS) or fps,
            "effective_fps": effective_fps,
            "next_frame": 0,
            # Anchor time (in seconds on the main video timeline) used to map
            # `current_time` -> overlay frame index. This is set whenever a clip
            # starts a new (non-continuation) overlay segment.
            "time_anchor": None,
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
        overlay_to_overlay_lead_margin = 0.08  # smaller early shift between overlays

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
                # OVERLAY → OVERLAY transition.
                #
                # Even here, WhisperX can place the start slightly late. Apply a
                # smaller early shift; downstream overlap-resolution trims the
                # previous overlay if needed, which is preferable to a late start.
                start_time_early = max(0.0, start_time - overlay_to_overlay_lead_margin)
                logger.info(
                "[HFR_SETUP] seg=%d OVERLAY→OVERLAY gap=%.3fs "
                "start=%.3fs early=%.3fs (small early shift)",
                idx,
                gap,
                start_time,
                start_time_early,
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
            # Prefer trimming the previous overlay so we don't delay the current one.
            # Late overlay starts are more noticeable than slightly shorter tails.
            new_prev_end = curr_start - 1
            if new_prev_end < prev_start:
                # Degenerate overlap: keep prev as a single frame and shift current.
                new_prev_end = prev_start
                new_curr_start = new_prev_end + 1
                if new_curr_start > curr_end:
                    new_curr_start = curr_end
                curr[0] = new_curr_start
            prev[1] = new_prev_end
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

    # Identify overlay->overlay links (back-to-back highlights). This is used to
    # prevent a 1-frame flash of MAIN when the next overlay clip fails to decode
    # its first frame.
    link_eps_frames = 1  # <= 1 frame gap counts as overlay→overlay
    cluster_end_segments: set[int] = set()
    cluster_start_segments: set[int] = set()
    for i in range(len(highlight_frame_ranges) - 1):
        prev_start, prev_end, prev_idx = highlight_frame_ranges[i]
        next_start, next_end, next_idx = highlight_frame_ranges[i + 1]
        if (next_start - prev_end) <= link_eps_frames:
            cluster_end_segments.add(int(prev_idx))
            cluster_start_segments.add(int(next_idx))


    frame_index = 0
    highlight_ranges_for_words = [
        (seg["start_word"], seg["end_word"]) for seg in highlight_segments
    ]
    prev_active_overlay_index: Optional[int] = None
    logger.info("  [VIDEO PROCESSING] Starting frame processing loop...")

    last_good_overlay_canvas: Optional[np.ndarray] = None  # no-main-flash fallback

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
                    # Even if we previously hit the end of the clip, we can still
                    # keep showing the last successfully decoded frame for the rest
                    # of the segment (prevents flashing back to the main video).
                    if overlay_cap is not None and (
                        not clip_info.get("finished", False)
                        or clip_info.get("last_frame") is not None
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
                                clip_info["time_anchor"] = current_time
                            else:
                                target_next = max(
                                    int(clip_info.get("next_frame", 0)), 0
                                )
                                clip_info["seek_frame"] = target_next
                                clip_info["needs_seek"] = True
                                clip_info["continuation_pending"] = True
                                clip_info["frames_to_drop"] = 0
                                clip_info["hold_last_frame"] = False
                                if clip_info.get("time_anchor") is None:
                                    clip_info["time_anchor"] = current_time

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
                        overlay_fps = float(
                            clip_info.get("effective_fps")
                            or clip_info.get("fps")
                            or 0.0
                        ) or float(fps)
                        if overlay_fps <= 0:
                            overlay_fps = float(fps) if fps > 0 else 25.0

                        time_anchor = clip_info.get("time_anchor")
                        if time_anchor is None:
                            time_anchor = current_time
                            clip_info["time_anchor"] = time_anchor

                        # Map main-video time -> overlay frame index so slowed clips
                        # (via setpts) actually play at their effective FPS and don't
                        # run out of frames early.
                        elapsed = max(0.0, float(current_time) - float(time_anchor))
                        desired_index = int(elapsed * overlay_fps + 1e-6)
                        current_index = desired_index
                        frame_to_overlay: Optional[np.ndarray] = None

                        if overlay_total_frames <= 0:
                            frame_to_overlay = clip_info.get("last_frame")
                        else:
                            last_capture_index = clip_info.get(
                                "last_capture_index", -1
                            )
                            # Clamp to the last valid frame index so we never seek
                            # to frame_count (which fails) and we can hold the last
                            # frame for the rest of the overlay segment.
                            if current_index >= overlay_total_frames:
                                current_index = overlay_total_frames - 1
                                clip_info["hold_last_frame"] = True

                            # If we're asking for the same frame as last time, reuse it.
                            if current_index == last_capture_index and clip_info.get("last_frame") is not None:
                                frame_to_overlay = clip_info.get("last_frame")
                            else:
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

                                if not ret_o or overlay_frame is None:
                                    # Try one retry seek to be safe
                                    logger.warning(
                                        "  [OVERLAY_READ_FAIL] seg=%d clip=%s frame_index=%d overlay_total_frames=%d",
                                        active_overlay_index,
                                        segment_clip_paths[active_overlay_index],
                                        current_index,
                                        overlay_total_frames,
                                    )

                                    # Attempt a re-seek and re-read once
                                    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                                    ret_o2, overlay_frame2 = overlay_cap.read()

                                    if ret_o2 and overlay_frame2 is not None:
                                        overlay_frame = overlay_frame2
                                    else:
                                        # Fall back to last_frame if we have one; otherwise mark finished.
                                        overlay_frame = clip_info.get("last_frame")
                                        if overlay_frame is None:
                                            clip_info["finished"] = True

                                if overlay_frame is not None:
                                    clip_info["last_frame"] = overlay_frame
                                    clip_info["last_capture_index"] = current_index
                                    frame_to_overlay = overlay_frame
                                    clip_info["next_frame"] = current_index + 1
                                else:
                                    frame_to_overlay = clip_info.get("last_frame")

                        # If we're at an overlay→overlay boundary and decoding fails,
                        # keep the previous overlay for a frame instead of flashing MAIN.
                        if (
                            frame_to_overlay is None
                            and active_overlay_index in cluster_start_segments
                            and last_good_overlay_canvas is not None
                            and last_good_overlay_canvas.shape == frame.shape
                        ):
                            frame[:, :] = last_good_overlay_canvas
                            frame_to_overlay = None  # handled; skip normal overlay pipeline

                        if active_overlay_index is not None and frame_to_overlay is None:
                            logger.warning(
                                "  [OVERLAY_MISS] frame=%d time=%.3fs seg=%d clip=%s "
                                "(overlay_total_frames=%d, next_frame=%d, last_capture_index=%d, finished=%s)",
                                frame_index,
                                current_time,
                                active_overlay_index,
                                segment_clip_paths[active_overlay_index],
                                overlay_total_frames,
                                current_index,
                                clip_info.get("last_capture_index", -1),
                                clip_info.get("finished", False),
                            )
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

                                last_good_overlay_canvas = resized_overlay.copy()

                                # 3) Replace the entire frame with the overlay so
                                #    the original and B-roll align pixel-perfectly.
                                frame[:, :] = resized_overlay


        # Draw subtitles on this frame (disabled)
        # frame_with_subtitles = draw_subtitle_on_frame(
        #     frame,
        #     transcript,
        #     current_time,
        #     subtitle_design,
        #     highlight_ranges_for_words,
        #     subtitle_segments=subtitle_segments,
        #     custom_subtitles=custom_subtitles,
        # )
        writer.write(frame)
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

    def _resolve_overlay_clip_path(path: str) -> str:
        if not path:
            return path
        if os.path.exists(path):
            return path

        candidates: List[str] = []
        norm = path.replace("\\", "/")
        if norm.startswith("clips/"):
            candidates.append(os.path.join("data", norm))
            candidates.append(os.path.join("data", "clips", os.path.basename(norm)))
        elif not os.path.isabs(path) and norm.endswith(".mp4"):
            candidates.append(os.path.join("clips", norm))
            candidates.append(os.path.join("data", "clips", os.path.basename(norm)))

        for candidate in candidates:
            if os.path.exists(candidate):
                logger.info(
                    "  [CLIP] Resolved missing overlay clip %s -> %s", path, candidate
                )
                return candidate

        return path

    logger.info("  [VIDEO PROCESSING] Opening video file...")
    cap = cv2.VideoCapture(main_video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open main video: {main_video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Use ffprobe-backed fps if possible (prevents drift if CAP_PROP_FPS lies)
    fps = compute_effective_fps(cap, main_video_path, fallback_fps=float(raw_fps))
    video_duration = total_frames / fps if fps > 0 else 0.0

    target_aspect_ratio = parse_aspect_ratio(aspect_ratio)
    width, height = compute_cropped_dimensions(
        source_width, source_height, target_aspect_ratio
    )

    (
        overlay_schedule_times,
        _highlight_subtitle_indices,
        _highlight_subtitle_spans,
        segment_on_durations_sec,
    ) = build_overlay_schedule_times(
        transcript=transcript,
        highlight_segments=highlight_segments,
        subtitle_segments=subtitle_segments,
        cluster_gap_seconds=None,
        lead_in_seconds=0.15,
        tail_out_seconds=0.28,
        use_subtitle_bounds_for_overlay=False,
        logger=logger,
    )

    logger.info(
        f"  [VIDEO PROCESSING] Video info: {source_width}x{source_height} @ {fps:.2f}fps, {video_duration:.2f}s duration"
    )
    logger.info(
        f"  [VIDEO PROCESSING] Target size: {width}x{height} (aspect ratio: {aspect_ratio})"
    )
    logger.info(f"  [VIDEO PROCESSING] Total frames to process: {total_frames}")

    segment_clip_paths: List[Optional[str]] = []
    clip_state: Dict[str, Dict[str, object]] = {}

    processed_clips: Dict[str, str] = {}  # original clip path -> processed clip path
    temp_files: List[str] = []  # Track temp files for cleanup

    # Compute, per clip, the longest ON-SCREEN duration it must cover (based on final schedule)
    clip_target_durations: Dict[str, float] = {}
    for seg_idx, seg in enumerate(highlight_segments):
        clip_path = seg.get("clip_path")
        if not clip_path:
            continue

        clip_path = _resolve_overlay_clip_path(str(clip_path))
        seg["clip_path"] = clip_path

        dur = float(segment_on_durations_sec[seg_idx] or 0.0)
        prev = clip_target_durations.get(clip_path)
        if prev is None or dur > prev:
            clip_target_durations[clip_path] = dur

    # Associate each highlight with a clip path and prepare capture state.
    # Slow down any clip that is shorter than the longest segment that uses it.
    for idx, segment in enumerate(highlight_segments):
        clip_path = segment.get("clip_path")
        if not clip_path:
            segment_clip_paths.append(None)
            continue

        clip_path = _resolve_overlay_clip_path(str(clip_path))
        segment["clip_path"] = clip_path

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
                clip_fps = float(tmp_cap.get(cv2.CAP_PROP_FPS) or fps)
                clip_frame_count = float(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                tmp_cap.release()

                if clip_fps <= 1e-6 or clip_frame_count <= 0:
                    logger.info(
                        f"  [CLIP] {clip_path}: unable to read duration via OpenCV "
                        f"(fps={clip_fps}, frames={clip_frame_count}) -> using clip as-is."
                    )
                    processed_clips[clip_path] = clip_path
                else:
                    clip_duration = ffprobe_duration_seconds(clip_path)
                    if clip_duration is None:
                        clip_duration = float(clip_frame_count) / float(clip_fps)

                    extra_margin = 0.30  # seconds; tweak if ever needed
                    required_duration = float(target_duration) + extra_margin

                    if clip_duration >= required_duration:
                        logger.info(
                            f"  [CLIP] {clip_path}: orig_dur={clip_duration:.3f}s, "
                            f"target_dur={target_duration:.3f}s, "
                            f"required={required_duration:.3f}s -> no slowdown."
                        )
                        processed_clips[clip_path] = clip_path
                    else:
                        speed_factor = clip_duration / required_duration
                        logger.info(
                            f"  [CLIP] {clip_path}: orig_dur={clip_duration:.3f}s, "
                            f"target_dur={target_duration:.3f}s, "
                            f"required={required_duration:.3f}s -> "
                            f"slowing down (speed_factor={speed_factor:.4f})."
                        )

                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tmp_path = tmp_file.name
                        tmp_file.close()

                        slow_down_video(clip_path, tmp_path, speed_factor, target_fps=fps)
                        processed_clips[clip_path] = tmp_path
                        temp_files.append(tmp_path)

        processed_clip_path = processed_clips[clip_path]
        segment_clip_paths.append(processed_clip_path)

        # Create and cache VideoCapture + metadata once per unique processed clip path.
        if processed_clip_path in clip_state:
            continue

        overlay_capture = cv2.VideoCapture(processed_clip_path)
        if not overlay_capture.isOpened():
            raise IOError(f"Cannot open overlay clip: {processed_clip_path}")

        clip_total_frames = int(overlay_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        eff_fps = compute_effective_fps(
            overlay_capture, processed_clip_path, fallback_fps=float(fps)
        )
        clip_state[processed_clip_path] = {
            "capture": overlay_capture,
            "total_frames": clip_total_frames,
            "fps": overlay_capture.get(cv2.CAP_PROP_FPS) or fps,
            "segment_start_time": 0.0,
            "effective_fps": eff_fps,
            "current_segment_index": None,
            "needs_seek": True,
            "last_frame": None,
            "last_capture_index": -1,
            "finished": False,
        }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot create output file: {output_path}")

    # Controls visual smoothness. Set to 0.0 for hard cuts.
    OVERLAY_FADE_SECONDS = 0.07

    logger.info("  [SCHED] Overlay schedule (time-based, run-aware):")

    # Build per-segment timing so we can fade safely without “bleeding”.
    seg_timing: Dict[int, OverlayTiming] = {}

    for st_sched, et_sched, seg_idx in overlay_schedule_times:
        sw = int(highlight_segments[seg_idx]["start_word"])
        ew = int(highlight_segments[seg_idx]["end_word"])

        last_word = len(transcript) - 1
        sw = max(0, min(sw, last_word))
        ew = max(0, min(ew, last_word))
        if ew < sw:
            sw, ew = ew, sw

        words_start = float(transcript[sw]["start_time"])
        words_end = float(transcript[ew]["end_time"])

        seg_timing[seg_idx] = OverlayTiming(
            sched_start=float(st_sched),
            sched_end=float(et_sched),
            words_start=words_start,
            words_end=words_end,
        )

        words = " ".join(entry["word"] for entry in transcript[sw : ew + 1])
        clip = highlight_segments[seg_idx].get("clip_path")
        logger.info(
            "  [SCHED] seg=%d clip=%s sched=%.3f-%.3fs words_time=%.3f-%.3fs words=%s",
            seg_idx,
            clip,
            st_sched,
            et_sched,
            words_start,
            words_end,
            words,
        )

    # Identify overlay->overlay links (cluster transitions)
    link_eps = 0.75 / float(fps)  # < 1 frame tolerance
    cluster_end_segments: set[int] = set()
    cluster_start_segments: set[int] = set()

    for i in range(len(overlay_schedule_times) - 1):
        st_i, et_i, idx_i = overlay_schedule_times[i]
        st_n, et_n, idx_n = overlay_schedule_times[i + 1]
        if (st_n - et_i) <= link_eps:
            cluster_end_segments.add(idx_i)  # don't fade OUT idx_i
            cluster_start_segments.add(idx_n)  # don't fade IN idx_n

    frame_index = 0
    highlight_ranges_for_words = [
        (int(seg["start_word"]), int(seg["end_word"])) for seg in highlight_segments
    ]
    logger.info("  [VIDEO PROCESSING] Starting frame processing loop...")

    last_good_overlay_canvas: Optional[np.ndarray] = None  # no-main-flash fallback

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect_ratio(frame, target_aspect_ratio)

        # Use actual video position for timing when available
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec >= 0:
            current_time = pos_msec / 1000.0
        else:
            current_time = frame_index / fps

        # IMPORTANT: use ONE time everywhere (schedule pick + elapsed + alpha)
        t_eval = float(current_time)

        # Pick active overlay segment
        active_overlay_index = None
        segment_start_time = None

        for st, et, seg_idx in overlay_schedule_times:
            if st <= t_eval < et:
                # If overlaps ever happen, later-starting wins (by not breaking)
                active_overlay_index = seg_idx
                segment_start_time = st
            elif st > t_eval:
                break

        if active_overlay_index is not None:
            clip_path = segment_clip_paths[active_overlay_index]
            if clip_path:
                clip_info = clip_state.get(clip_path)
                if clip_info:
                    overlay_cap = clip_info["capture"]

                    if clip_info.get("current_segment_index") != active_overlay_index:
                        clip_info["current_segment_index"] = active_overlay_index
                        clip_info["segment_start_time"] = float(
                            segment_start_time
                            if segment_start_time is not None
                            else current_time
                        )
                        clip_info["needs_seek"] = True
                        clip_info["last_capture_index"] = -1
                        clip_info["last_frame"] = None  # IMPORTANT: don't reuse stale frames
                        clip_info["finished"] = False  # IMPORTANT: always try to read

                    eff_fps = float(clip_info.get("effective_fps") or fps)
                    if eff_fps <= 1e-3:
                        eff_fps = float(fps)

                    seg_start_t = float(clip_info.get("segment_start_time", current_time))
                    elapsed = max(0.0, t_eval - seg_start_t)
                    desired_index = int(elapsed * eff_fps + 1e-6)

                    frame_to_overlay: Optional[np.ndarray] = None
                    last_idx = int(clip_info.get("last_capture_index", -1))
                    last_frame = clip_info.get("last_frame")

                    if clip_info.get("finished", False):
                        frame_to_overlay = last_frame
                    elif desired_index == last_idx and last_frame is not None:
                        frame_to_overlay = last_frame
                    else:
                        if clip_info.get("needs_seek", False) or desired_index != last_idx + 1:
                            ok = overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, desired_index)
                            if not ok:
                                overlay_cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000.0)
                            clip_info["needs_seek"] = False

                        ret_o, overlay_frame = overlay_cap.read()

                        # one more attempt if read failed
                        if not ret_o or overlay_frame is None:
                            overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, desired_index)
                            ret2, overlay_frame2 = overlay_cap.read()
                            overlay_frame = overlay_frame2 if ret2 else None

                        if overlay_frame is not None:
                            clip_info["last_frame"] = overlay_frame
                            clip_info["last_capture_index"] = desired_index
                            frame_to_overlay = overlay_frame
                        else:
                            clip_info["finished"] = True
                            frame_to_overlay = clip_info.get("last_frame")

                    # if we're at an overlay->overlay boundary and decoding fails,
                    # keep the previous overlay for a frame instead of flashing MAIN.
                    if (
                        frame_to_overlay is None
                        and (active_overlay_index in cluster_start_segments)
                        and last_good_overlay_canvas is not None
                    ):
                        frame[:, :] = last_good_overlay_canvas
                        frame_to_overlay = None  # handled; skip normal overlay pipeline

                    if frame_to_overlay is not None:
                        overlay_cropped = crop_to_aspect_ratio(
                            frame_to_overlay, target_aspect_ratio
                        )
                        oh, ow = overlay_cropped.shape[:2]
                        if oh > 0 and ow > 0:
                            h, w = frame.shape[:2]
                            if (ow != w) or (oh != h):
                                overlay_cropped = cv2.resize(overlay_cropped, (w, h))

                            last_good_overlay_canvas = overlay_cropped
                            timing = seg_timing.get(active_overlay_index)

                            if timing is None or OVERLAY_FADE_SECONDS <= 1e-6:
                                frame[:, :] = overlay_cropped
                                if active_overlay_index in cluster_end_segments:
                                    last_good_overlay_canvas = overlay_cropped.copy()
                            else:
                                is_cluster_start = active_overlay_index in cluster_start_segments
                                is_cluster_end = active_overlay_index in cluster_end_segments

                                a = overlay_alpha(
                                    t_eval,
                                    timing,
                                    OVERLAY_FADE_SECONDS,
                                    is_cluster_start=is_cluster_start,
                                    is_cluster_end=is_cluster_end,
                                )

                                if a >= 0.999:
                                    frame[:, :] = overlay_cropped
                                    if is_cluster_end:
                                        last_good_overlay_canvas = overlay_cropped.copy()
                                elif a <= 0.001:
                                    pass
                                else:
                                    frame[:, :] = cv2.addWeighted(
                                        overlay_cropped,
                                        float(a),
                                        frame,
                                        float(1.0 - a),
                                        0.0,
                                    )
                                    if is_cluster_end:
                                        last_good_overlay_canvas = overlay_cropped.copy()

                        if active_overlay_index in cluster_end_segments:
                            last_good_overlay_canvas = overlay_cropped.copy()
                        else:
                            last_good_overlay_canvas = overlay_cropped

        # Draw subtitles on this frame (disabled)
        # frame_with_subtitles = draw_subtitle_on_frame(
        #     frame,
        #     transcript,
        #     t_eval,
        #     subtitle_design,
        #     highlight_ranges_for_words,
        #     subtitle_segments=subtitle_segments,
        #     custom_subtitles=custom_subtitles,
        # )
        writer.write(frame)
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
            logger.warning(f"  [CLEANUP] Could not remove temporary file {temp_file}: {e}")

    step_duration = time.time() - step_start
    logger.info(
        f"  [VIDEO PROCESSING] ✓ Completed processing {frame_index} frames in {step_duration:.2f}s ({step_duration/60:.2f} min)"
    )
    if step_duration > 0:
        logger.info(
            f"  [VIDEO PROCESSING] Average processing speed: {frame_index/step_duration:.2f} fps"
        )


def _merge_audio_tracks_legacy(
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

    # --- Determine video duration (seconds) ---
    video_duration = None
    try:
        cap = cv2.VideoCapture(silent_video_path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap.release()
            if fps > 1e-6 and frame_count > 0:
                video_duration = frame_count / fps
    except Exception:
        video_duration = None

    if not video_duration or video_duration <= 0:
        video_duration = ffprobe_duration_seconds(silent_video_path)

    if not video_duration or video_duration <= 0:
        raise ValueError(f"Could not determine video duration for {silent_video_path}")

    logger.info(f"  [AUDIO MERGE] Video duration: {video_duration:.3f}s")

    # --- Build ffmpeg inputs (with looping for music) ---
    cmd: List[str] = ["ffmpeg", "-y", "-i", silent_video_path]
    filter_parts: List[str] = []
    mix_labels: List[str] = []

    input_idx = 1  # ffmpeg input index (0 is silent_video_path)

    # Helper: check main video has audio
    def _video_has_audio(path: str) -> bool:
        try:
            r = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return (r.returncode == 0) and (r.stdout.strip() == "audio")
        except Exception:
            return False

    # 1) Main audio
    if preserve_main_audio and _video_has_audio(main_video_path):
        cmd += ["-i", main_video_path]
        main_idx = input_idx
        input_idx += 1

        filter_parts.append(
            f"[{main_idx}:a]atrim=0:{video_duration:.6f},asetpts=PTS-STARTPTS,volume=1.0[main_audio]"
        )
        mix_labels.append("main_audio")
        logger.info("  [AUDIO MERGE] - Added main audio")

    # 2) Global music (loop input infinitely, then trim)
    if global_music_path:
        if not os.path.exists(global_music_path):
            raise FileNotFoundError(f"Global music file not found: {global_music_path}")

        cmd += ["-stream_loop", "-1", "-i", global_music_path]
        g_idx = input_idx
        input_idx += 1

        vol = float(global_music_volume)
        filter_parts.append(
            f"[{g_idx}:a]atrim=0:{video_duration:.6f},asetpts=PTS-STARTPTS,volume={vol}[global_music]"
        )
        mix_labels.append("global_music")
        logger.info(f"  [AUDIO MERGE] - Added global music (looped): {global_music_path}")

    # 3) Per-segment music (loop each track infinitely, then trim + delay)
    for seg_i, segment in enumerate(highlight_segments):
        music_path = segment.get("music_path")
        if not music_path:
            continue
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Music file not found: {music_path}")

        start_word = int(segment["start_word"])
        end_word = int(segment["end_word"])
        start_time = float(transcript[start_word]["start_time"])
        end_time = float(transcript[end_word]["end_time"])
        duration = max(0.0, end_time - start_time)
        if duration <= 1e-6:
            continue

        cmd += ["-stream_loop", "-1", "-i", music_path]
        s_idx = input_idx
        input_idx += 1

        label = f"segment_{seg_i}"
        vol = float(segment.get("music_volume", 1.0))
        delay_ms = int(start_time * 1000)

        # NOTE: adelay=ms|ms assumes stereo; if you ever use 5.1, swap to an all-channels-safe version.
        filter_parts.append(
            f"[{s_idx}:a]atrim=0:{duration:.6f},asetpts=PTS-STARTPTS,"
            f"volume={vol},adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_labels.append(label)
        logger.info(
            f"  [AUDIO MERGE] - Added segment music (looped): seg={seg_i} file={music_path}"
        )

    # If nothing to mix, copy video without audio
    if not mix_labels:
        logger.info(
            "  [AUDIO MERGE] No audio tracks requested; writing video without audio."
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                silent_video_path,
                "-c:v",
                "copy",
                "-an",
                final_output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return

    # 4) Mix tracks
    if len(mix_labels) == 1:
        mix_out = mix_labels[0]
    else:
        mix_inputs = "".join(f"[{lab}]" for lab in mix_labels)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0[mixed_audio]"
        )
        mix_out = "mixed_audio"

    filter_complex = ";".join(filter_parts)

    # 5) Output
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        f"[{mix_out}]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        final_output_path,
    ]

    logger.info("  [AUDIO MERGE] Running ffmpeg...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(
            f"  [AUDIO MERGE] ✓ Audio merge completed in {time.time() - step_start:.2f}s"
        )
    except subprocess.CalledProcessError as e:
        logger.error("  [AUDIO MERGE] ✗ ffmpeg failed.")
        logger.error(e.stderr)
        raise


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

    # Defensive: ensure subtitle segments are monotonic and non-overlapping.
    #
    # The renderer keeps the "previous" subtitle visible until the next segment
    # starts. If segments overlap (e.g., due to ASR drift/backtracking or repaired
    # indices), the later segment can effectively show words/text that belong to
    # an earlier segment, which users perceive as "double/stacked" subtitles.
    if subtitle_segments:
        needs_normalization = False
        prev_end_idx = -1
        for seg_start, seg_end in subtitle_segments:
            try:
                seg_start_i = int(seg_start)
                seg_end_i = int(seg_end)
            except (TypeError, ValueError):
                continue
            if seg_end_i < seg_start_i:
                seg_start_i, seg_end_i = seg_end_i, seg_start_i
            if seg_start_i <= prev_end_idx:
                needs_normalization = True
                break
            prev_end_idx = seg_end_i

        if needs_normalization:
            logger.warning(
                "[SUBTITLE] Overlapping subtitle segments detected; normalizing to prevent stacked subtitles."
            )
            last_word_idx = len(transcript) - 1
            normalized_segments: List[Tuple[int, int]] = []
            normalized_texts: Optional[List[str]] = (
                [] if custom_subtitle_texts is not None else None
            )
            last_end = -1
            for idx, (seg_start, seg_end) in enumerate(subtitle_segments):
                seg_start_i = int(seg_start)
                seg_end_i = int(seg_end)
                if seg_end_i < seg_start_i:
                    seg_start_i, seg_end_i = seg_end_i, seg_start_i
                seg_start_i = max(0, min(seg_start_i, last_word_idx))
                seg_end_i = max(0, min(seg_end_i, last_word_idx))
                if seg_end_i < seg_start_i:
                    seg_end_i = seg_start_i

                if seg_start_i <= last_end:
                    seg_start_i = min(last_word_idx, last_end + 1)
                if seg_end_i < seg_start_i:
                    seg_end_i = seg_start_i

                normalized_segments.append((seg_start_i, seg_end_i))
                if normalized_texts is not None and custom_subtitle_texts is not None:
                    if idx < len(custom_subtitle_texts):
                        normalized_texts.append(custom_subtitle_texts[idx])
                last_end = max(last_end, seg_end_i)

            subtitle_segments = normalized_segments
            if normalized_texts is not None:
                custom_subtitle_texts = normalized_texts

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
