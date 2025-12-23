"""
Flask web application for video editing frontend.
Allows users to upload videos, select transcript highlights, and process videos.
"""

import os
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from flask import request

def log_request_context(tag: str):
    logger.info(
        "[%s] %s %s content_length=%s, remote_addr=%s, files=%s",
        tag,
        request.method,
        request.path,
        request.content_length,
        request.remote_addr,
        list(request.files.keys()),
    )

from video_overlay_script import (
    ProjectConfig,
    HighlightAssignment,
    build_transcript,
    render_project,
    SubtitleSentence,
    get_subtitle_design_for_aspect_ratio
)
from typing import List
# Check if React build exists
USE_REACT_BUILD = os.path.exists('frontend/dist/index.html')

if USE_REACT_BUILD:
    # Serve React build
    app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
else:
    # Serve traditional templates
    app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# S3 Configuration
S3_BUCKET_NAME = 's3videocrafter'
S3_REGION = 'us-east-1'
AWS_ACCESS_KEY_ID = 'AKIA3ETPXFJGJVSFMZPX'
AWS_SECRET_ACCESS_KEY = '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'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('clips', exist_ok=True)
os.makedirs('audio_files', exist_ok=True)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)


# S3 Upload Functions
def upload_file_to_s3(file_path, s3_key):
    """Upload a file to S3 bucket."""
    try:
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': get_content_type(file_path)}
        )
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        return s3_url
    except ClientError as e:
        print(f"Error uploading {file_path} to S3: {e}")
        raise


def get_content_type(file_path):
    """Get content type based on file extension."""
    ext = Path(file_path).suffix.lower()
    content_types = {
        '.mp4': 'video/mp4',
        '.json': 'application/json',
        '.txt': 'text/plain',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
    }
    return content_types.get(ext, 'application/octet-stream')


def create_project_json(video_path, highlights, transcript, subtitle_sentences, 
                        aspect_ratio, output_filename=None, output_path=None):
    """Create a project JSON file with all project data."""
    # Convert subtitle sentences to serializable format
    serializable_subtitles = []
    for s in subtitle_sentences:
        if isinstance(s, SubtitleSentence):
            serializable_subtitles.append({
                'text': s.text,
                'phrase': s.phrase if s.phrase else s.text,
                'occurrence': s.occurrence,
                'start_word': s.start_word if hasattr(s, 'start_word') else None,
                'end_word': s.end_word if hasattr(s, 'end_word') else None,
            })
        elif isinstance(s, dict):
            serializable_subtitles.append(s)
        else:
            # String or other type
            serializable_subtitles.append({
                'text': str(s),
                'phrase': str(s),
                'occurrence': 1
            })
    
    project_data = {
        'project_info': {
            'created_at': datetime.now().isoformat(),
            'video_path': video_path,
            'output_filename': output_filename,
            'output_path': output_path,
            'aspect_ratio': aspect_ratio,
        },
        'highlights': highlights,
        'transcript': transcript,
        'subtitle_sentences': serializable_subtitles,
        'statistics': {
            'total_highlights': len(highlights),
            'total_transcript_words': len(transcript),
            'total_subtitle_sentences': len(subtitle_sentences),
        }
    }
    return project_data


# def render_project_with_transcript(config: ProjectConfig, transcript: list):
#     """
#     Render project using an existing transcript instead of regenerating it.
#     This avoids calling Whisper again which is slow and unnecessary.
#     """
#     from video_overlay_script import (
#         map_assignments_to_segments,
#         process_video_with_overlays,
#         merge_audio_tracks,
#         generate_default_subtitle_segments,
#         HAVE_MOVIEPY
#     )

#     # Use the provided transcript instead of calling build_transcript
#     highlight_segments = map_assignments_to_segments(
#         transcript, config.highlight_assignments
#     )

#     any_segment_music = any(
#         assignment.music_path for assignment in config.highlight_assignments
#     )
#     needs_audio_merge = HAVE_MOVIEPY and (
#         config.preserve_audio or bool(config.global_music_path) or any_segment_music
#     )
#     final_output_path = config.output_path
#     silent_output_path = final_output_path

#     # Generate subtitle segments
#     subtitle_segments = config.subtitle_segments
#     if subtitle_segments is None:
#         subtitle_segments = generate_default_subtitle_segments(
#             transcript, highlight_segments
#         )

#     if needs_audio_merge:
#         root, ext = os.path.splitext(final_output_path)
#         ext = ext or ".mp4"
#         silent_output_path = f"{root}.silent{ext}"

#     # Render video with overlays
#     process_video_with_overlays(
#         config.main_video_path,
#         transcript,
#         highlight_segments,
#         config.subtitle_design,
#         silent_output_path,
#         subtitle_segments=subtitle_segments,
#         custom_subtitles=None,
#     )

#     # Merge audio if needed
#     if needs_audio_merge:
#         merge_audio_tracks(
#             silent_output_path,
#             config.main_video_path,
#             transcript,
#             highlight_segments,
#             final_output_path,
#             preserve_main_audio=config.preserve_audio,
#             global_music_path=config.global_music_path,
#             global_music_volume=config.global_music_volume,
#         )

#         if os.path.exists(silent_output_path) and silent_output_path != final_output_path:
#             os.remove(silent_output_path)

#     return {
#         "output_path": final_output_path,
#         "transcript": transcript,
#         "highlight_segments": highlight_segments,
#     }

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'm4a'}


def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    """Render the main page."""
    if USE_REACT_BUILD:
        return send_from_directory(app.static_folder, 'index.html')
    return render_template('index.html')


@app.route('/test-route')
def test_route():
    """Test route to verify server is responding."""
    return jsonify({'message': 'Server is working!', 'routes': ['upload-video', 'upload-video-with-txt']})


@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Handle main video upload and generate transcript."""
    log_request_context("UPLOAD_VIDEO")

    if 'video' not in request.files:
        logger.warning("[UPLOAD_VIDEO] No 'video' file in request")
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400

    try:
        # Save the uploaded video
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info("[UPLOAD_VIDEO] Saving video as %s", video_path)
        file.save(video_path)
        logger.info(
            "[UPLOAD_VIDEO] Saved video (%s), size ~%.2f MB",
            filename,
            os.path.getsize(video_path) / (1024 * 1024),
        )

        # Generate transcript using Whisper
        whisper_model = request.form.get('whisper_model', 'base')
        transcript = build_transcript(video_path, None, whisper_model)

        # Extract just the words for display
        words = [entry['word'] for entry in transcript]
        full_text = ' '.join(words)

        logger.info(
            "[UPLOAD_VIDEO] Transcript generated, words=%d", len(words)
        )

        return jsonify({
            'success': True,
            'video_path': video_path,
            'transcript': transcript,
            'full_text': full_text,
            'word_count': len(words)
        })

    except Exception as e:
        logger.exception("[UPLOAD_VIDEO] Error processing video")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500


@app.route('/upload-video-with-txt', methods=['POST'])
def upload_video_with_txt():
    """Handle video upload with TXT transcript file."""
    log_request_context("UPLOAD_VIDEO_WITH_TXT")

    if 'video' not in request.files:
        logger.warning("[UPLOAD_VIDEO_WITH_TXT] No 'video' file in request")
        return jsonify({'error': 'No video file provided'}), 400

    if 'transcript_file' not in request.files:
        return jsonify({'error': 'No transcript file provided'}), 400

    video_file = request.files['video']
    txt_file = request.files['transcript_file']

    if video_file.filename == '' or txt_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid video file type'}), 400

    if not txt_file.filename.endswith('.txt'):
        return jsonify({'error': 'Transcript must be a .txt file'}), 400

    try:
        # Save the uploaded video
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Video saved to %s (%.2f MB)",
            video_path,
            os.path.getsize(video_path) / (1024 * 1024),
        )

        # Read the transcript text and split by lines
        transcript_text = txt_file.read().decode("utf-8", errors="replace")
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Transcript text length=%d",
            len(transcript_text),
        )
        lines = [line.strip() for line in transcript_text.split('\n') if line.strip()]
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Parsed %d non-empty lines from transcript",
            len(lines),
        )

        # Fast path: build a transcript from the provided TXT only.
        #
        # We intentionally DO NOT run Whisper/WhisperX here so the UI stays responsive.
        # Transcription + waveform alignment happens when the user clicks "Process video".
        transcript = []
        subtitles = []
        word_cursor = 0
        default_word_duration = 0.5

        for line in lines:
            tokens = [tok for tok in line.split() if tok]
            if not tokens:
                continue

            start_word = word_cursor
            for idx, tok in enumerate(tokens):
                t0 = (word_cursor + idx) * default_word_duration
                t1 = t0 + default_word_duration
                transcript.append({"word": tok, "start_time": t0, "end_time": t1})
            word_cursor += len(tokens)
            end_word = word_cursor - 1

            subtitles.append(
                {
                    "text": line,
                    "start_word": start_word,
                    "end_word": end_word,
                    "word_count": len(tokens),
                }
            )

        if not transcript:
            return jsonify({'error': 'Transcript file produced no words'}), 400

        words = [entry['word'] for entry in transcript]
        full_text = ' '.join(words)

        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Prepared draft transcript (words=%d, subtitles=%d). "
            "Transcription is deferred to processing.",
            len(words),
            len(subtitles),
        )

        response_data = {
            'success': True,
            'video_path': video_path,
            'transcript': transcript,
            'full_text': full_text,
            'word_count': len(words),
            'subtitles': subtitles
        }
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Returning draft transcript (words=%d, subtitles=%d)",
            len(words),
            len(subtitles),
        )
        return jsonify(response_data)

    except Exception as e:
        logger.exception("[UPLOAD_VIDEO_WITH_TXT] Error processing files")
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500


@app.route('/upload-clip', methods=['POST'])
def upload_clip():
    """Handle clip/audio file upload for highlights."""
    log_request_context("UPLOAD_CLIP")

    if 'file' not in request.files:
        logger.warning("[UPLOAD_CLIP] No 'file' in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if it's video or audio
    is_video = allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    is_audio = allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS)

    if not (is_video or is_audio):
        return jsonify({'error': 'Invalid file type. Please upload a video or audio file.'}), 400

    try:
        filename = secure_filename(file.filename)

        # Save to appropriate folder
        if is_video:
            save_path = os.path.join('clips', filename)
        else:
            save_path = os.path.join('audio_files', filename)

        file.save(save_path)

        logger.info(
            "[UPLOAD_CLIP] Saved %s file to %s (%.2f MB)",
            "video" if is_video else "audio",
            save_path,
            os.path.getsize(save_path) / (1024 * 1024),
        )

        return jsonify({
            'success': True,
            'file_path': save_path,
            'file_type': 'video' if is_video else 'audio'
        })

    except Exception as e:
        logger.exception("[UPLOAD_CLIP] Error uploading file")
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500


@app.route('/process-video', methods=['POST'])
def process_video():
    """Process the video with highlights and generate output."""
    request_start_time = time.time()
    logger.info("=" * 80)
    logger.info("RECEIVED VIDEO PROCESSING REQUEST")
    logger.info("=" * 80)
    
    try:
        data = request.json

        video_path = data.get('video_path')
        highlights = data.get('highlights', [])
        transcript = data.get('transcript', [])
        subtitle_sentences = data.get('subtitle_sentences', [])
        aspect_ratio = data.get('aspect_ratio', '4:5')  # Default to 4:5

        def _safe_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # If the user creates manual highlights out of order in the editor, the
        # backend mapping can clamp/skip earlier selections due to its monotonic
        # cursor. Sort by word indices (when provided) so the mapping always
        # runs chronologically.
        if isinstance(highlights, list) and highlights:
            indexed_highlights = list(enumerate(highlights))

            def _highlight_sort_key(item):
                original_index, highlight = item
                if not isinstance(highlight, dict):
                    return (1, original_index)
                start_idx = _safe_int(highlight.get("start_word"))
                end_idx = _safe_int(highlight.get("end_word"))
                if start_idx is None and end_idx is None:
                    return (1, original_index)
                if start_idx is None:
                    start_idx = end_idx
                if end_idx is None:
                    end_idx = start_idx
                low = min(start_idx, end_idx)
                high = max(start_idx, end_idx)
                return (0, low, high, original_index)

            highlights = [
                highlight
                for _, highlight in sorted(indexed_highlights, key=_highlight_sort_key)
            ]

        logger.info(f"[REQUEST] Video path: {video_path}")
        logger.info(f"[REQUEST] Highlights: {len(highlights)}")
        logger.info(f"[REQUEST] Transcript words: {len(transcript)}")
        logger.info(f"[REQUEST] Subtitle sentences: {len(subtitle_sentences)}")
        logger.info(f"[REQUEST] Aspect ratio: {aspect_ratio}")

        # Validate aspect ratio
        if aspect_ratio not in ['4:5', '9:16']:
            aspect_ratio = '4:5'  # Default to 4:5 if invalid

        if not video_path or not os.path.exists(video_path):
            logger.error(f"[REQUEST] Video file not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 400

        # Build highlight assignments
                # Build highlight assignments (respect manual word indices + occurrence)
        assignments = []
        for highlight in highlights:
            assignment = HighlightAssignment(
                phrase=highlight.get("phrase"),
                clip_path=highlight.get("clip_path"),
                music_path=highlight.get("music_path"),
                music_volume=float(highlight.get("music_volume", 1.0)),
                occurrence=int(highlight.get("occurrence", 1) or 1),
                start_word=highlight.get("start_word"),
                end_word=highlight.get("end_word"),
            )
            assignments.append(assignment)


        # Generate output filename
        output_filename = f"output_{Path(video_path).stem}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Convert subtitles to subtitle_segments format (list of tuples)
        sentences: List[SubtitleSentence] = []
        if subtitle_sentences:
            if isinstance(subtitle_sentences, list):
                for item in subtitle_sentences:
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
                        phrase_value = item.get("phrase", text_value) or text_value
                        occurrence_value = max(1, int(item.get("occurrence", 1) or 1))
                        # Do not trust UI word indices here.
                        # The render pipeline maps subtitle sentences onto the audio-aligned transcript.
                        sentences.append(
                            SubtitleSentence(
                                text=text_value,
                                phrase=phrase_value,
                                occurrence=occurrence_value,
                            )
                        )

        t_text = " ".join(i['word'] for i in transcript).strip()
        
        # Get subtitle design for the selected aspect ratio
        subtitle_design = get_subtitle_design_for_aspect_ratio(aspect_ratio)
        
        # Create project config
        config = ProjectConfig(
            main_video_path=video_path,
            output_path=output_path,
            transcript_text=t_text,
            highlight_assignments=assignments,
            preserve_audio=data.get('preserve_audio', True),
            subtitle_sentences=sentences,
            aspect_ratio=aspect_ratio,
            subtitle_design=subtitle_design
        )

        # Render the project with the existing transcript
        logger.info("[REQUEST] Starting render_project...")
        render_start = time.time()
        render_project(config)
        render_duration = time.time() - render_start
        logger.info(f"[REQUEST] ✓ render_project completed in {render_duration:.2f}s")

        # Create project JSON file
        logger.info("[REQUEST] Creating project JSON file...")
        project_data = create_project_json(
            video_path=video_path,
            highlights=highlights,
            transcript=transcript,
            subtitle_sentences=subtitle_sentences,
            aspect_ratio=aspect_ratio,
            output_filename=output_filename,
            output_path=output_path
        )
        
        # Save project JSON locally
        project_filename = f"project_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        project_path = os.path.join(app.config['OUTPUT_FOLDER'], project_filename)
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        # Upload to S3
        s3_video_url = None
        s3_project_url = None
        try:
            # Create S3 keys with timestamp for organization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_s3_key = f"videos/{timestamp}_{output_filename}"
            project_s3_key = f"projects/{timestamp}_{project_filename}"
            
            # Upload video to S3
            logger.info(f"[REQUEST] Uploading video to S3: {video_s3_key}")
            s3_start = time.time()
            s3_video_url = upload_file_to_s3(output_path, video_s3_key)
            s3_duration = time.time() - s3_start
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"[REQUEST] ✓ Video uploaded to S3 in {s3_duration:.2f}s ({file_size_mb:.2f} MB) - {s3_video_url}")
            
            # Upload project JSON to S3
            logger.info(f"[REQUEST] Uploading project file to S3: {project_s3_key}")
            s3_start = time.time()
            s3_project_url = upload_file_to_s3(project_path, project_s3_key)
            s3_duration = time.time() - s3_start
            logger.info(f"[REQUEST] ✓ Project file uploaded to S3 in {s3_duration:.2f}s - {s3_project_url}")
            
        except Exception as s3_error:
            logger.warning(f"[REQUEST] S3 upload failed: {s3_error}")
            # Continue even if S3 upload fails - local files are still available

        total_request_time = time.time() - request_start_time
        logger.info("=" * 80)
        logger.info(f"[REQUEST] ✓ REQUEST COMPLETED in {total_request_time:.2f}s ({total_request_time/60:.2f} minutes)")
        logger.info("=" * 80)

        return jsonify({
            'success': True,
            'output_path': output_path,
            'output_filename': output_filename,
            'project_filename': project_filename,
            'project_path': project_path,
            's3_video_url': s3_video_url,
            's3_project_url': s3_project_url,
            'message': 'Video processed successfully!' + (' (Uploaded to S3)' if s3_video_url else ' (S3 upload failed)')
        })

    except Exception:
        logger.exception("[REQUEST] Unhandled error while processing video")
        return jsonify({'error': 'Error processing video'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    output_folder = app.config['OUTPUT_FOLDER']
    # Only allow .mp4 files
    if not filename.lower().endswith('.mp4'):
        return jsonify({'error': 'File not found'}), 404
    file_path = os.path.join(output_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    # Stream the file as an attachment
    resp = send_from_directory(
        output_folder,
        filename,
        as_attachment=True,
        mimetype='video/mp4',
    )

    # Disable caching so every click always fetches the latest file
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"

    return resp

@app.route('/video/<filename>')
def view_video(filename):
    """Serve the processed video for preview (not as download)."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        resp = send_file(file_path, mimetype='video/mp4')
        # Disable caching so the preview ALWAYS matches the latest render
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    return jsonify({'error': 'File not found'}), 404



@app.route('/list-clips')
def list_clips():
    """List available clips and audio files."""
    clips = []
    audio_files = []

    if os.path.exists('clips'):
        clips = [f for f in os.listdir('clips') if allowed_file(f, ALLOWED_VIDEO_EXTENSIONS)]

    if os.path.exists('audio_files'):
        audio_files = [f for f in os.listdir('audio_files') if allowed_file(f, ALLOWED_AUDIO_EXTENSIONS)]

    return jsonify({
        'clips': clips,
        'audio_files': audio_files
    })


@app.route('/list-projects', methods=['GET'])
def list_projects():
    """List all project files from S3."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix='projects/'
        )
        
        projects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    # Extract project info from key
                    filename = obj['Key'].split('/')[-1]
                    projects.append({
                        'key': obj['Key'],
                        'filename': filename,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        's3_url': f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{obj['Key']}"
                    })
        
        # Sort by last modified (newest first)
        projects.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'projects': projects
        })
    except ClientError as e:
        print(f"Error listing projects from S3: {e}")
        return jsonify({'error': f'Error listing projects: {str(e)}'}), 500


@app.route('/load-project', methods=['POST'])
def load_project():
    """Load a project from S3 and return its data."""
    try:
        data = request.json
        project_key = data.get('project_key') or data.get('s3_url')
        
        if not project_key:
            return jsonify({'error': 'Project key or S3 URL is required'}), 400
        
        # Extract key from URL if full URL is provided
        if project_key.startswith('http'):
            # Extract key from URL: https://bucket.s3.region.amazonaws.com/projects/filename.json
            project_key = '/'.join(project_key.split('/')[3:])
        
        # Download project file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=project_key)
        project_data = json.loads(response['Body'].read().decode('utf-8'))
        
        return jsonify({
            'success': True,
            'project': project_data
        })
    except ClientError as e:
        print(f"Error loading project from S3: {e}")
        return jsonify({'error': f'Error loading project: {str(e)}'}), 500
    except Exception as e:
        print(f"Error parsing project: {e}")
        return jsonify({'error': f'Error parsing project: {str(e)}'}), 500


@app.route('/save-project', methods=['POST'])
def save_project():
    """Save a project to S3 without processing the video."""
    request_start = time.time()
    logger.info("[SAVE PROJECT] Received save project request")
    
    try:
        data = request.json

        video_path = data.get('video_path')
        highlights = data.get('highlights', [])
        transcript = data.get('transcript', [])
        subtitle_sentences = data.get('subtitle_sentences', [])
        aspect_ratio = data.get('aspect_ratio', '4:5')
        project_name = data.get('project_name', None)  # Optional custom name
        
        logger.info(f"[SAVE PROJECT] Highlights: {len(highlights)}, Transcript words: {len(transcript)}")

        # Validate aspect ratio
        if aspect_ratio not in ['4:5', '9:16']:
            aspect_ratio = '4:5'

        if not video_path:
            return jsonify({'error': 'Video path is required'}), 400

        # Create project JSON file
        if project_name:
            # Use custom name if provided
            project_filename = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            project_filename = f"project_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        project_data = create_project_json(
            video_path=video_path,
            highlights=highlights,
            transcript=transcript,
            subtitle_sentences=subtitle_sentences,
            aspect_ratio=aspect_ratio,
            output_filename=None,  # No output yet
            output_path=None  # No output yet
        )
        
        # Mark as draft/unsaved if not processed
        project_data['project_info']['status'] = 'draft'
        project_data['project_info']['saved_at'] = datetime.now().isoformat()
        
        # Save project JSON locally first
        project_path = os.path.join(app.config['OUTPUT_FOLDER'], project_filename)
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        # Upload to S3
        s3_project_url = None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_s3_key = f"projects/{timestamp}_{project_filename}"
            
            logger.info(f"[SAVE PROJECT] Uploading project to S3: {project_s3_key}")
            s3_start = time.time()
            s3_project_url = upload_file_to_s3(project_path, project_s3_key)
            s3_duration = time.time() - s3_start
            logger.info(f"[SAVE PROJECT] ✓ Project uploaded to S3 in {s3_duration:.2f}s - {s3_project_url}")
            
        except Exception as s3_error:
            logger.error(f"[SAVE PROJECT] S3 upload failed: {s3_error}")
            return jsonify({
                'error': f'S3 upload failed: {str(s3_error)}',
                'local_path': project_path
            }), 500

        total_time = time.time() - request_start
        logger.info(f"[SAVE PROJECT] ✓ Save completed in {total_time:.2f}s")
        
        return jsonify({
            'success': True,
            'project_filename': project_filename,
            'project_path': project_path,
            's3_project_url': s3_project_url,
            'message': 'Project saved successfully to S3!'
        })

    except Exception as e:
        logger.exception("[SAVE PROJECT] Error while saving project")
        return jsonify({'error': f'Error saving project: {str(e)}'}), 500


if __name__ == '__main__':
    # Disable reloader to prevent server restarts during video processing
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
