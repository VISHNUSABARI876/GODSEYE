import os
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from models import db, Detection
from services.image_detector import ImageDetectorService
from services.video_detector import VideoDetectorService

detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')

# Global service instances
_image_detector = None
_video_detector = None

def get_image_detector():
    global _image_detector
    model_path = current_app.config.get('MODEL_PATH')
    if _image_detector is None or getattr(_image_detector, 'model_path', None) != model_path:
        _image_detector = ImageDetectorService(model_path)
    return _image_detector

def get_video_detector():
    global _video_detector
    model_path = current_app.config.get('MODEL_PATH')
    if _video_detector is None:
        _video_detector = VideoDetectorService(model_path)
    return _video_detector


@detection_bp.route('/analyze', methods=['POST'])
def analyze_file():
    """
    POST /api/detection/analyze
    Accepts multipart/form-data with key 'file'.
    Performs PyTorch EfficientNet-B0 + 2D FFT ML detection and saves record to Neon PostgreSQL.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file parameter provided in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    raw_filename = secure_filename(file.filename)
    ext = raw_filename.rsplit('.', 1)[-1].lower() if '.' in raw_filename else ''

    allowed_images = current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', {'png', 'jpg', 'jpeg', 'webp', 'bmp'})
    allowed_videos = current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', {'mp4', 'avi', 'mov', 'mkv', 'webm'})

    if ext not in allowed_images and ext not in allowed_videos:
        return jsonify({
            'success': False,
            'message': f"Unsupported file extension .{ext}. Allowed image types: {sorted(list(allowed_images))}. Allowed video types: {sorted(list(allowed_videos))}."
        }), 400

    # Ensure uploads directory exists
    upload_dir = current_app.config.get('UPLOAD_FOLDER')
    os.makedirs(upload_dir, exist_ok=True)

    # Save file with unique timestamp
    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{timestamp_str}_{raw_filename}"
    file_path = os.path.join(upload_dir, saved_filename)
    file.save(file_path)

    media_type = 'image' if ext in allowed_images else 'video'

    try:
        if media_type == 'image':
            detector = get_image_detector()
            analysis = detector.predict_image(file_path)
            frames_analyzed = 1
            suspicious_frames = 1 if analysis['result'] == 'AI-Generated' else 0
        else:
            detector = get_video_detector()
            analysis = detector.predict_video(file_path)
            frames_analyzed = analysis.get('frames_analyzed', 1)
            suspicious_frames = analysis.get('suspicious_frames', 0)

        # Save analysis record to Neon PostgreSQL
        detection_record = Detection(
            user_id=None,  # Guest detection or logged-in user in future auth step
            filename=raw_filename,
            media_type=media_type,
            result=analysis['result'],
            ai_probability=analysis['ai_probability'],
            real_probability=analysis['real_probability'],
            confidence=analysis['confidence'],
            frames_analyzed=frames_analyzed,
            suspicious_frames=suspicious_frames
        )

        db.session.add(detection_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'detection_id': detection_record.id,
            'filename': raw_filename,
            'media_type': media_type,
            'result': analysis['result'],
            'label': analysis.get('label', analysis['result']),
            'ai_probability': analysis['ai_probability'],
            'real_probability': analysis['real_probability'],
            'confidence': analysis['confidence'],
            'confidence_category': analysis.get('confidence_category', 'High Confidence'),
            'threshold_used': analysis.get('threshold_used', 0.50),
            'frames_analyzed': frames_analyzed,
            'suspicious_frames': suspicious_frames,
            'created_at': detection_record.created_at.isoformat() if detection_record.created_at else None,
            'message': 'Analysis completed and saved to Neon PostgreSQL.'
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error analyzing file {raw_filename}: {e}")
        return jsonify({
            'success': False,
            'message': f"Detection analysis failed: {str(e)}"
        }), 500


@detection_bp.route('/supported-formats', methods=['GET'])
def get_supported_formats():
    return jsonify({
        'images': sorted(list(current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', []))),
        'videos': sorted(list(current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', [])))
    }), 200
