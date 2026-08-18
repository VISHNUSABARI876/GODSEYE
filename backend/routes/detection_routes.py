import os
import sys
import logging
import traceback
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from models import db, Detection
from services.image_detector import ImageDetectorService
from services.video_detector import VideoDetectorService

logger = logging.getLogger(__name__)

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


@detection_bp.route('/status', methods=['GET'])
def get_detection_status():
    """
    GET /api/detection/status
    Returns status of ML dependencies, PyTorch runtime, model files, and device availability.
    """
    try:
        import torch
        import cv2
        import PIL
        import numpy

        model_path = current_app.config.get('MODEL_PATH')
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(base_dir, 'ml', 'models')
        
        spatial_path = os.path.abspath(model_path) if model_path else os.path.abspath(os.path.join(models_dir, 'best_model.pth'))
        temporal_path = os.path.abspath(os.path.join(models_dir, 'video_temporal_model.pth'))

        spatial_exists = os.path.exists(spatial_path)
        temporal_exists = os.path.exists(temporal_path)

        return jsonify({
            'success': True,
            'torch_version': torch.__version__,
            'torch_cuda_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'opencv_version': cv2.__version__,
            'pillow_version': PIL.__version__,
            'numpy_version': numpy.__version__,
            'models': {
                'spatial_model_path': spatial_path,
                'spatial_model_exists': spatial_exists,
                'spatial_model_size_mb': round(os.path.getsize(spatial_path) / (1024 * 1024), 2) if spatial_exists else 0,
                'temporal_model_path': temporal_path,
                'temporal_model_exists': temporal_exists,
                'temporal_model_size_mb': round(os.path.getsize(temporal_path) / (1024 * 1024), 2) if temporal_exists else 0
            }
        }), 200
    except Exception as e:
        logger.error(f"Error checking model status: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error_type': type(e).__name__,
            'message': str(e)
        }), 500


@detection_bp.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_file():
    """
    POST /api/detection/analyze
    Accepts multipart/form-data with key 'file'.
    Performs PyTorch EfficientNet-B0 + 2D FFT ML detection and saves record to Neon PostgreSQL.
    """
    logger.info(f"{request.method} /api/detection/analyze - Origin: {request.headers.get('Origin')}")

    if request.method == 'OPTIONS':
        return '', 200

    raw_filename = "unknown"
    ext = "unknown"
    file_mimetype = "unknown"

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file parameter provided in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        raw_filename = secure_filename(file.filename)
        ext = raw_filename.rsplit('.', 1)[-1].lower() if '.' in raw_filename else ''
        file_mimetype = getattr(file, 'content_type', getattr(file, 'mimetype', 'unknown'))

        allowed_images = current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', {'png', 'jpg', 'jpeg', 'webp', 'bmp'})
        allowed_videos = current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', {'mp4', 'avi', 'mov', 'mkv', 'webm'})

        if ext not in allowed_images and ext not in allowed_videos:
            return jsonify({
                'success': False,
                'message': f"Unsupported file extension .{ext}. Allowed image types: {sorted(list(allowed_images))}. Allowed video types: {sorted(list(allowed_videos))}."
            }), 400

        # Path-safe absolute upload directory creation
        upload_dir = current_app.config.get('UPLOAD_FOLDER')
        if not upload_dir or not os.path.isabs(upload_dir):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            upload_dir = os.path.abspath(os.path.join(base_dir, upload_dir or 'uploads'))

        os.makedirs(upload_dir, exist_ok=True)

        # Save file with unique timestamp
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp_str}_{raw_filename}"
        file_path = os.path.join(upload_dir, saved_filename)
        file.save(file_path)

        media_type = 'image' if ext in allowed_images else 'video'

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

        # Attempt saving analysis record to Neon PostgreSQL
        detection_id = None
        db_saved = False
        try:
            detection_record = Detection(
                user_id=None,
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
            detection_id = detection_record.id
            db_saved = True
        except Exception as db_err:
            db.session.rollback()
            logger.warning(f"Database save skipped or failed for '{raw_filename}': {db_err}")

        return jsonify({
            'success': True,
            'detection_id': detection_id,
            'db_saved': db_saved,
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
            'created_at': datetime.now(timezone.utc).isoformat(),
            'message': 'Analysis completed successfully.'
        }), 200

    except Exception as e:
        db.session.rollback()
        exc_type = type(e).__name__
        exc_msg = str(e)
        tb_str = traceback.format_exc()
        content_type_hdr = request.headers.get('Content-Type', 'unknown')

        logger.error(
            f"DETECTION ANALYSIS ERROR 500: [{exc_type}] {exc_msg}\n"
            f"Uploaded File: '{raw_filename}' (ext: '{ext}', mimetype: '{file_mimetype}', content-type header: '{content_type_hdr}')\n"
            f"Traceback:\n{tb_str}"
        )

        return jsonify({
            'success': False,
            'error_type': exc_type,
            'message': f"Detection analysis failed: [{exc_type}] {exc_msg}"
        }), 500


@detection_bp.route('/supported-formats', methods=['GET'])
def get_supported_formats():
    return jsonify({
        'images': sorted(list(current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', []))),
        'videos': sorted(list(current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', [])))
    }), 200
