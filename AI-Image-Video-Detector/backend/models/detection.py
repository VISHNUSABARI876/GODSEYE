from datetime import datetime, timezone
from models.database import db

class Detection(db.Model):
    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(50), nullable=False)  # 'image' or 'video'
    result = db.Column(db.String(50), nullable=False)      # 'AI-Generated' or 'Real'
    ai_probability = db.Column(db.Float, nullable=False)
    real_probability = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    frames_analyzed = db.Column(db.Integer, nullable=True, default=1)
    suspicious_frames = db.Column(db.Integer, nullable=True, default=0)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'media_type': self.media_type,
            'result': self.result,
            'ai_probability': self.ai_probability,
            'real_probability': self.real_probability,
            'confidence': self.confidence,
            'frames_analyzed': self.frames_analyzed,
            'suspicious_frames': self.suspicious_frames,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f"<Detection {self.id} - {self.filename} ({self.result})>"
