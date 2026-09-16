import cv2
import math
import numpy as np

class FrameExtractor:
    def __init__(self, target_sample_fps=1.0, max_frames=20):
        self.target_sample_fps = target_sample_fps
        self.max_frames = max_frames

    def extract_frames(self, video_path):
        """
        Extracts representative video frames from video_path.
        Returns:
            frames: list of numpy arrays (BGR OpenCV frames)
            metadata: dict containing fps, width, height, total_frames, duration_seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid frame count in video: {video_path}")

        # Calculate sampling interval
        step = max(1, int(fps / self.target_sample_fps))
        sample_indices = list(range(0, total_frames, step))

        # If sample count exceeds max_frames, pick max_frames uniformly
        if len(sample_indices) > self.max_frames:
            uniform_indices = np.linspace(0, total_frames - 1, num=self.max_frames, dtype=int)
            sample_indices = list(dict.fromkeys(uniform_indices))

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()

        metadata = {
            'fps': round(float(fps), 2),
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration_seconds': round(float(duration), 2),
            'extracted_count': len(frames)
        }

        return frames, metadata
