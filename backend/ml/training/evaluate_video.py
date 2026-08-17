import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.inference.evaluate_video import run_video_evaluation
except ImportError:
    from ml.inference.evaluate_video import run_video_evaluation

if __name__ == '__main__':
    run_video_evaluation()
