import os
import sys
import io
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from PIL import Image, ImageDraw
from app import app
from services.image_detector import ImageDetectorService

def verify_consistency():
    print("=" * 60)
    print("VERIFYING INFERENCE PIPELINE CONSISTENCY (CLI vs API)")
    print("=" * 60)

    # Generate test image buffer
    img = Image.new('RGB', (300, 300), color=(120, 80, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 250, 250], fill=(200, 150, 50))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Save to temp file for CLI service
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, 'consistency_test.png')
    with open(temp_filepath, 'wb') as f:
        f.write(img_bytes)

    # 1. CLI Service Prediction
    cli_service = ImageDetectorService()
    cli_result = cli_service.predict_image(temp_filepath)

    # 2. Flask REST API Prediction via Test Client
    client = app.test_client()
    api_response = client.post(
        '/api/detection/analyze',
        data={'file': (io.BytesIO(img_bytes), 'consistency_test.png')},
        content_type='multipart/form-data'
    )
    api_result = api_response.get_json()

    print(f"CLI Prediction Result: {cli_result['result']} | AI Prob: {cli_result['ai_probability']}% | Real Prob: {cli_result['real_probability']}%")
    print(f"API Prediction Result: {api_result['result']} | AI Prob: {api_result['ai_probability']}% | Real Prob: {api_result['real_probability']}%")

    # Assert match within tolerance 1e-4
    ai_prob_diff = abs(cli_result['ai_probability'] - api_result['ai_probability'])
    real_prob_diff = abs(cli_result['real_probability'] - api_result['real_probability'])

    print(f"Probability Differences: AI Delta = {ai_prob_diff:.6f}, Real Delta = {real_prob_diff:.6f}")

    assert api_response.status_code == 200, f"API returned non-200 code: {api_response.status_code}"
    assert cli_result['result'] == api_result['result'], "Mismatch between CLI and API classification result!"
    assert ai_prob_diff < 1e-4, f"AI Probability mismatch exceeds tolerance: {ai_prob_diff}"
    assert real_prob_diff < 1e-4, f"Real Probability mismatch exceeds tolerance: {real_prob_diff}"

    # Clean temp file
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)

    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE VERIFICATION PASSED PERFECTLY (DELTA < 1e-4)")
    print("=" * 60)

if __name__ == '__main__':
    verify_consistency()
