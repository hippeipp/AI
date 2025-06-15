import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from src.ocr_processor import VietnameseOCR
from src.preprocessor import ImagePreprocessor
import threading
import re

def clean_plate_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    corrections = {'O': '0', 'I': '1', 'Q': '0', 'B': '8', 'D': '0', 'U': '0', 'G': '6', 'Z': '2'}
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

class MotorbikePlateDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.ocr = VietnameseOCR()
        self.preprocessor = ImagePreprocessor()

    def _extract_text_with_timeout(self, plate_img, timeout=10):
        result = [None]
        def run_ocr():
            try:
                processed = self.ocr.preprocess_plate_image(plate_img)
                ocr_results = self.ocr.reader.readtext(processed, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-', detail=1)
                ocr_results = [r for r in ocr_results if r[2] > 0.15]
                if not ocr_results:
                    result[0] = None
                    return
                lines = {}
                for box, text, conf in ocr_results:
                    y_center = (box[0][1] + box[2][1]) / 2
                    assigned = False
                    for key in lines:
                        if abs(key - y_center) < 15:
                            lines[key].append(text)
                            assigned = True
                            break
                    if not assigned:
                        lines[y_center] = [text]
                sorted_keys = sorted(lines.keys())
                merged_lines = [''.join(lines[k]) for k in sorted_keys[:2]]
                full_text = ''.join(merged_lines)
                cleaned = clean_plate_text(full_text)
                result[0] = cleaned
            except Exception:
                result[0] = None
        thread = threading.Thread(target=run_ocr)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return None
        return result[0]

    def detect_plates(self, image: np.ndarray) -> List[Dict]:
        processed_image = self.preprocessor.preprocess_for_model(image)
        results = self.model.predict(processed_image, conf=self.confidence_threshold)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                pad = int((y2 - y1) * 0.2)
                x1 = max(x1 - pad, 0)
                y1 = max(y1 - pad, 0)
                x2 = min(x2 + pad, image.shape[1])
                y2 = min(y2 + pad, image.shape[0])
                plate_img = image[y1:y2, x1:x2]
                plate_text = self.ocr.extract_text(plate_img)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'text': plate_text
            })
        return detections

    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label_text = det['text'] if det['text'] else "Không nhận diện"
            label = f"{label_text} ({det['confidence']:.2f})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return annotated