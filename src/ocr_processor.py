import easyocr
import cv2
import numpy as np
import re
from typing import Optional

class VietnameseOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.patterns = [
            r'^[0-9]{2}[A-Z]{1,2}-[0-9]{3,5}$',
            r'^[0-9]{2}[A-Z]{1,2}[0-9]{3,5}$'
        ]

    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        # Chuyển sang grayscale nếu cần
        if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # Tăng cường tương phản bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Resize lớn hơn để OCR tốt hơn
        resized = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text(self, plate_img: np.ndarray) -> Optional[str]:
        try:
            processed = self.preprocess_plate_image(plate_img)
            results = self.reader.readtext(processed, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-', detail=0)

            if not results or len(results) == 0:
                return None

            # Ghép các kết quả lại thành một chuỗi
            full_text = ''.join(results)
            cleaned = self.clean_text(full_text)
            return cleaned

        except Exception as e:
            print(f"❌ Lỗi OCR: {e}")
            return None

    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r'[^A-Z0-9-]', '', text.upper())
        corrections = {
            'O': '0',
            'I': '1',
            'Q': '0',
            'H': '-' 
        }
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        return cleaned

    def validate_plate_format(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.patterns)