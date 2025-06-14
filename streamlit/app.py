import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.plate_detector import MotorbikePlateDetector
from src.ocr_processor import VietnameseOCR
from src.preprocessor import ImagePreprocessor
import yaml

def format_plate(text):
    import re
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Định dạng lại biển số nếu đủ 8 ký tự
    if len(text) == 8:
        return f"{text[:4]}-{text[4:]}"
    return text

# Đọc cấu hình
with open("config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Khởi tạo bộ phát hiện biển số và OCR
plate_detector = MotorbikePlateDetector(
    model_path=config['paths']['model_path'], 
    confidence_threshold=config['model']['confidence_threshold']
)
ocr_processor = VietnameseOCR()

# Ứng dụng Streamlit
st.title("Nhận diện biển số xe máy")
st.write("Tải lên ảnh xe máy để nhận diện biển số.")

# Tải ảnh lên
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc và hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    st.write("")

    # Chuyển ảnh sang numpy array để xử lý
    image_np = np.array(image)

    # Tiền xử lý ảnh
    preprocessor = ImagePreprocessor()
    processed_image = preprocessor.preprocess_for_model(image_np)

    # Phát hiện biển số
    detections = plate_detector.detect_plates(processed_image)

    # Hiển thị kết quả
    if detections:
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            plate_text = format_plate(detection['text'])
            st.write(f"Biển số phát hiện: {plate_text} (Độ tin cậy: {confidence:.2f})")
            # Vẽ khung quanh biển số trên ảnh
            cv2.rectangle(image_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    else:
        st.write("Không phát hiện được biển số nào.")

    # Hiển thị ảnh đã đánh dấu
    st.image(image_np, caption='Ảnh đã nhận diện', use_container_width=True)