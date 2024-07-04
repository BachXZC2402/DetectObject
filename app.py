import streamlit as st
import cv2
import os
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLOv10
import shutil

# Define paths
base_dir = r'D:\DetectObject4(17k)'  # Ensure this directory exists
preprocessed_image_name = 'preprocessed_image.jpg'
yolov8_output_dir_name = 'yolov8sahi_output'
yolov10_output_name = 'yolov10_output.jpg'
combined_output_name = 'combined_output.jpg'

preprocessed_image_path = os.path.join(base_dir, preprocessed_image_name)
yolov10_output_path = os.path.join(base_dir, yolov10_output_name)
yolov8_output_dir = os.path.join(base_dir, yolov8_output_dir_name)
combined_output_path = os.path.join(base_dir, combined_output_name)

# Ensure output directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(yolov8_output_dir, exist_ok=True)

# Define preprocess function
def preprocess_image(input_path, output_path, target_size=(2560, 2560)):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image from {input_path}")
    resized_image = cv2.resize(image, target_size)
    denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)
    cv2.imwrite(output_path, sharpened_image)
    print(f"Preprocessed image saved to {output_path}")

# YOLOv8 with SAHI Detection
def yolov8sahi_detection(image_path, output_dir):
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path='models/yolov8x.pt', 
        confidence_threshold=0.5, 
        device='cuda:0'
    )
    results = get_sliced_prediction(
        image_path, 
        detection_model=model, 
        slice_height=512, 
        slice_width=512, 
        overlap_height_ratio=0.2, 
        overlap_width_ratio=0.2
    )
    results.export_visuals(output_dir)
    visuals = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]
    if not visuals:
        raise ValueError(f"YOLOv8+SAHI detection failed, no output images found in {output_dir}")
    return os.path.join(output_dir, visuals[0])  # Return the first image path

# YOLOv10 Detection
def yolov10_detection(image_path, output_path):
    print(f"Loading YOLOv10 model from path: {r'D:\DetectObject4(17k)\runs\detect\train2\weights\best.pt'}")
    model = YOLOv10(r'D:\DetectObject4(17k)\runs\detect\train2\weights\best.pt')
    results = model.predict(source=image_path, save=True, save_txt=False)
    
    if not results or len(results) == 0:
        raise ValueError("YOLOv10 detection failed, no results returned.")
    
    # Find the latest 'predict' directory
    runs_dir = os.path.join(base_dir, 'runs', 'detect')
    predict_dirs = [d for d in os.listdir(runs_dir) if d.startswith('predict')]
    if not predict_dirs:
        raise ValueError("No 'predict' directories found in runs/detect")

    latest_predict_dir = max(predict_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
    latest_predict_dir_path = os.path.join(runs_dir, latest_predict_dir)
    
    detected_image_path = None
    for file in os.listdir(latest_predict_dir_path):
        if file.endswith(('.jpg', '.png')):
            detected_image_path = os.path.join(latest_predict_dir_path, file)
            break
    
    if detected_image_path is None:
        raise ValueError(f"YOLOv10 detection failed, no output image found in {latest_predict_dir_path}")

    shutil.move(detected_image_path, output_path)
    print(f"YOLOv10 output moved to {output_path}")
    return output_path

# Combine YOLOv8 + SAHI and YOLOv10 Results
def combine_detections(yolov8sahi_image_path, yolov10_image_path, output_path):
    yolov8sahi_img = cv2.imread(yolov8sahi_image_path)
    yolov10_img = cv2.imread(yolov10_image_path)
    if yolov8sahi_img is None or yolov10_img is None:
        raise ValueError("Error reading the detection result images.")
    combined_img = cv2.addWeighted(yolov8sahi_img, 0.5, yolov10_img, 0.5, 0)
    cv2.imwrite(output_path, combined_img)
    print(f"Combined output saved to {output_path}")
    return output_path

# Streamlit App
st.title("Object Detection Interface")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file
    input_image_path = os.path.join(base_dir, 'input_image.jpg')
    with open(input_image_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button('Process Image'):
        try:
            # Preprocess Image
            preprocess_image(input_image_path, preprocessed_image_path)
            st.image(preprocessed_image_path, caption='Preprocessed Image.', use_column_width=True)

            # YOLOv8 + SAHI Detection
            yolov8_output_path = yolov8sahi_detection(preprocessed_image_path, yolov8_output_dir)
            st.image(yolov8_output_path, caption='YOLOv8+SAHI Detection Results.', use_column_width=True)

            # YOLOv10 Detection
            yolov10_output_path = yolov10_detection(preprocessed_image_path, yolov10_output_path)
            st.image(yolov10_output_path, caption='YOLOv10 Detection Results.', use_column_width=True)

            # Combine Results
            combined_output_path = combine_detections(yolov8_output_path, yolov10_output_path, combined_output_path)
            st.image(combined_output_path, caption='Combined Detection Results.', use_column_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

        # Clean up temporary files
        try:
            if os.path.exists(preprocessed_image_path):
                os.remove(preprocessed_image_path)
                print(f"Removed preprocessed image: {preprocessed_image_path}")
            else:
                print(f"Preprocessed image not found: {preprocessed_image_path}")

            if os.path.exists(yolov10_output_path):
                os.remove(yolov10_output_path)
                print(f"Removed YOLOv10 output image: {yolov10_output_path}")
            else:
                print(f"YOLOv10 output image not found: {yolov10_output_path}")

            if os.path.exists(yolov8_output_dir):
                shutil.rmtree(yolov8_output_dir)
                print(f"Removed YOLOv8 output directory: {yolov8_output_dir}")
            else:
                print(f"YOLOv8 output directory not found: {yolov8_output_dir}")
        except Exception as e:
            st.error(f"Error during cleanup: {e}")
