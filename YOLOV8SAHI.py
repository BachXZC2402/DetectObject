import torch
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_sliced_prediction, predict
from sahi.prediction import visualize_object_predictions
from IPython.display import Image
import numpy as np
import cv2
import os

# # Install necessary packages
# !pip install -U scikit-image imagecodecs
# !pip install -q torch sahi yolov8 ultralytics numpy opencv-python

# Define model path and download YOLOv8 model
yolov8x_model_path = 'models/yolov8x.pt'
# download_yolov8s_model(yolov8x_model_path)

# Load the model
detection_model1 = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8x_model_path,
    confidence_threshold=0.5,
    device='cuda:0'
)

# Perform prediction using SAHI with sliced image
results = get_sliced_prediction(
    r'D:\DetectObject4(17k)\runs\detect\testv10\anh1.jpg',
    detection_model=detection_model1,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Export and visualize results
results.export_visuals(r'D:\DetectObject4(17k)\runs\detect\testv8SAHI')
Image(r'D:\DetectObject4(17k)\runs\detect\testv8SAHI\anh1.jpg')


# results.export_visuals(r'D:\DetectObject4(17k)\runs\detect\testv8SAHI')

# folder_path = r'D:\DetectObject4(17k)\runs\detect\testv8SAHI'
# files = os.listdir(folder_path)

# for idx, file in enumerate(files):
#     if file.endswith(".jpg"):  # Chỉ đổi tên các tệp .jpg
#         original_path = os.path.join(folder_path, file)
#         new_name = f"image_{idx+1}.jpg"
#         new_path = os.path.join(folder_path, new_name)
#         os.rename(original_path, new_path)


# from IPython.display import Image
# Image(r'D:\DetectObject4(17k)\runs\detect\testv8SAHI\image_1.jpg')
