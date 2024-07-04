import os
import shutil
import requests
from roboflow import Roboflow
import subprocess
from ultralytics import YOLOv10
import supervision as sv
import cv2

# Define paths
base_dir = 'D:/DetectObject4(17k)'
model_dir = os.path.join(base_dir, 'YOLOv10')
dataset_dir = os.path.join(base_dir, 'DataOBJ')
train_dir = os.path.join(base_dir, 'yolov10_train')

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# # Download YOLOv10 model
# model_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt"
# model_path = os.path.join(model_dir, 'yolov10n.pt')
# response = requests.get(model_url)
# with open(model_path, 'wb') as file:
#     file.write(response.content)
# print("Model downloaded successfully.")

# # Initialize Roboflow and download dataset
# rf = Roboflow(api_key="lIdJ2QWF2eGYlThGfEfy")
# project = rf.workspace("jacob-solawetz").project("pascal-voc-2012")
# version = project.version(1)
# dataset = version.download("yolov8")

# # Adjust data.yaml for local paths
# data_yaml_path = os.path.join(dataset.location, "data.yaml")
# with open(data_yaml_path, 'r') as file:
#     lines = file.readlines()

# # Remove last four lines
# lines = lines[:-4]

# # Append new paths
# lines.append("test: ../test/images\n")
# lines.append("train: ../train/images\n")
# lines.append("val: ../valid/images\n")

# # Write back to data.yaml
# with open(data_yaml_path, 'w', encoding='utf-8') as file:  # Specify encoding as 'utf-8'
#     file.writelines(lines)
# print("data.yaml updated successfully.")

# # Train the model using GPU
# model_path = os.path.join(model_dir, 'yolov10n.pt')
# data_path = os.path.join(dataset.location, 'data.yaml')

# # Construct the training command
# train_command = f"yolo task=detect mode=train epochs=10 batch=32 plots=True model={model_path} data={data_path} device=0"

# # Execute the training command
# os.system(train_command)

# Load the YOLOv10 model
model = YOLOv10(r'D:\DetectObject4(17k)\runs\detect\train2\weights\best.pt')

# Define the dataset path
dataset_path = r"D:\DetectObject4(17k)\runs\detect\train2"

# Load the dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=r"D:\DetectObject4(17k)\Pascal-VOC-2012-1\valid\images",
    annotations_directory_path=r"D:\DetectObject4(17k)\Pascal-VOC-2012-1\valid\labels",
    data_yaml_path=r"D:\DetectObject4(17k)\Pascal-VOC-2012-1\data.yaml"
)

# Create annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Predict and display results
subprocess.run([
    "yolo", "predict",
    f"model={r'D:\\DetectObject4(17k)\\runs\\detect\\train2\\weights\\best.pt'}",
    f"source={r'D:\\DetectObject4(17k)\\output\\anh1.jpg'}",
    f"name=testv10"
])


