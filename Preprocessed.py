import cv2
import os
import numpy as np

def preprocess_image(input_path, output_path, target_size=(2560, 2560)):
    # Đọc ảnh đầu vào
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image from {input_path}")
    print(f"Đã đọc thành công ảnh từ {input_path}")

    # Resize ảnh về kích thước mong muốn
    resized_image = cv2.resize(image, target_size)
    print(f"Đã thay đổi kích thước ảnh về {target_size}")

    # Khử nhiễu ảnh sử dụng GaussianBlur
    denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    print("Đã khử nhiễu ảnh")

    # Làm sắc nét ảnh
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)
    print("Đã làm sắc nét ảnh")

    # Lưu ảnh đã tiền xử lý
    cv2.imwrite(output_path, sharpened_image)
    print(f"Đã lưu ảnh tiền xử lý tại {output_path}")

# Đường dẫn ảnh đầu vào và ảnh sau khi tiền xử lý
input_image_path = r'D:\DetectObject4(17k)\input\noise.jpg'
preprocessed_image_path = r'D:\DetectObject4(17k)\output\noise.jpg'

# # Kiểm tra xem thư mục đích có tồn tại không, nếu không thì tạo ra
# output_dir = os.path.dirname(preprocessed_image_path)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#     print(f"Đã tạo thư mục đích tại {output_dir}")

# Tiền xử lý ảnh đầu vào
preprocess_image(input_image_path, preprocessed_image_path)
