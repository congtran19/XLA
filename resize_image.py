import os
import cv2
import numpy as np

def resize_image(image, new_size):
    original_height, original_width = image.shape[:2]
    new_height, new_width = new_size

    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized = np.zeros((new_height, new_width), dtype=image.dtype)

    row_ratio = original_height / new_height
    col_ratio = original_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i * row_ratio)
            orig_j = int(j * col_ratio)
            resized[i, j] = image[orig_i, orig_j]

    return resized

def resize_all_images(input_folder, output_folder, size=(224, 224), grayscale=False):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            if grayscale:
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(input_path)

            if img is None:
                print(f"Không đọc được ảnh: {input_path}")
                continue

            resized = resize_image(img, size)
            cv2.imwrite(output_path, resized)
            print(f"Đã resize ảnh: {output_path}")

        except Exception as e:
            print(f"Lỗi ảnh {input_path}: {e}")
