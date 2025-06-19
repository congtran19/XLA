import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def read_image(image_dir):
    image_files = sorted(os.listdir(image_dir)) 
    images_path = []

    for image in image_files:
        image_path = os.path.join(image_dir, image)
        images_path.append(image_path)

    images_matrix = []
    for image in images_path:
        matrix = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if matrix is not None:
            print('Đang đọc ảnh:', image)
            images_matrix.append(matrix)
        else:
            print(f"Không thể đọc ảnh: {image}")

    print(f"Số ảnh đã đọc: {len(images_matrix)}")

    try:
        images_array = np.stack(images_matrix, axis=0)
    except Exception as e:
        print("Lỗi khi stack ảnh:", e)
        return None

    images_array = images_array.astype(np.uint8)
    print("Kích thước mảng:", images_array.shape)
    print("Kiểu dữ liệu:", images_array.dtype)

    return images_array

images_array = read_image(image_dir=r"C:\Users\NguyenNgocHuy\.cache\kagglehub\datasets\nischaydnk\isic-2018-jpg-224x224-resized\versions\2\train-image\image")
np.save('images_arr.npy', images_array)

a = np.load('images_arr.npy')
print(a.shape)
