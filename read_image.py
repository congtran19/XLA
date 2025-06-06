import cv2
import os
import numpy as np
def read_image(image_dir):
    images_path= []
    try:
        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir,image)
            images_path.append(image_path)
    except Exception as e: 
        print(e)

    images_matrix = []
    try:
        for image in images_path:
            matrix = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            if matrix is not None:
                images_matrix.append(matrix)
            else:
                print(f"Không thể đọc ảnh: {image}")
    except Exception as e: 
        print(e)

    print(f"Số ảnh đã đọc: {len(images_matrix)}")
    images_array = np.stack(images_matrix, axis=0)  
    print(images_array.shape)

    return images_array


