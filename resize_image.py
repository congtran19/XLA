import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def resize_all_images(input_arr, size=(128, 128)):
    resized_arr=[]
    resized= np.zeros((128,128),dtype=np.uint8)
    try:
        for image in input_arr:
            resized = resize_image(image, size)
            resized_arr.append(resized)
    except Exception as e:
        print(f"Lỗi ảnh : {e}")

    print(len(resized_arr))
    return resized_arr

images_arr = np.load("images_arr.npy")
resize_arr = resize_all_images(images_arr)
resized_arr=np.stack(resize_arr, axis=0)
print("Kiểu dữ liệu:", resized_arr.dtype)
print(resized_arr.shape)
np.save('resized_arr',resized_arr)



# a = np.load('resized_arr.npy')
# print(a.shape)
#
# plt.figure(figsize=(10,8))
# plt.imshow(a[0],cmap='gray')
# plt.show()