import cv2
import numpy as np
import matplotlib.pyplot as plt
def manual_hist_equalization_gray(img):
    img = img.astype(np.uint8)  # Đảm bảo ảnh là kiểu uint8 để truy cập histogram

    # Bước 1: Tính histogram
    hist = np.zeros(256, dtype=int)
    for pixel in img.ravel():
        hist[pixel] += 1

    # Bước 2: Tính hàm phân phối tích lũy (CDF)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)].min()

    # Bước 3: Chuẩn hóa CDF về [0, 255]
    cdf_norm = ((cdf - cdf_min) * 255) / (cdf[-1] - cdf_min)
    cdf_norm = cdf_norm.astype(np.uint8)

    # Bước 4: Tra cứu ánh xạ
    img_eq = cdf_norm[img]
    return img_eq


def process_directory_manual(input_arr):
    after_processing = []
    for img in input_arr:
        img_eq = manual_hist_equalization_gray(img)
        after_processing.append(img_eq)

    after_processing = np.stack(after_processing, axis=0)
    print("Shape:", after_processing.shape)
    print("Dtype:", after_processing.dtype)
    return after_processing

images_arr = np.load("resized_arr.npy")
afterprocessing_arr = process_directory_manual(images_arr)
np.save('afterprocessing_arr.npy', afterprocessing_arr)
