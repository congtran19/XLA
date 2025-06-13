import cv2
import numpy as np
import os
from tqdm import tqdm

def manual_hist_equalization_gray(img):
    # Bước 1: Tính histogram
    hist = np.zeros(256, dtype=int)  # Mỗi giá trị xám từ 0 đến 255 sẽ có số lần xuất hiện
    for pixel in img.ravel():        # Duyệt qua tất cả pixel và "làm phẳng" mảng 2D
        hist[pixel] += 1

    # Bước 2: Tính hàm phân phối tích lũy (CDF)
    cdf = np.cumsum(hist)  # Tính tổng tích lũy từ histogram

    # Bước 3: Chuẩn hóa CDF
    cdf_min = cdf[np.nonzero(cdf)].min()  # Trị nhỏ nhất khác 0 trong CDF (tránh chia cho 0)
    cdf_norm = ((cdf - cdf_min) * 255) / (cdf[-1] - cdf_min)  # Chuẩn hóa về [0, 255]
    cdf_norm = cdf_norm.astype('uint8')  # Ép kiểu về 8-bit

    # Bước 4: Ánh xạ pixel theo CDF đã chuẩn hóa
    img_eq = cdf_norm[img]  # Tra cứu lại từng pixel để thay đổi theo cdf mới

    return img_eq

def manual_hist_equalization_color(img):
    # Chuyển sang không gian YCrCb để chỉ xử lý độ sáng
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Cân bằng kênh Y (độ sáng)
    y_eq = manual_hist_equalization_gray(y)

    # Gộp lại và chuyển về BGR
    img_ycrcb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return img_eq

def process_directory_manual(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir), desc="Cân bằng histogram"):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Không thể đọc ảnh: {filename}")
            continue

        # Ảnh màu hay ảnh xám?
        if len(img.shape) == 3:
            img_eq = manual_hist_equalization_color(img)
        else:
            img_eq = manual_hist_equalization_gray(img)

        cv2.imwrite(output_path, img_eq)

    print("Hoàn thành xử lý tất cả ảnh.")

# Ví dụ sử dụng # process_directory_manual("thu_muc_goc", "thu_muc_ket_qua")
#process_directory_manual(r"E:\BTL_XLA\XLA\images", r"E:\BTL_XLA\XLA\anh_cb")
