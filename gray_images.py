import os
from PIL import Image

# thư mục ảnh gốc
input_folder = "E:\Workspace\BTL_XLA\images"
# thư mục lưu ảnh xám
output_folder = "E:\Workspace\BTL_XLA\gray_images_output"

# tạo thư mục lưu ảnh đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)


def gray_images():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_gray.jpg")

            try:
                # mở ảnh và chuyển sang ảnh xám
                img = Image.open(input_path).convert("L")  # "L" là chế độ grayscale
                img.save(output_path, "JPEG")
                print(f"Đã chuyển ảnh: {filename}")
            except Exception as e:
                print(f"Lỗi với ảnh {filename}: {e}")
