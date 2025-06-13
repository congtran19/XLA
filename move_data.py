import os
import shutil
from pathlib import Path


def move_data(path):
    from pathlib import Path
    current_dir = os.getcwd()
    
    source_path = path

    # Tìm thư mục chứa ảnh (tên là "image" hoặc có chứa thư mục "image")
    image_dir_name = None
    metadata_file_name = None

    for root, dirs, files in os.walk(source_path):
        if "image" in dirs:
            image_dir_name = os.path.join(root, "image")
        for file in files:
            if file.endswith(".csv"):
                metadata_file_name = os.path.join(root, file)

    if not image_dir_name or not metadata_file_name:
        raise FileNotFoundError("Không tìm thấy ảnh hoặc metadata trong dataset!")

    # Đích đến
    image_dir = os.path.join(current_dir, "images")
    metadata_dir = os.path.join(current_dir, "metadata.csv")

    # Di chuyển dữ liệu
    shutil.copytree(image_dir_name, image_dir, dirs_exist_ok=True)
    shutil.copy(metadata_file_name, metadata_dir)

    return image_dir, metadata_dir
