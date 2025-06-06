import os
import shutil

def move_data(path) :
    from pathlib import Path 
    current_dir = os.getcwd()
    print(current_dir)
    
    # Đường dẫn gốc file tải về
    source_path = path  
    data_path = []
    for f in os.listdir(path):
        data_path.append(f)
    print(data_path[-1])
    images_path = os.path.join(source_path, data_path[-1],"image") 
    metadata_path = os.path.join(source_path,data_path[0])
    image_dir = os.path.join(current_dir,"images")
    print(image_dir)
    
    metadata_dir = os.path.join(current_dir,"metadata")
    print(metadata_dir)

    # Copy file về thư mục đang làm
    shutil.copytree(images_path,image_dir)
    shutil.copy(metadata_path,metadata_dir)
    

    return image_dir, metadata_dir

    

    
    