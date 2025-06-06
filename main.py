import kagglehub
import os
import shutil
import cv2
from read_metadata import read_metadata
from move_data import move_data
from read_image import read_image

# Tai dataset 
path = kagglehub.dataset_download("nischaydnk/isic-2018-jpg-224x224-resized")
print("Path to dataset files:", path)


#di chuyen data sang thu muc XLA
image_dir,metadata_dir= move_data(path)
print(image_dir,metadata_dir)

label = read_metadata(metadata_path=metadata_dir)
images_array = read_image(image_dir=image_dir)

print(images_array[1:5])