import os 
import pandas as pd
import numpy as np
def read_metadata(metadata_path,image_path):
    
    #Doc meta_data
    df = pd.read_csv(filepath_or_buffer=metadata_path)

    imgs_name=[]
    for img in os.listdir(image_path):
        img_name = img.split(".")[0]
        imgs_name.append(img_name)

    # print(imgs_name[:10])
    # print(len(imgs_name))
    

    # Cách 2: Dùng phủ định (~) để xóa dòng không khớp
    df = df[df['isic_id'].isin(imgs_name)].sort_values('isic_id')
    print(df.shape)
    print(df[-10:])
    df1= df['target']

    return df1

labels = read_metadata(metadata_path='metadata.csv',image_path='images')
np.save('labels',labels)