def read_metadata(metadata_path):
    import pandas as pd
    #Doc meta_data
    df = pd.read_csv(filepath_or_buffer=metadata_path)

    #Sap xep lai data
    df.sort_values(by="isic_id",ascending=True,inplace=True)
    # print(df.head())

    #Mo ta metadata 
    # print(df.info())
    # print(df.shape)
    # print(df.isnull().sum())


    # Lay ten va nhan
    name = df['isic_id']
    label = df['target']

    return label

