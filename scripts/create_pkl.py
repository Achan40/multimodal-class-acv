import pickle
import numpy as np
import pandas as pd
from pathlib import Path

'''
This script contains functions use to create a .pkl file.
This .pkl file contains the textual data, unstructured and structured, that we can use to pass through our model.
Note that the CheXpert dataset that is being used does not contain unstructured data. Feel free to 
make changes to this code to fit your own dataset.
'''

# Returns a list of paths to all images in a specified directory
def image_path(folder_path):
    file_paths = []
    path = Path(folder_path)
    for file_path in path.rglob('*'):
        if file_path.is_file():
            file_paths.append(str(file_path))
    return file_paths

# Sends a dictionary to a .pkl file
# Specify output location in the file name
def create_pkl(dct, filename='./data/predev/dct.pkl'):
    with open(filename, 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Returns the relevant portion of some path. Very specific to the way our dataset is structured
def convert_path(input_path):
    # Replace backslashes with forward slashes
    updated_path = input_path.replace('\\', '/')

    # Extract the relevant part of the path
    relevant_part = '/'.join(updated_path.split('/')[-4:])

    return relevant_part

# Data wrangling. Converting some columns to binary.
def data_wrangling(file):
    df = pd.read_csv(file)

    df["Sex"] = df["Sex"].replace({'Male': 1, 'Female': 0, 'Unknown': 1}) # generate binary indicators
    df["Frontal/Lateral"] = df["Frontal/Lateral"].replace({'Frontal': 1, 'Lateral': 0}) 

    return df

# Creating the dictionary. This function is the bulk of how we setup our textual data, is highly specific to the datasource.
def create_dct(df, arr):
    d = {}

    for path in arr:
        p = convert_path(path)
        key = 'CheXpert-v1.0/' + p
        
        row = df.loc[df['Path'] == key].to_dict()
        row_ind = df.loc[df['Path'] == key].index.item()

        labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices', 'No Finding']

        d[p] = {
            'pdesc': np.ones((1, 40, 768)), # tokens for unstructured data. Setting all values equal to 1 since we don't have any unstructured text in our dataset.
            'bics': np.array([row['Sex'][row_ind], row['Age'][row_ind]]), 
            'bts': np.array([row['Frontal/Lateral'][row_ind]]),
            'label': np.array([row[i][row_ind] for i in labels])
        } 
    
    return d

'''
See README.md in the data folder for additional notes on directory structure.
Running this script will output a .pkl file you can pass as input to the model.
Run multiple times for your train, validation and test sets.
'''

d_set = 'train'

arr = image_path('./data/predev/'+d_set)
df = data_wrangling('./data/predev/'+d_set+'.csv')

dct = create_dct(df=df, arr=arr)
create_pkl(dct=dct, filename='./data/predev/train.pkl')