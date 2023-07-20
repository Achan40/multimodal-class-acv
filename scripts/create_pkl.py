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
def convert_path(input_path, offset=-4):
    # Replace backslashes with forward slashes
    updated_path = input_path.replace('\\', '/')

    # Extract the relevant part of the path
    relevant_part = '/'.join(updated_path.split('/')[offset:])

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

        '''
        items in the datafiles indexed as follows:
        valid.csv: CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg
        train.csv: CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
        '''
        key = 'CheXpert-v1.0/' + p

        # test.pkl will need to get index from train.csv, have to retrieve by key first.
        if 'test' in p:
            p = convert_path(path, offset=-3)
            key = 'CheXpert-v1.0/train/' + p

        row = df.loc[df['Path'] == key].to_dict()
        row_ind = df.loc[df['Path'] == key].index.item()

        labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices', 'No Finding']
        
        '''
        image files are accessed later on by dictionary key.
        '''
        d[p] = {
            'pdesc': np.ones((1, 40, 768)), # tokens for unstructured data. Setting all values equal to 1 since we don't have any unstructured text in our dataset.
            'bics': np.array([row['Sex'][row_ind], row['Age'][row_ind]]), # Structured Data
            'bts': np.array([row['Frontal/Lateral'][row_ind]]), # Structured Data
            'label': np.array([row[i][row_ind] for i in labels]) # Labels
        } 
    
    return d


'''
See README.md in the data folder for additional notes on directory structure.
Running this script will output a .pkl file you can pass as input to the model.
Run multiple times for your train, validation and test sets.
'''
if __name__ == "__main__":

    d_set = 'train' # the set you want to create a .pkl file for
    d_file = 'train.csv' # point to your textual datafile. Train and test will use the same one.

    arr = image_path('./data/predev/'+d_set)
    df = data_wrangling('./data/predev/'+d_file) 

    dct = create_dct(df=df, arr=arr)
    create_pkl(dct=dct, filename='./data/predev/'+d_set+'.pkl')