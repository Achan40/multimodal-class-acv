import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

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
    for file_path in tqdm(path.rglob('*')):
        if file_path.is_file():
            file_paths.append(str(file_path))
    return file_paths

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

# Load the existing dictionary (or create an empty one if the file is not found)
def load_dict_from_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        return {}
    
# Save the updated dictionary back to the .pkl file
def save_dict_to_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Dividing an array into equal parts
# Use so that we can write to our .pkl file in batches
def divide_array_equal_parts(arr, num_parts):
    if num_parts <= 0:
        raise ValueError("Number of parts should be greater than 0.")

    part_size = len(arr) // num_parts
    remaining_elements = len(arr) % num_parts

    divided_array = []
    start = 0

    for i in range(num_parts):
        end = start + part_size + (1 if i < remaining_elements else 0)
        divided_array.append(arr[start:end])
        start = end

    return divided_array

# Writing and saving data iteratively to .pkl file
def save_dict_iterative(arr, file_path, splits=1):

    # arr will now look something like [[path,path,path],[path,path,path],...]
    m_arr = divide_array_equal_parts(arr, splits)

    # for each iterable in m_arr
    for slice in tqdm(m_arr):
        # load a dictionary saved as a pkl file or create an empty one
        d = load_dict_from_pkl(file_path) 

        # for each item in the iterable
        for path in tqdm(slice):
            p = convert_path(path)

            '''
            items in the datafiles indexed as follows:
            valid.csv: CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg
            train.csv: CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
            '''
            key = 'CheXpert-v1.0/' + p

            # test.pkl will need to get index from train.csv, have to retrieve by key first.
            if 'test' in p:
                new_p = convert_path(path, offset=-3)
                key = 'CheXpert-v1.0/train/' + new_p

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

        save_dict_to_pkl(d, file_path)
'''
See README.md in the data folder for additional notes on directory structure.
Running this script will output a .pkl file you can pass as input to the model.
Run multiple times to generate for your train, validation and test sets.
'''
if __name__ == "__main__":

    d_set = 'train' # the set you want to create a .pkl file for
    d_file = 'train.csv' # point to your textual datafile. Train and test will use the same one.
    d_path = './data/actual/' # folder path

    arr = image_path(d_path+d_set)
    df = data_wrangling(d_path+d_file) 

    save_dict_iterative(arr=arr, file_path=d_path+d_set+'.pkl', splits=100)
    
    #dct = create_dct(df=df, arr=arr)
    #create_pkl(dct=dct, filename=d_path+d_set+'.pkl')