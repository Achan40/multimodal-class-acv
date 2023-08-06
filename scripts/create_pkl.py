import pickle
import numpy as np
import pandas as pd
import random

from pathlib import Path
from tqdm import tqdm

'''
This script contains functions use to create a .pkl file.
This .pkl file contains the textual data, unstructured and structured, that we can use to pass through our model.
Note that the CheXpert dataset that is being used does not contain unstructured data. Feel free to 
make changes to this code to fit your own dataset.
'''

def image_path(folder_path):
    '''
    Returns a list of paths to all images in a specified directory
    '''
    file_paths = []
    path = Path(folder_path)
    for file_path in tqdm(path.rglob('*')):
        if file_path.is_file():
            file_paths.append(str(file_path))
    return file_paths


def convert_path(input_path, offset=-4):
    '''
    Returns the relevant portion of some path. Very specific to the way our dataset is structured
    '''

    # Replace backslashes with forward slashes
    updated_path = input_path.replace('\\', '/')

    # Extract the relevant part of the path
    relevant_part = '/'.join(updated_path.split('/')[offset:])

    return relevant_part

def data_wrangling(file):
    '''
    Data wrangling. Converting some columns to binary.
    '''
    df = pd.read_csv(file)

    # generate binary indicators
    df["Sex"] = df["Sex"].replace({'Male': 1, 'Female': 0, 'Unknown': 1}) 
    df["Frontal/Lateral"] = df["Frontal/Lateral"].replace({'Frontal': 1, 'Lateral': 0}) 

    return df


def load_dict_from_pkl(file_path):
    '''
    Load the existing dictionary (or create an empty one if the file is not found)
    '''
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            # randomly shuffle the data
            random.shuffle(data)

        return data
    except FileNotFoundError:
        return {}
    

def save_to_pkl(data, file_path):
    '''
    Save the updated dictionary back to the .pkl file
    '''
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def divide_array_equal_parts(arr, num_parts):
    '''
    Dividing an array into equal parts
    Use so that we can write to our .pkl file in batches
    '''

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


def save_dict_iterative(arr, df, file_path, splits=1):
    '''
    Writing and saving data iteratively to .pkl file
    Takes in an array of paths,  the file path we want to write to, and the 
    number of write iterations
    '''

    # arr will now look something like [[path,path,path],[path,path,path],...]
    m_arr = divide_array_equal_parts(arr, splits)

    split_num = 0

    # for each iterable in m_arr
    for slice in tqdm(m_arr):

        # create empty dict to store data
        d = {}
        # load a dictionary saved as a pkl file or create an empty one
        # d = load_dict_from_pkl(file_path) 

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
        
        # For each split, create a new .pkl file
        save_to_pkl(d, file_path+"_"+str(split_num)+".pkl")
        split_num += 1


def shuffle_arr(arr):
    '''
    Randomly shuffles values in an array
    '''


'''
See README.md in the data folder for additional notes on directory structure.
Running this script will output.pkl file you can pass as input to the model.
Run multiple times to generate for your train, validation and test sets.
'''
if __name__ == "__main__":

    d_set = 'test' # the set you want to create a .pkl file for
    d_file = 'train.csv' # point to your textual datafile. Train and test will use the same one.
    d_path = './data/actual/' # folder path

    arr = image_path(d_path+d_set)
    #save_to_pkl(arr,d_path+d_set+'_arr.pkl')
    #arr = load_dict_from_pkl(d_path+d_set+'_arr.pkl')

    # perform data wrangling on the structured dataset
    df = data_wrangling(d_path+d_file)

    # split up the dataset into n parts
    save_dict_iterative(arr=arr, df=df, file_path=d_path+d_set, splits=1)