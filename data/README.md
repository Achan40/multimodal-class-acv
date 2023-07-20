# Data

The training dataset used is the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) dataset. The validation and test datasets can be found [here](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c) contained within the CheXpert directory.

Note: The easiest way to download the validation and test dataset is by using [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10). Additional documentation [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs-download).

2023-07-19: The test dataset does not contain the same structured labels (ie: `Age`, `Sex`, etc..) as the train and validation set. May need to do some random sampling from the train set to retrieve the test set...

### 1. Directory and File Structure
After downloading the CheXpert dataset, place the train/valid/test directories and their corresponding .csv files into the data directory. Rename .csv files accordingly. 

-data <br>
--train <br>
--valid <br>
--train.csv <br>
--valid.csv <br>

### 2. Generate Test Set
While located in this repositories root directory in your python shell, edit and run the `split_train.py` script.

Remember to set the `percentage_to_sample` and `action=move` if you're ready to generate your test set. WARNING: YOU ONLY NEED TO DO THIS ONCE. This `action=move` parameter in the `random_sample_files` function will move data out of the train directory and into the test directory. 

### 3. Generate .pkl Files
While located in this repositories root directory in your python shell, edit and run the `create_pkl.py` script to create the required `.pkl` files for each of your sets. 

The model takes in a dictionary as input. 