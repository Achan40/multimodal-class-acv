import os
import random
import shutil


def random_sample_files(source_dir, destination_dir, percentage=0.3, action='copy'):
    '''
    This function is used to create a test set.
    Utilize this to create a test set from the training set because the test set provided in the 
    CheXpert dataset does not contain The test dataset does not contain the same structured labels (ie: `Age`, `Sex`, etc..) as the train and validation set.

    ---------------
    USE WITH CAUTION.
    ---------------

    ONLY NEEDS TO BE CALLED ONCE. WILL MOVE FILES OUT FROM THE SOURCE FOLDER YOU SELECT.
    '''

    # Get a list of all subdirectories in the source directory
    subdirectories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Calculate the number of subdirectories to sample
    num_to_sample = int(len(subdirectories) * percentage)

    # Randomly select subdirectories to copy
    subdirectories_to_act = random.sample(subdirectories, num_to_sample)

    if action == 'move':
        for i in subdirectories_to_act:
            src_dir = source_dir + i
            tar_dir = destination_dir + i
            shutil.move(src_dir, tar_dir)
    else:
        for i in subdirectories_to_act:
            src_dir = source_dir + i
            tar_dir = destination_dir + i
            shutil.copytree(src_dir, tar_dir)

    print("Test set generated from " + str(percentage*100) + "% of " + SOURCE_DIRECTORY)

if __name__ == "__main__":
    # Replace these paths with your actual source and destination directories
    SOURCE_DIRECTORY = "./data/actual/train/"
    DESTINATION_DIRECTORY = "./data/actual/test/"

    # Adjust the percentage as needed (e.g., 0.3 for 30%, 0.5 for 50%)
    PERCENTAGE_TO_SAMPLE = 0.02

    # Change action to 'move' if you want to move files out from the train set. USE WITH CAUTION.
    random_sample_files(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, PERCENTAGE_TO_SAMPLE, action='move')