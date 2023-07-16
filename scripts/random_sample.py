import os
import random
import shutil

'''
Script generated using chatGPT. Since CheXpert dataset is so large, we should start by selecting a subset for testing purposes.
'''

def random_sample_directories(source_dir, destination_dir, num_samples):
    # Get a list of all directories in the source directory
    directories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Randomly sample directories
    sampled_directories = random.sample(directories, num_samples)
    
    # Copy the sampled directories to the destination directory
    for directory in sampled_directories:
        source_path = os.path.join(source_dir, directory)
        destination_path = os.path.join(destination_dir, directory)
        shutil.copytree(source_path, destination_path)
        print(f"Copied '{directory}' to '{destination_dir}'")

# Example usage
source_directory = "data/CheXpert-v1.0/"
destination_directory = "data/dev/"
num_samples = 2

random_sample_directories(source_directory, destination_directory, num_samples)