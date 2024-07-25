#Done!
import os
import random
from shutil import copyfile
import shutil
import argparse

# Digits: "0/" to "9/" (10 classes)
# Uppercase letters: "A/" to "Z/" (26 classes)
# Special character: "</" (1 class)
CLASSES = [
    "0/",
    "1/",
    "2/",
    "3/",
    "4/",
    "5/",
    "6/",
    "7/",
    "8/",
    "9/",
    "A/",
    "B/",
    "C/",
    "D/",
    "E/",
    "F/",
    "G/",
    "H/",
    "I/",
    "J/",
    "K/",
    "L/",
    "M/",
    "N/",
    "O/",
    "P/",
    "Q/",
    "R/",
    "S/",
    "T/",
    "U/",
    "V/",
    "W/",
    "X/",
    "Y/",
    "Z/",
    "_seperator/",
]

# 
def delete_images_in_dir(split_path):
    # list of all the directories (classes) within the split_path directory
    for this_class in os.listdir(split_path):
        # list of all the files and directories within that class directory
        for filename in os.listdir(split_path + this_class):
            # full file path
            file_path = os.path.join(split_path + this_class, filename)
            try:
                # check for file or link to delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # check for directory to delete it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

def split_data(
    source_path,
    training_path,
    validation_path,
    testing_path,
    split_size,
    max_class_size=None,    # maximum number of files per class
):
    for this_class in CLASSES:
        files = []
        # a list of all filenames in the subdirectory of 'source_path'
        for filename in os.listdir(source_path + this_class):
            file = source_path + this_class + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                # filters out any zero-length files
                print(filename + " is zero length, so ignoring.")

        # shuffle the remaining files
        shuffled_set = random.sample(files, len(files))
        # trims the list to the maximum size if numbers of files exceeds it
        if max_class_size != None:
            if len(shuffled_set) > max_class_size:
                shuffled_set = shuffled_set[0:max_class_size]
        # calculates the number of files to be allocated to the training, validation, and testing sets based on 'split_size'
        # testing_length = int(len(shuffled_set) * split_size)
        # validation_length = int(len(shuffled_set) * split_size)
        # make sure that validation and testing set have at least and at most 1 data for 'split_size' = 0.1
        testing_length = max(1, int(len(shuffled_set) * split_size))
        validation_length = max(1, int(len(shuffled_set) * split_size))
        training_length = int(len(shuffled_set) - testing_length - validation_length)

        training_set = shuffled_set[0:training_length]
        # slicing the 'shuffled_set' list after the training set and before the testing set
        validation_set = shuffled_set[
            training_length : training_length + validation_length
        ]
        # slicing the 'shuffled_set' list from the end, using the 'testing_length' to determine how many files to include
        testing_set = shuffled_set[-testing_length:]

        # create directories
        os.makedirs(training_path, exist_ok=True)
        for c in CLASSES:
            os.makedirs(training_path + c, exist_ok=True)

        os.makedirs(validation_path, exist_ok=True)
        for c in CLASSES:
            os.makedirs(validation_path + c, exist_ok=True)

        os.makedirs(testing_path, exist_ok=True)
        for c in CLASSES:
            os.makedirs(testing_path + c, exist_ok=True)

        # copy files to directories
        for filename in training_set:
            this_file = source_path + this_class + filename
            destination = training_path + this_class + filename
            copyfile(this_file, destination)

        for filename in validation_set:
            this_file = source_path + this_class + filename
            destination = validation_path + this_class + filename
            copyfile(this_file, destination)

        for filename in testing_set:
            this_file = source_path + this_class + filename
            destination = testing_path + this_class + filename
            copyfile(this_file, destination)

# This parser will handle the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="data/labeled/")
    parser.add_argument("--training_path", type=str, default="data/train/")
    parser.add_argument("--validation_path", type=str, default="data/validation/")
    parser.add_argument("--testing_path", type=str, default="data/test/")
    parser.add_argument("--split_size", type=float, default=0.1)
    parser.add_argument("--max_class_size", type=int, default=1000)
    return parser.parse_args()



# script is run directly
if __name__ == "__main__":
    args = parse_args()
    # creates a dictionary 'all_examples' that stores the number of files for each class in the 'source_path' directory
    all_examples = {}
    all_class_dir = os.listdir(args.source_path)
    for all_class in all_class_dir:
        all_examples[all_class] = len(os.listdir(args.source_path + all_class))
    # calculates the total number of files
    total = 0
    for value in all_examples.values():
        total += value
    print(
        "%%%% Data SET Directory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    print(all_examples)
    print(total)
    # check the directories to delete all files in them
    if os.path.exists(args.training_path):
        delete_images_in_dir(args.training_path)
    if os.path.exists(args.validation_path):
        delete_images_in_dir(args.validation_path)
    if os.path.exists(args.testing_path):
        delete_images_in_dir(args.testing_path)

    split_data(
        source_path=args.source_path,
        training_path=args.training_path,
        validation_path=args.validation_path,
        testing_path=args.testing_path,
        split_size=args.split_size,
        max_class_size=args.max_class_size,
    )
