#Done!
import cv2
import imutils
import numpy as np
from utils_old import get_characters_from_image
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List
from pathlib import Path

# Read image and Resize 
def _process(image_name: str, input_path: Path, output_path: Path) -> None:
    image_path = cv2.imread(input_path + image_name)
    try:
        total = get_characters_from_image(image_path)
        for i in range(3):
            for j in range(30):
                c = total[i][j]
                dim = (40, 60)
                c = cv2.resize(c, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    output_path + image_name + "_" + str(i) + "_" + str(j) + ".png",
                    c,
                )
                # output_path/image_name_i_j.png
                # ./images/my_image_2_5.png
    except BaseException as err:
        print(f"Fail: Unexpected {err=}, {type(err)=}")
        print("Can't extract characters from image {}".format(image_name))

# This parser will handle the command-line arguments
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i,", type=str, default="data/raw/")
    parser.add_argument("--output_path", "-o", type=str, default="data/processed/")
    return parser.parse_args()
    # python script.py -i "path/to/input" -o "path/to/output"
    # or python script.py --input_path "path/to/input" --output_path "path/to/output"




# ensures that the code inside it is only executed when the script is run directly (not imported as a module)
if __name__ == "__main__":
    args = arg_parser()
    # a partially applied function: create a new function by fixing one or more arguments of an existing function, leaving the remaining arguments to be provided later
    process = partial(_process, input_path=args.input_path, output_path=args.output_path)
    # gets a list of all the files in the 'input_path' directory
    images = os.listdir(args.input_path)
    # filtering the list
    images = [
        image
        for image in images
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")
    ]

    # a multiprocessing Pool object: allows tasks to be submitted as functions to be executed concurrently across multiple worker processes
    with Pool(processes=cpu_count()) as pool:
        pool.map(process, images)
