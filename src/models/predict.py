import sys

sys.path.append("src/data/")
from utils_old import get_characters_from_image
import cv2
import tensorflow as tf
import numpy as np
import argparse

CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<",
]

# takes a list of characters as input and returns a list of the characters that are used in the input
def get_used_characters(characters):
    # Identify Number in range (5,13)
    line_01 = [characters[0][i] for i in range(5, 14)]
    # Birthdate, Sex, Expiration date in range (0,13)
    line_02 = [characters[1][i] for i in range(0, 15)]
    # Lastname and Firstname from line 3 directly
    line_03 = characters[2]
    images = line_01 + line_02 + line_03
    return images


"""
def get_class_from_prediction(prediction, CLASSES):
    classes = np.array(CLASSES)
    ind = np.argmax(prediction, axis=1)
    return classes[ind]
"""

# get predicted class labels based on the input predictions
def get_class_from_prediction(predictions, CLASSES):
    # creates a new numpy array mask with the same shape as the predictions array, but with all elements set to 1
    mask = np.ones_like(predictions)
    # creates a numpy array classes from the CLASSES list
    classes = np.array(CLASSES)
    # finds the indices of the classes that are not "M" or "F" and stores them
    indices_m_and_n = np.where(((classes != "M") & (classes != "F")))[0]
    # sets the values in a specific range of the mask array to 0
    # Highlighting Specific Patterns
    mask[0:15, 10:] = 0
    mask[16, indices_m_and_n] = 0
    mask[17:24, 10:] = 0
    mask[24:, 0:10] = 0
    classes = np.array(CLASSES)
    # creates a new array 'constrainted_prediction' by multiplying the 'predictions' array element-wise with the 'mask' array
    constrainted_prediction = np.multiply(predictions, mask)
    # find the indices of the maximum values in each row of the 'constrainted_prediction' array
    ind = np.argmax(constrainted_prediction, axis=1)
    return classes[ind]

# Information extraction from an ID card
def get_personal_information_from_predictions(outputs):
    # divide output into 3 lines
    line0 = outputs[0:9]
    line1 = outputs[9:24]
    line2 = outputs[24:]
    # get id number from line 0
    id_number = "".join(line0)
    # get birthdate (0,5) from line 1 and format it into YEAR-MONTH-DATE
    birthdate = "".join(line1[0:6])
    birthdate = birthdate[0:2] + "-" + birthdate[2:4] + "-" + birthdate[4:]
    # get sex (7) from line 1
    sex = line1[7]
    # get experation date (8,13) from line 1 and format it into YEAR-MONTH-DATE
    exp_date = "".join(line1[8:14])
    exp_date = exp_date[0:2] + "-" + exp_date[2:4] + "-" + exp_date[4:]
    
    # define some variables for last name and first name
    last_name = ""
    first_name = ""
    i = 0
    # set flags for last name
    prec = False        # previous
    actual = False
    sep = False         # seperator = 'double space' or '<'
    # loop through index of line 2 to create last name until a seperator is met
    while (i < len(line2)) and not (sep):
        c = line2[i]
        prec = actual
        if c != "<":
            last_name += c
            actual = False
        else:
            last_name += " "
            actual = True
        sep = actual and prec
        i += 1
    # reset the flags for first name
    prec = False
    actual = False
    sep = False
    # continue loop through index of line 2 to create first name until another seperator is met
    while (i < len(line2)) and not (sep):
        c = line2[i]
        prec = actual
        if c != "<":
            first_name += c
            actual = False
        else:
            first_name += " "
            actual = True
        sep = actual and prec
        i += 1
    # create a ditionary to store extracted information
    informations = {
        "first_name": first_name,
        "last_name": last_name,
        "birthdate": birthdate,
        "sex": sex,
        "identity_num": id_number,
        "expiration_date": exp_date,
    }
    return informations

# process, extract, and return information
def predict_fn(model, input_image):
    # get 3 lines from image processing
    all_characters = get_characters_from_image(input_image)
    # get only the needed characters from each line
    characters_to_predict = get_used_characters(all_characters)
    # convert image to a 32-bit floating-point data type
    model_input = [tf.cast(image, dtype=tf.float32) for image in characters_to_predict]
    # add new dimension at the last axis position (-1)
    model_input = [tf.expand_dims(image, axis=-1) for image in model_input]
    # stack a list of input models along the first axis (0)
    model_input = tf.stack(model_input, axis=0)
    # pass model inputs into model and get the output layer
    predictions = model(model_input)["sequential_4"].numpy()
    # converts predictions to class labels
    outputs = get_class_from_prediction(predictions, CLASSES)
    # extract information from predicted classes
    informations = get_personal_information_from_predictions(outputs)
    return informations

# load a model and return it for making predictions
def model_fn(model_checkpoint):
    # self.model = tf.keras.models.load_model("models/model.h5")
    # load a pre-trained model from a checkpoint
    loaded = tf.saved_model.load(model_checkpoint)
    # get a signature from the loaded model
    # "serving_default" signature: for serving the model in production environments
    model = loaded.signatures["serving_default"]
    return model, loaded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="models/model")
    parser.add_argument("--input_image", type=str, default="data/raw/kh_idCard_04.png")
    # parser.add_argument("--input_image", type=str, default="data/raw/kh_idCard_03.jpg")
    # parser.add_argument("--input_image", type=str, default="data/raw/kh_idCard_02.jpg")
    # parser.add_argument("--input_image", type=str, default="data/raw/kh_idCard_01.jpg")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    image = cv2.imread(args.input_image)
    model, loaded = model_fn(model_checkpoint=args.model_checkpoint)
    response = predict_fn(model=model, input_image=image)
    print()
    print()
    # loop over the key-value pairs in the 'response' dictionary
    for k, v in response.items():
        print(k, v)
