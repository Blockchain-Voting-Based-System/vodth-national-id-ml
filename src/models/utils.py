#Done!
import tensorflow as tf
import math
import os
import numpy as np
from typing import List, Dict

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
    "_seperator",
]

# loads and preprocesses image data for training and validating a machine learning model
# -> List[tf.data.Dataset] = return a list of 'tf.data.Dataset' objects
def get_loaders(train_path, validation_path, batch_size) -> List[tf.data.Dataset]:
    # create a training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",      # Infers the labels from the directory structure
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=True,
        # setting a fixed seed value only affects the shuffling of the training data
        seed=123,       # 123 = same random sequence
        image_size=(60, 40),
        batch_size=batch_size,
    )
    
    # create a validation dataset
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=False,
        seed=123,
        image_size=(60, 40),
        batch_size=batch_size,
    )

    return train_ds, validation_ds

# loads and preprocesses image data for testing a machine learning model
def get_test_loader(test_path, batch_size) -> tf.data.Dataset:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=False,
        seed=123,
        image_size=(60, 40),
        batch_size=batch_size,
    )
    return test_ds

# calculates class weights for a machine learning model
# Class Weights - address class imbalance in the dataset
# mu: The target fraction of the majority class (default is 0.15)
def get_class_weight(source_path, mu=0.15) -> Dict:
    all_examples = []
    for this_class in CLASSES:
        # count the number of files in each class directory
        all_examples += [len(os.listdir(source_path + this_class))]
    # create a dictionary, where the keys = the class indices, and the values = the number of examples for each class
    labels_dict = dict(enumerate(all_examples))

    # calculates the total number of examples across all classes
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    # iterates through the class indices (keys)
    for key in keys:
        # mu * total = target number of examples
        # float(labels_dict[key]) = actual number of examples for the current class
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

#  perform a learning rate search, where you want to explore a range of learning rates to find the optimal value for your machine learning model
def learning_rate_search_space(lower=1e-4, upper=1e-1, size=5):
    a = np.log10(lower)
    b = np.log10(upper)
    r = (b - a) * np.random.rand(size) + a      # generates size number of random values between a and b
    lr = np.power(10, r)    # transform back to the original scale
    lr = np.sort(lr)    # generate learning rates are sorted in ascending order
    
    return lr

# calculates the confusion matrix for a classification
class ConfusionMatrix(tf.keras.metrics.Metric):
    # initializes the attributes
    def __init__(self, name="cm", num_classes=2, normalized=False, **kwargs):
        """initializes attributes of the class"""
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.normalized = normalized
        # Variables - store and update values during the execution of a model
        self.c_matrix = tf.Variable(
            # creates the initial value for 'c_matrix' variable
            # 'tf.zeros_initializer()' function creates a tensor filled with zeros
            initial_value=tf.zeros_initializer()(
                # 2D tensor with a shape of (num_classes, num_classes)
                shape=(num_classes, num_classes), dtype=tf.int32
            )
        )

    # update the state of the confusion matrix
    def update_state(self, y_true, y_pred, sample_weight=None):
        # calculates the indices
        y_pred_index = tf.math.argmax(y_pred, axis=1)
        y_true_index = tf.math.argmax(y_true, axis=1)
        # calculates the confusion matrix
        conf_matrix = tf.math.confusion_matrix(
            y_true_index, y_pred_index, num_classes=self.num_classes
        )
        # add to 'c_matrix' variable
        self.c_matrix.assign_add(conf_matrix)

    # retrieve the current value of the confusion matrix
    def result(self):
        """Computes and returns the metric value tensor."""
        if self.normalized:
            # divide each element by the sum of the corresponding column
            c = self.c_matrix / tf.reduce_sum(self.c_matrix, axis=0)
            return c
        else:
            return self.c_matrix

    # reset the state of the confusion matrix
    def reset_state(self):
        """Resets all of the metric state variables."""
        # The state of the metric will be reset at the start of each epoch by setting c_matrix variable to a tensor of zeros with the same shape as the initial
        self.c_matrix.assign(0)
