#Done!
import tensorflow as tf
from utils import CLASSES

# Class-based
# custom Classifier class that inherits from the 'tf.keras.Model' class
class Classifier(tf.keras.Model):
    # num_classes: The number of classes in the classification problem
    # conv_dims: A list of convolutional layer dimensions (default is [16, 32, 64])
    def __init__(self, num_classes, conv_dims=[16, 32, 64]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv_dims = conv_dims
        self.conv_layers = []   # store the convolutional blocks.
        
        # creates the convolutional blocks and appends them to the 'self.conv_layers' list
        for conv in conv_dims:
            conv_block = tf.keras.models.Sequential(
                [
                    # apply a set of learnable filters (or kernels) to the input image
                    tf.keras.layers.Conv2D(
                        filters=conv, kernel_size=(3, 3), activation="relu"
                    ),
                    # reduce the number of parameters in the model and extract the most important features
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                ]
            )
            self.conv_layers.append(conv_block)
        
        # creates a multilayer perceptron (MLP)
        self.mlp = tf.keras.models.Sequential(
            [
                # a densely-connected neural network layer,
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        
        # create a layer that normalize pixel values between 0 and 1
        self.scale = tf.keras.layers.Rescaling(1.0 / 255)
        
        # create a layer that flattens the input tensor
        self.flatten = tf.keras.layers.Flatten()

    # Forward Pass: the input data flows through the various layers of the model to produce the output
    def call(self, x) -> tf.Tensor:
        x = self.scale(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

# Function-based
# creates a Convolutional Neural Network (CNN) model for image classification tasks
def get_classifier_model(num_classes, conv_dims=[16, 32, 64]):
    model = tf.keras.models.Sequential()
    # input images should have a size of 60 pixels by 40 pixels, with a single channel (grayscale)
    model.add(tf.keras.Input(shape=(60, 40, 1)))
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    for conv in conv_dims:
        conv_block = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=conv, kernel_size=(3, 3), activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            ]
        )
        model.add(conv_block)
    model.add(tf.keras.layers.Flatten())
    mlp = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ]
    )
    model.add(mlp)
    return model
    # Ex: model = get_classifier_model(num_classes=10)
    # This will create a CNN model with 3 convolutional blocks, each with 16, 32, and 64 filters, respectively, and a final MLP with 50 units in the hidden layer and 10 units in the output layer (corresponding to 10 classes).

# create a custom InferenceClassifier
class InferenceClassifier(tf.keras.Model):
    # classifier: a pre-trained Keras model that will be used for classification
    # classes: a list of class labels
    def __init__(self, classifier: tf.keras.Model, classes=CLASSES) -> None:
        super().__init__()
        self.classes = classes
        self.classifier = classifier
        # maps the class indices to the corresponding class labels
        self.index_to_class = tf.constant(value=self.classes, dtype=tf.string)

    def call(self, x):
        # The input tensor x is expected to have the correct shape and format required by the 'self.classifier' model
        y = self.classifier(x)
        # find the index of the class with the highest logit value for each input sample
        class_index = tf.math.argmax(y, axis=1)
        # maps the predicted class indices to their corresponding class labels
        # Ex: if class_index is a tensor [0, 2, 1], and 
        # self.index_to_class is a tensor ['cat', 'dog', 'bird'], 
        # then tf.gather will return a tensor ['cat', 'bird', 'dog']
        return tf.gather(self.index_to_class, class_index)
 