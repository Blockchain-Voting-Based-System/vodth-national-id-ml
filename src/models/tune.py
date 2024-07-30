import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import argparse
from models import get_classifier_model
from utils import get_loaders, get_class_weight, CLASSES, learning_rate_search_space
import numpy as np

def train_test_model(args, hparams) -> float:
    # Model Compilation: perform after writing the statements in a model and before training starts
    model = get_classifier_model(num_classes=len(CLASSES)) 
    model.compile(
        # Loss function: Categorical Cross-Entropy
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # Optimizer: Adam optimizer
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE]),
        # Metric: F1-score (macro-averaged)
        metrics=[
            tfa.metrics.F1Score(
                num_classes=len(CLASSES), average="macro", name="f1_score"
            )
        ],
    )

    # Data Loaders: gets the training and validation data loaders
    train_ds, validation_ds = get_loaders(
        train_path=args.train_path,
        validation_path=args.validation_path,
        batch_size=hparams[HP_BATCH_SIZE],
    )
    # The training and validation datasets are cached and prefetched to improve performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,     # stop the training if the validation loss does not improve by at least 1e-2 for 5 epochs
        patience=5,
        verbose=1,
    )
    callbacks = [early_stopping]
    # determines whether to use class weights during the model training process or not
    if hparams[HP_USE_CLASS_WEIGHT]:
        class_weight = get_class_weight(source_path=args.train_path)
    else:
        class_weight = None

    # Model Training
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        class_weight=class_weight,
        epochs=args.epochs,     # ?
        verbose=1,      # verbosity level: determines how much information is displayed during the training process
        callbacks=[callbacks],
    )
    
    loss, f1_score = model.evaluate(validation_ds)
    
    accuracy = model.evaluate(validation_ds, verbose=0)[1]
    metrics = {
        METRIC_F1Score: f1_score,
        METRIC_ACCURACY: accuracy,
    }

    return metrics

# train and test a machine learning model, record the hyperparameters used, and log the F1 score to the TensorFlow summary
def run(run_dir, hparams, args):
    # creates a TensorFlow summary file writer and sets it as default summary writer
    with tf.summary.create_file_writer(run_dir).as_default():
        # record the hyperparametr values used in this trial
        hp.hparams(hparams)
        # write F1 score to the TensorFlow summary
        metrics = train_test_model(args, hparams)

        for metric, value in metrics.items():
            tf.summary.scalar(metric, value, step=1) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train/")
    parser.add_argument("--validation_path", type=str, default="data/validation/")
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()




# sets up the hyperparameter tuning process
if __name__ == "__main__":
    args = parse_args()
    # defines the search space for the learning rate
    learning_rate_values = learning_rate_search_space(
        lower=4e-4, upper=4e-3, size=10
    ).tolist()
    # creates three hyperparameter objects
    HP_LEARNING_RATE = hp.HParam("learning_rate", hp.Discrete(learning_rate_values))
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([128]))
    HP_USE_CLASS_WEIGHT = hp.HParam("class_weight", hp.Discrete([True, False]))

    METRIC_F1Score = "f1_score"
    METRIC_ACCURACY = "accuracy"
    
    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
        # configures the hyperparameters and metrics
        hp.hparams_config(
            hparams=[HP_LEARNING_RATE, HP_BATCH_SIZE, HP_USE_CLASS_WEIGHT],
            metrics=[
                hp.Metric(METRIC_F1Score, display_name="F1_Score"),
                hp.Metric(METRIC_ACCURACY, display_name="Accuracy"),
            ],
        )

    # iterate over the set of valid values for each hyperparameter in the nested loops
    session_num = 0
    # loop over a single value of [128]
    for batch_size in HP_BATCH_SIZE.domain.values:
        # loop over the 10 values
        for learning_rate in HP_LEARNING_RATE.domain.values:
            # loop over 2 values of [True, False]
            for use_class_weight in HP_USE_CLASS_WEIGHT.domain.values:
                # create an 'hparams' dictionary
                hparams = {
                    HP_BATCH_SIZE: batch_size,
                    HP_LEARNING_RATE: learning_rate,
                    HP_USE_CLASS_WEIGHT: use_class_weight,
                }
                run_name = "run-%d" % session_num
                print("--- Starting trial: %s" % run_name)
                print({h.name: hparams[h] for h in hparams})
                # perform the training and evaluation process
                run("logs/hparam_tuning/" + run_name, hparams, args)
                session_num += 1
