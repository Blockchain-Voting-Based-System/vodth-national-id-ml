import tensorflow as tf
import argparse

def convert_tflite(args, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(args.model_checkpoint + "model.tflite", 'wb') as f:
        f.write(tflite_model)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="models/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = tf.keras.models.load_model(args.model_checkpoint + "model.h5")
    convert_tflite(args, model)