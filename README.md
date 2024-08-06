### Setup

- Clone this repository

- Create new Conda Environment

        conda create -n vodth_national_id_ml_env python=3.10.8

- Activate the env

        conda activate vodth_national_id_ml_env

- Install the requirement packages

        pip install -r requirements.txt

### How to train the model

- Preprocessing steps to build the unlabeled data

        python src/data/process.py -i

- Label your characters images into _37_ classes

  - Numbers (10 classes): 0 - 9
  - Letters (26 classes): A - Z
  - Separator (1 class): <

- Split your data into training, validation and test

        python src/features/split.py

- Hyperparameters Optimization (HPO) using TensorBoard:

        python src/models/tune.py

- Train the model with the best hyperparameters set:

        python src/models/train.py

- Evaluate the trained model

        python src/models/evaluate.py

- Save the trained model in the SaveModel format of TensorFlow

        python src/models/save.py

- Use the model to inference on a single image using CLI

        python src/models/predict.py --input_image "image_path"

- Covert the model file from h5 to tflite

        python src/models/convert.py

### How to run this project with FastAPI

- Run the commend below

        uvicorn src.app.api:app --host 0.0.0.0 --port 8000
