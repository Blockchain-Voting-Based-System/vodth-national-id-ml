import sys
sys.path.append("src/models/")
sys.path.append("src/app/")

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import tensorflow as tf
from typing import Optional
import numpy as np
from schemas import PersonalInformation, Settings
from predict import model_fn, predict_fn

app = FastAPI(
    title="National ID Card OCR",
    description="Extract name, date of birth, ID and other informations from an ID card",
)

settings = Settings()

@app.get("/")
def read_root():
    response = {
        "data": "Everything is working as exepected",
    }
    return response

@app.post("/extract_image/")
def predict(image: UploadFile = File(...)):
    inference_model = ExtractInformationFromImage()
    inference_model.load_model(settings)
    return inference_model.predict(image)

class ExtractInformationFromImage:
    model: Optional[tf.keras.Model]

    def load_model(self, settings):
        model, loaded = model_fn(settings.model_checkpoint)
        self.loaded = loaded
        self.model = model

    def predict(self, image: UploadFile) -> JSONResponse:
        if not self.model:
            raise RuntimeError
        data = np.fromfile(image.file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        informations = predict_fn(model=self.model, input_image=image)
        
        personal_info = PersonalInformation(**informations)
        
        return JSONResponse(content={"Information": personal_info.dict()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)