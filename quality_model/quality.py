import glob

import keras

from quality_model.models.triq_model import create_triq_model
import numpy as np
from PIL import Image

from scipy.stats import norm

HIGH_GRADE = 85
MEDIUM_GRADE = 0.75
LOW_GRADE = 0.6

MAX_QUALITY_LEVEL = 5
MIN_QUALITY_LEVEL = 1

RGB_MAX = 255

MEAN = 3
DEVIATION = 0.6666666666666666666666666666666

MODEL_WEIGHTS_PATH=r'quality_model/pretrained_weights/TRIQ.h5'

def get_response(grade):
    if grade > HIGH_GRADE:
        response = "The image quality is great!"
    elif grade > MEDIUM_GRADE:
        response = "The image quality is good."
    elif grade > LOW_GRADE:
        response = "The image quality is not good."
    else:
        response = "The image quality is not good, you should consider taking another photo."
    return response

def predict_image_quality():
    model = create_triq_model(MAX_QUALITY_LEVEL)
    model.load_weights(MODEL_WEIGHTS_PATH)

    triq_rates = []

    for i, path in enumerate(glob.glob("temp/*")):
        image = Image.open(path)
        image = np.asarray(image, dtype=np.float32)
        image /= (RGB_MAX/2.0)
        image -= 1.

        prediction = model.predict(np.expand_dims(image, axis=0))

        mos_scales = np.arange(MIN_QUALITY_LEVEL, MAX_QUALITY_LEVEL + 1)
        predicted_mos = (np.sum(np.multiply(mos_scales, prediction[0])))

        pdf = norm.cdf(predicted_mos, loc=MEAN, scale=DEVIATION)
        grade = np.float64(pdf)
        response = get_response(grade)
        path = path.replace("temp/", "images\\")
        triq_rates.append((grade, response, path))
    keras.backend.clear_session()
    return triq_rates

def quality_model():
    predict_mos = predict_image_quality()
    return predict_mos

    
