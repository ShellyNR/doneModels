from keras.applications.xception import Xception
from glob import glob
import numpy
import cv2 as cv
import os
import boto3
import json

imagesPath = 'images'
IMG_SIZE = 299  # match Xception input size

ENDPOINT_NAME = 'sagemaker-tensorflow-serving-2022-06-17-11-47-25-306'

HIGH_GRADE = 90
MEDIUM_GRADE = 70
LOW_GRADE = 50

BASE_MODEL = Xception(include_top=False, weights='imagenet', pooling='avg')

def resize(image):
    h, w, c = image.shape
    cropped = image
    if h < w:
        diff = (w - h) // 2
        cropped = image[:, diff: (diff + h), :]
    elif h > w:
        diff = (h - w) // 2
        cropped = image[diff: (diff + w), :, :]

    h, w, c = cropped.shape
    if h > IMG_SIZE:    # shrink
        return cv.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
    elif h < IMG_SIZE:  # enlarge
        return cv.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_CUBIC)
    else:
        return cropped

def preprocessing():
    analyzeImagesPath = "analyze"
    filenames = glob(os.path.join(imagesPath, '*'))
    for file in filenames:
        if "\\" in file:
            filename = file.split('\\')[1]
        else:
            filename = file.split('/')[1]
        stream = open(file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
        img = cv.imdecode(numpyarray, cv.IMREAD_UNCHANGED)
        resized = resize(img)
        filepath = os.path.join(analyzeImagesPath, filename)
        cv.imwrite(filepath, resized)

def load_test_images(filespath):
    test_set = list()
    test_set_rgb = list()
    for path in filespath:
        img = cv.imread(path)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        test_set.append(img)
        test_set_rgb.append(img_rgb)
    return numpy.asarray(test_set), numpy.asarray(test_set_rgb)

def normalizeData(images):
    channel_mean = numpy.array([110.73151039, 122.90935242, 136.82249855])
    channel_std = numpy.array([69.39734207, 67.48444001, 66.66808662])
    images = images.astype('float32')
    channels = 3
    for j in range(channels):
        images[:, :, :, j] = (images[:, :, :, j] - channel_mean[j]) / channel_std[j]
    return images

def predict(images):
    runtime = boto3.Session().client(service_name='runtime.sagemaker', region_name='us-east-1')
    predictions = []
    for img in images:
        img_test = numpy.expand_dims(img, axis=0)
        features = BASE_MODEL(img_test, training=False)
        data = numpy.array(features.numpy())
        payload = json.dumps(data.tolist())
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=payload)
        result = json.loads(response['Body'].read().decode())
        res = result['predictions']
        predictions.append(res[0][0])
    return predictions

def getTextPerGrade(grade):
    if HIGH_GRADE <= grade:
        return "Your room is very neat!"
    if MEDIUM_GRADE <= grade:
        return "Your room needs a little bit of work."
    if LOW_GRADE <= grade:
        return "Your room is quite messy."
    return "Your room is messy - we recommend that you tidy up the room and upload a new picture."

def createResponse(filesname, predictions):
    tidyRates = []
    for i, imgName in enumerate(filesname):
        path = 'images\\' + imgName
        messyGrade = int(float(predictions[i]) * 100)
        grade = 100 - messyGrade
        text = getTextPerGrade(grade)
        tidyRates.append((grade/100, text, path))
    return tidyRates

def tidy_model():
    preprocessing()
    imagePrediectionPath = 'analyze'
    filesname = os.listdir(imagePrediectionPath)
    filespath = glob(os.path.join(imagePrediectionPath, '*'))
    images, images_rgb = load_test_images(filespath)
    images = normalizeData(images)
    predictions = predict(images)
    tidyRates = createResponse(filesname, predictions)
    return tidyRates
