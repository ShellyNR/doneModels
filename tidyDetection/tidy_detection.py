from keras.applications.xception import Xception
from glob import glob
import numpy
import cv2 as cv
import os
import boto3
import json

imagesPath = './images'

def clearAnalyzeDir(analyzeImagesPath):
    files = glob(os.path.join(analyzeImagesPath, '*'))
    for f in files:
        os.remove(f)

def resize(image):
    img_size = 299  # match Xception input size
    h, w, c = image.shape
    cropped = image
    if h < w:
        diff = (w - h) // 2
        cropped = image[:, diff: (diff + h), :]
    elif h > w:
        diff = (h - w) // 2
        cropped = image[diff: (diff + w), :, :]

    h, w, c = cropped.shape
    if h > img_size:    # shrink
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_AREA)
    elif h < img_size:  # enlarge
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_CUBIC)
    else:
        return cropped

def preprocessing():
    analyzeImagesPath = os.path.join(imagesPath, 'analyze')
    if not os.path.exists(analyzeImagesPath):
        os.mkdir(analyzeImagesPath)
    clearAnalyzeDir(analyzeImagesPath)
    filenames = glob(os.path.join(imagesPath, '*'))
    if analyzeImagesPath in filenames:
        filenames.remove(analyzeImagesPath)
    for file in filenames:
        if "\\" in file:
            filename = file.split('\\')[1]
        else:
            filename = file.split('/')[2]
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
    for j in range(3):
        images[:, :, :, j] = (images[:, :, :, j] - channel_mean[j]) / channel_std[j]
    return images

def predict(images):
    endpoint_name = 'sagemaker-tensorflow-serving-2022-06-17-11-47-25-306'
    runtime = boto3.Session().client(service_name='runtime.sagemaker', region_name='us-east-1')
    base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
    predictions = []
    for img in images:
        img_test = numpy.expand_dims(img, axis=0)
        features = base_model(img_test, training=False)
        data = numpy.array(features.numpy())
        payload = json.dumps(data.tolist())
        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=payload)
        result = json.loads(response['Body'].read().decode())
        res = result['predictions']
        predictions.append(res[0][0])
    return predictions

def getTextPerGrade(grade):
    if 90 <= grade:
        return "Your room is very neat!"
    if 65 <= grade:
        return "Your room is quite messy."
    return "Your room is messy - we recommend that you tidy up the room and upload new pictures."

def createResponse(filesname, predictions):
    tidyRates = []
    for i, imgName in enumerate(filesname):
        path = 'images\\' + imgName
        messyGrade = int(float(predictions[i]) * 100)
        grade = 100 - messyGrade
        text = getTextPerGrade(grade)
        tidyRates.append((grade/100, text, path))
    return tidyRates

def tidy_detect():
    preprocessing()
    imagePrediectionPath = os.path.join(imagesPath, 'analyze')
    filesname = os.listdir(imagePrediectionPath)
    filespath = glob(os.path.join(imagePrediectionPath, '*'))
    images, images_rgb = load_test_images(filespath)
    images = normalizeData(images)
    predictions = predict(images)
    tidyRates = createResponse(filesname, predictions)
    return tidyRates
