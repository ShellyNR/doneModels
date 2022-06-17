from keras.applications.xception import Xception
from keras.models import load_model
from glob import glob
import numpy as np
import cv2 as cv
import os

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
    analyzePath = imagesPath + '/analyze'
    filenames = glob(os.path.join(imagesPath, '*'))
    print(filenames)
    if analyzePath in filenames:
        filenames.remove(analyzePath)
    print(filenames)
    for file in filenames:
        print(file)
        filename = file.split('/')[2]
        print(filename)
        stream = open(file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
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
    return np.asarray(test_set), np.asarray(test_set_rgb)

def normalizeData(images):
    channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
    channel_std = np.array([69.39734207, 67.48444001, 66.66808662])
    images = images.astype('float32')
    for j in range(3):
        images[:, :, :, j] = (images[:, :, :, j] - channel_mean[j]) / channel_std[j]
    return images

def predict(images):
    base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
    path = os.path.dirname(os.path.abspath(__file__))
    room_model = load_model(os.path.join(path, 'tidyModel.h5'))
    features = base_model.predict(images)
    predictions = room_model.predict(features)
    return predictions

def getTextPerGrade(grade):
    if grade <= 65:
        return "your room is messy - we recommend that you tidy up the room and upload new pictures."
    return "your room is nety."

def createResponse(filesname, predictions):
    tidyRates = []
    for i, imgName in enumerate(filesname):
        path = 'images\\' + imgName
        messyGrade = int(float(predictions[i][0]) * 100)
        grade = 100 - messyGrade
        text = getTextPerGrade(grade)
        tidyRates.append((grade, text, path))
    return tidyRates

def tidy_detect():
    preprocessing()
    imagePrediectionPath = os.path.join(imagesPath, 'analyze')
    filesname = os.listdir(imagePrediectionPath)
    filespath = glob(os.path.join(imagePrediectionPath, '*'))
    images, images_rgb = load_test_images(filespath)
    images = normalizeData(images)
    predictions = predict(images)
    clearAnalyzeDir(imagePrediectionPath)
    os.rmdir(imagePrediectionPath)
    tidyRates = createResponse(filesname, predictions)
    print("tidyyyyy")
    print(tidyRates)
    print("tidyyyyy")
    return tidyRates