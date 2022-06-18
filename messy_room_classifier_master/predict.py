# import base64
import json

from keras.applications.xception import Xception
from keras.models import load_model
import keras
from glob import glob
import numpy as np
import cv2 as cv
import os
import boto3

def load(image):
    img = cv.imread(image)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img, img_rgb


def load_test_images_a(file_list):
    test_set = list()
    test_set_rgb = list()
    for file in file_list:
        img = cv.imread(file)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        test_set.append(img)
        test_set_rgb.append(img_rgb)

    return np.asarray(test_set), np.asarray(test_set_rgb)

def load_test_images(image):
    img = cv.imread(image)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return np.asarray(img), np.asarray(img_rgb)

def load_from_directory_a():
    # load test images
    test_dir = r'images/*'
    filenames = glob(os.path.join(test_dir, '*.png')) + glob(os.path.join(test_dir, '*.jpg'))
    images, images_rgb = load_test_images(filenames)
    return images, images_rgb

def load_from_directory(image):
    # load test image
    images, images_rgb = load_test_images(image)
    return images, images_rgb

# def predict(image, images_rgb):
def predict():
    keras.backend.clear_session()

    # calculate from the training set
    channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
    channel_std = np.array([69.39734207, 67.48444001, 66.66808662])

    predictions = []

    for i, path in enumerate(glob("temp/*")):
        base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
        room_model = load_model('model/room_model_1552970840.h5')
        print ("in here")
        # image, images_rgb = load_test_images(path)
        image = cv.imread(path)
        images_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # normalize images
        image = image.astype('float32')

        for j in range(3):
            image[ :, :, j] = (image[ :, :, j] - channel_mean[j]) / channel_std[j]

        #np.expand_dims(images[0], axis=0)
        img_test = np.expand_dims(image, axis=0)
        features = base_model(img_test, training=False)
        prediction = room_model(features, training=False)

        predictions.append((np.float64(prediction.numpy()[0][0]), "desc", path))

    keras.backend.clear_session()

    return predictions

def predict_a():
    endpoint_name = 'sagemaker-tensorflow-serving-2022-06-17-11-47-25-306'
    runtime = boto3.Session().client(service_name='runtime.sagemaker', region_name='us-east-1')
    # runtime = boto3.client('runtime.sagemaker')

    # calculate from the training set
    channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
    channel_std = np.array([69.39734207, 67.48444001, 66.66808662])
    base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
    predictions = []
    for i, path in enumerate(glob("images/*")):

        image = cv.imread(path)
        images_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # normalize images
        image = image.astype('float32')

        for j in range(3):
            image[ :, :, j] = (image[ :, :, j] - channel_mean[j]) / channel_std[j]

        img_test = np.expand_dims(image, axis=0)
        features = base_model(img_test, training=False)

        data = np.array(features.numpy())
        payload = json.dumps(data.tolist())
        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType='application/json',
                                           Body=payload)
        result = json.loads(response['Body'].read().decode())
        res = result['predictions']
        predictions.append((res[0][0],"desc", path))

    keras.backend.clear_session()
    return predictions


# def isMessy(image):
def isMessy():
    # images, images_rgb = load_from_directory(image)
    # prediction = predict(images, images_rgb)
    # return prediction.numpy()[0][0]
    return predict_a()

