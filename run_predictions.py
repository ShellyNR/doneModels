import os
import glob


import cv2
from dark_vs_bright_model.run import isBright
from tidyDetection.tidy_detection import tidy_detect
import tidyDetection
from image_manipulation_detection.detect_manipulation import detect_manupulation
from messy_room_classifier_master.predict import isMessy
from triq.image_quality_prediction import triq_pred
# from roomTypeModel.roomType_detection import roomType_model
import json
import numpy as np
from download_images import get
from textModel.textModel import text_model
from blurDetection.blur_detection import blur_detect
import io
import os
import PIL.Image as Image
import base64
from check_BuzzWords.checkBuzzWords import check_text_quality
from sentiment_model.sentiment import sentiments_model
#
#
# class Echo(protocol.Protocol):
#     """This is just about the simplest possible protocol"""
#
#
#     def dataReceived(self, data):
#         "As soon as any data is received, write it back."
#         j = json.loads(data.decode('utf-8'))
#         print(j["description"])
#         f = open('resp.json')
#         self.transport.write(f.read().encode('utf-8'))
#         f.close()
#
#
# def main():
#     """This runs the protocol on port 8000"""
#     factory = protocol.ServerFactory()
#     factory.protocol = Echo
#     reactor.listenTCP(8000, factory)
# #     reactor.run()
#
#
# # this only runs if the module was *not* imported
# if __name__ == '__main__':
#     main()

from flask import Flask, request

def load_from_url(url):
    get(url)

def decode_images():
    with open('t.text') as f:
        image = f.read()
    with open("imageToSave.png", "wb") as fh:
        fh.write(base64.decodebytes(image))
    #im = bytes(image, encoding='utf-8')
    im = str(base64.decodebytes(bytes(image, encoding='utf-8')))
    path = os.path.join("images/rr" + ".jpg")
    im.save(path)
    f.close()
    return;

def calc_preds():
    dict = {
        "num_of_images": -1,
        "i_bright_rate": -1,
        "i_messy_rate": -1,
        "i_triq_model": -1,
        "i_blur_rate": -1,
        "i_fake_rate": -1,
        "grammar_model": -1,
        "sentiment_model": -1,
        "buzzwords_model": -1,
        "roomType_model": -1

    }

    tidyDetection.tidy_detection.clearAnalyzeDir("images/analyze")
    if os.path.exists("images/analyze"):
        os.rmdir("images/analyze")


    description = "Large old apartment in Tel Aviv city, a large and nice living room and a large balcony with a beautiful view no parking but have many on the road na na na more info la la la. great day"

    if (len(glob.glob("images/*")) < 4):
        dict["num_of_images"] = "Please add more images to your listing"

    for i, path in enumerate(glob.glob("images/*")):
        resizeInTemp(path)
        
    # dict["i_triq_model"] = triq_pred()
    # print("model triq done")
    #
    # dict["i_bright_rate"] = isBright()
    # print ("model bright done")

    # dict["i_messy_rate"] =isMessy()
    dict["i_messy_rate"] =tidy_detect()
    print("model messy done")
    
    if os.path.exists("images/analyze"):
        os.rmdir("images/analyze")

    # dict["i_blur_rate"] = blur_detect()
    # print("model blur done")
    
    # dict["i_fake_rate"] = detect_manupulation()
    
    # dict["grammar_model"] = text_model(description)
    # print("model grammar done")
    #
    # dict["sentiment_model"] = sentiments_model(description)
    # print("model sentiment done")
    #
    # dict["buzzwords_model"] = check_text_quality(description)
    # print("model buzzwords done")
    
    # dict["roomType_model"] = roomType_model(description)

    removeTemp()

    with open('resp.json', 'w') as f:
        json_object = json.dumps(dict, indent=4)
        f.write(json_object)
        f.close()
    return json_object

def temp_function_user_simulator(url_file):
    path= "images/"
    os.makedirs(path, exist_ok=True)
    with open(url_file, "r") as f:
        for url in f.read().split("\n"):
            try:
                get(url, path=path)
            except Exception as e:
                print(e)

def resizeInTemp(path):
    im = Image.open(path)
    # resizedImage = im.resize((1024, 768))
    resizedImage = im.resize((512, 384))
    name = os.path.basename(path)
    imgPath = os.path.join("temp/" + name)
    os.makedirs(os.path.dirname(imgPath), exist_ok=True)
    resizedImage.save(imgPath)
    return

def removeTemp():
    # if os.path.isdir("temp/"):
    files = glob.glob('temp/*')
    for f in files:
        os.remove(f)
    return

# be = Flask(__name__)

# both - text and photos endpoint
# @be.route('/', methods=['POST'])
# def hello():
#     # print(request.get_json()["description"])
#     # return "Hello World!"
#     print("in be server")
#     path = os.path.dirname(os.path.realpath(__file__)) + "/images/"
#     json = request.get_json()
#     description = json["description"]
#     photos = json["photos"]
#     photosFileNames = list(photos.keys())
#     for fileName in photosFileNames:
#         photo = photos.get(fileName)
#         print(fileName)
#         byte_data = str.encode(photo)
#         parsedPhoto = base64.b64decode(byte_data)
#         print(parsedPhoto.__sizeof__())
#         image = Image.open(io.BytesIO(parsedPhoto))
#         fullpath = path + fileName  # need to open the folder first!
#         print(fullpath)
#         image.show(fullpath)
#         image.save(fullpath)
#     return calc_preds()

if __name__ == '__main__':
    # be.run(host='0.0.0.0', port=8000,debug=True)
    calc_preds()
    # isMessy()

#temp_function_user_simulator("../dark_vs_bright_model/assets/dark.txt")
#temp_function_user_simulator("../dark_vs_bright_model/assets/bright.txt")
#decode_images()
# calc_preds()
#remove()
