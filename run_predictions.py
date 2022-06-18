import PIL.Image as Image
import glob
import shutil
import json
import base64
import io
import os

from dark_vs_bright_model.run import isBright
from tidyDetection.tidy_detection import tidy_detect
from image_manipulation_detection.detect_manipulation import detect_manupulation
from triq.image_quality_prediction import triq_pred
from roomTypeModel.roomType_detection import roomType_model
from textModel.textModel import text_model
from blurDetection.blur_detection import blur_detect
from check_BuzzWords.checkBuzzWords import check_text_quality
from sentiment_model.sentiment import sentiments_model

from flask import Flask, request

def calc_preds(description):
    dict = {
        "num_of_images": -1,
        "i_bright_rate": -1,
        "i_messy_rate": -1,
        "i_triq_model": -1,
        "i_blur_rate": -1,
        # "i_fake_rate": -1,
        "grammar_model": -1,
        "sentiment_model": -1,
        "buzzwords_model": -1,
        "roomType_model": -1

    }

    numOfImages = len(glob.glob("images/*"))
    if (numOfImages <= 2):
        dict["num_of_images"] = "Please add more images to your listing."

    if numOfImages != 0:
        for i, path in enumerate(glob.glob("images/*")):
            resizeInTemp(path)

        dict["i_triq_model"] = triq_pred()
        print("model triq done")

        dict["i_bright_rate"] = isBright()
        print("model bright done")

        dict["i_messy_rate"] = tidy_detect()
        print("model messy done")

        dict["i_blur_rate"] = blur_detect()
        print("model blur done")

        dict["roomType_model"] = roomType_model(description)
        print("model roomType_model done")

        # dict["i_fake_rate"] = detect_manupulation()

    if len(description) != 0:
        dict["grammar_model"] = text_model(description)
        print("model grammar done")

        dict["sentiment_model"] = sentiments_model(description)
        print("model sentiment done")

        dict["buzzwords_model"] = check_text_quality(description)
        print("model buzzwords done")

    with open('resp.json', 'w') as f:
        json_object = json.dumps(dict, indent=4)
        f.write(json_object)
        f.close()
    return json_object

def resizeInTemp(path):
    im = Image.open(path)
    # resizedImage = im.resize((1024, 768))
    resizedImage = im.resize((512, 384))
    name = os.path.basename(path)
    imgPath = os.path.join("temp/" + name)
    resizedImage.save(imgPath)

def resetDirs(path):
    print("reset images dir")
    forImages = path + "/images"
    if os.path.exists(forImages):
        shutil.rmtree(forImages)
    os.mkdir(forImages)

    print("reset temp dir")
    forTemp = path + "/temp"
    if os.path.exists(forTemp):
        shutil.rmtree(forTemp)
    os.mkdir(forTemp)

be = Flask(__name__)
@be.route('/', methods=['POST'])
def hello():
    print("in be server")
    path = os.path.dirname(os.path.realpath(__file__))
    resetDirs(path)
    path = path + "/images/"
    json = request.get_json()
    description = json["description"]
    photos = json["photos"]
    photosFileNames = list(photos.keys())
    for fileName in photosFileNames:
        photo = photos.get(fileName)
        # print(fileName)
        byte_data = str.encode(photo)
        parsedPhoto = base64.b64decode(byte_data)
        # print(parsedPhoto.__sizeof__())
        image = Image.open(io.BytesIO(parsedPhoto))
        fullpath = path + fileName  # need to open the folder first!
        image.save(fullpath)
    # response = calc_preds(description)
    response = roomType_model(description)
    return response

if __name__ == '__main__':
    be.run(host='0.0.0.0', port=8000, debug=True)