import PIL.Image as Image
import glob
import shutil
import json
import base64
import io
import os

from brightness_model.brightness import brightness_model
from tidy_model.tidy import tidy_model
from fake_model.fake import fake_model
from quality_model.quality import quality_model
from roomType_model.roomType import roomType_model
from grammar_model.grammar import grammar_model
from sharpness_model.sharpness import sharpness_model
from textQuality_model.textQuality import textQuality_model
from sentiment_model.sentiment import sentiments_model


from flask import Flask, request

def runTextModels(dict, description):
    dict["grammar_model"] = grammar_model(description)
    print("model grammar done")

    dict["sentiment_model"] = sentiments_model(description)
    print("model sentiments done")

    dict["buzzwords_model"] = textQuality_model(description)
    print("model textQuality done")

    return dict

def runPhotoModels(dict):
    for i, path in enumerate(glob.glob("images/*")):
        resizeInTemp(path)

    dict["i_triq_model"] = quality_model()
    print("model quality done")

    dict["i_bright_rate"] = brightness_model()
    print("model brightness done")

    dict["i_messy_rate"] = tidy_model()
    print("model tidy done")

    dict["i_blur_rate"] = sharpness_model()
    print("model sharpness done")

    dict["i_fake_rate"] = fake_model()
    # print("model fake done")

    return dict

def resizeInTemp(path):
    im = Image.open(path)
    resizedImage = im.resize((512, 384))
    name = os.path.basename(path)
    imgPath = os.path.join("temp/" + name)
    resizedImage.save(imgPath)

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
        dict = runPhotoModels(dict)

    if len(description) != 0:
        dict = runTextModels(dict, description)

    if numOfImages != 0 and len(description)!=0:
        dict["roomType_model"] = roomType_model(description)
        print("model roomType_model done")

    with open('resp.json', 'w') as f:
        json_object = json.dumps(dict, indent=4)
        f.write(json_object)
        f.close()
    return json_object

def resetDirs(path):
    print("reset images dir")
    forImages = path + "/images"
    if os.path.exists(forImages):
        shutil.rmtree(forImages)
    os.mkdir(forImages)

    print("reset analyze dir")
    forAnalyze = path + "/analyze"
    if os.path.exists(forAnalyze):
        shutil.rmtree(forAnalyze)
    os.mkdir(forAnalyze)

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
        byte_data = str.encode(photo)
        parsedPhoto = base64.b64decode(byte_data)
        image = Image.open(io.BytesIO(parsedPhoto))
        fullpath = path + fileName
        image.save(fullpath)
    response = calc_preds(description)
    return response

if __name__ == '__main__':
    be.run(host='0.0.0.0', port=8000, debug=True)
