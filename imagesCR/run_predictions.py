import glob
import hashlib

import cv2
import os
import doneModels.test
from dark_vs_bright_model.run import isBright
from messy_room_classifier_master.predict import isMessy
import json
import numpy as np
from download_images import get, remove
from textModel.textModel import text_model
#import boto3
import base64
# from PIL import Image
# from io import BytesIO

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
        "i_quality_rate": -1,
        "grammar_model": -1
    }

    if (len(glob.glob("images/*")) < 4):
        dict["num_of_images"] = "Please add more images to your listing"

    bright_rates = []
    messy_rates = []

    for i, path in enumerate(glob.glob("images/*")):
        # load image from path
        image = cv2.imread(path)

        # find if image is bright or dark
        # higher mean means that the image is brighter
        bright_rate = isBright(image)
        messy_rate = isMessy(path)

        bright_rates.append((np.float64(bright_rate), "desc", path))
        messy_rates.append((np.float64(messy_rate), "desc", path))

        #print ([path],bright_rate,",",messy_rate)

    dict["i_bright_rate"] = bright_rates
    dict["i_messy_rate"] = messy_rates
    dict["grammar_model"] = text_model("add desc here")


    with open('resp.json', 'w') as f:
        json_object = json.dumps(dict, indent=4)
        f.write(json_object)

def temp_function_user_simulator(url_file):
    path="images/"
    os.makedirs(path, exist_ok=True)
    with open(url_file, "r") as f:
        for url in f.read().split("\n"):
            try:
                get(url, path=path)
            except Exception as e:
                print(e)

def create_ec2_instance():{

}

#temp_function_user_simulator("../dark_vs_bright_model/assets/dark.txt")
#temp_function_user_simulator("../dark_vs_bright_model/assets/bright.txt")
decode_images()
calc_preds()
#remove()

