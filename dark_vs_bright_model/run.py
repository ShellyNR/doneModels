import os
import glob

import cv2
import numpy as np

def calc_brightness(image, dim=10):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # higher mean means that the image is brighter
    return np.mean(L)

def getResponse(grade):
    if grade > 80:
        text = "The image is too bright."
    elif grade < 50:
        text = "The image is dark, consider taking the photo again with more light."
    else:
        text = "The image brightness is alright!"
    grade = round(100 - (abs(65 - grade)))
    return text, grade

# create output directories if not exists
# os.makedirs("output/bright", exist_ok=True)
# os.makedirs("output/dark", exist_ok=True)
def isBright():
    bright_rates = []
    # iterate through images directory
    for path in glob.glob("images/*"):
        # load image from path
        image = cv2.imread(path)
        # find if image is bright or dark
        path = os.path.basename(path)
        # higher mean means that the image is brighter
        mean = calc_brightness(image)
        response, grade = getResponse(int(mean*100))
        bright_rates.append((grade/100, response, os.path.join("images\\" + path)))
    return bright_rates