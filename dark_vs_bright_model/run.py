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

# create output directories if not exists
# os.makedirs("output/bright", exist_ok=True)
# os.makedirs("output/dark", exist_ok=True)
def isBright():
    bright_rates = []
    # iterate through images directory
    for i, path in enumerate(glob.glob("images/*")):
        # load image from path
        image = cv2.imread(path)
        thresh = 0.45
        # find if image is bright or dark
        path = os.path.basename(path)
        # higher mean means that the image is brighter
        mean = calc_brightness(image)
        bright_rates.append((np.float64(mean), "desc", path))
        # text ="bright" if mean > thresh else "dark"
    return bright_rates

        # save image to disk
        # cv2.imwrite("output/{}/{}".format(text, path), image)
        # print(path, "=>", text, ". Scale -", mean)
