import glob
import cv2
import numpy as np
import random

threshold = 0.00175

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_grade(fm):
    grade = int((fm * 50 / threshold+3))
    if grade > 100:
        return random.randint(90,100)
    if grade < 20:
        grade = grade + 20
    return grade

def blur_detect():
    blurry_rates = []

    for i, path in enumerate(glob.glob("images/*")):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray/np.max(gray)
        fm = variance_of_laplacian(gray)

        grade = get_grade(fm)

        if fm <= threshold and grade < 50:
            text = "is blurry."
        else:
            text = "is sharp."

        blurry_rates.append((grade, text, path))

    return blurry_rates

