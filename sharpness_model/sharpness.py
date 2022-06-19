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

def parsePath(path):
    return path.replace("/", "\\")

def getText(fm, grade):
    if fm <= threshold and grade < 50:
        return "The image is too blurry, you should consider taking another photo."
    if fm <= threshold and 65 < grade:
        return "The image is quite blurry."
    return "The image is sharp!"

def sharpness_model():
    blurry_rates = []

    for i, path in enumerate(glob.glob("images/*")):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray/np.max(gray)
        fm = variance_of_laplacian(gray)

        grade = get_grade(fm)

        text = getText(fm, grade)

        blurry_rates.append((grade/100, text, parsePath(path)))

    return blurry_rates

