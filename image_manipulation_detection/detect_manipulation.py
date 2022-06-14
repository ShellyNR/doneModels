import glob
import os

import numpy as np
import image_slicer
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.io import imread
from itertools import combinations

N = 47185 # number of slices
dir = "./data"

# def N_based_on_image_size(x, y):

def num_of_slices(size):
    slices = np.rint(np.divide(size, N))
    if slices < 10 :
        return 10;
    if slices > 50:
        return 50
    return slices

def gaussian_filter1(image):
    image = img_as_float(image)
    image = gaussian_filter(image,1)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    return dilated

def filtered_image(image):
    image1 = image
    image2 = gaussian_filter1(image)
    img3 =  image1-image2
    io.imsave("out.png", img3)
    return "out.png"

def detect_manupulation():
    fake_rates = []
    for i, path in enumerate(glob.glob("images/*")):
        image = imread(path)
        print(image.size)
        slices = num_of_slices(image.size)
        print (slices)
        sliced_images = image_slicer.slice(filtered_image(image),slices, save=False)
        os.makedirs(dir, exist_ok=True)
        image_slicer.save_tiles(sliced_images, directory=dir, prefix='slice')

        list_files = []
        count = 0
        sum = 0
        for file in os.listdir(dir):
            list_files.append(file)
        for i in combinations(list_files,2):
            img1 = imread(dir + '/' + i[0])
            img2 = imread(dir + '/' + i[1])
            diff = img1 - img2

            diff_btwn_img_data = np.linalg.norm(diff,axis=1)
            # print("diff between " + str(i) + " two images is " + str(np.mean(diff_btwn_img_data)))
            sum += np.mean(diff_btwn_img_data)
            count +=1

        print ("total score:" + str(np.divide(sum, count)))
        print (path)
        fake_rates.append((np.float64(np.divide(sum, count)), "fake_decs_here" , path))
        files = glob.glob('data/*')
        for f in files:
            os.remove(f)

    return fake_rates


