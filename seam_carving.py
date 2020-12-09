import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import png
from PIL import Image
import cv2
import cv2 as cv2
from numpy.matrixlib.defmatrix import matrix
from numba import jit, prange


def getGradient(image):
    # Normalize Sobel
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(np.square(Gx) + np.square(Gy))


@jit(nopython=True)
# optimizing the bottom up recursive with parallel programming
def computeM(g):
    m = np.zeros(shape=(g.shape[0], g.shape[1]), dtype=np.float64)
    m[g.shape[0]-1] = np.copy(g[g.shape[0]-1])
    f = True
    for i in prange(g.shape[0]-2, -1, -1):
        for col in prange(0, g.shape[1]):
            val = [m[i+1, col]]
            if (col > 0):
                val.append(m[i+1, col-1])
            if (col < g.shape[1]-1):
                val.append(m[i+1, col+1])
            m[i, col] = (min(val) + g[i, col])
    return m


def constructImage(img, m):
    copy_img = np.copy(img)
    row = 0
    parent = np.array([row, np.argmin(m[row])])
    new_image = np.zeros((img.shape[0], img.shape[1]-1, 3), dtype=np.uint8)
    what = np.delete(copy_img[0], np.argmin(m[row]), axis=0)
    new_image[0, ] = what
    while(row < img.shape[0]-1):
        y = parent[0]
        x = parent[1]
        directions = [(y+1, x)]
        mins = [m[y+1, x]]

        if x > 0:
            directions.append((y+1, x-1))
            mins.append(m[y+1, x-1])
        if x < img.shape[1]-1:
            directions.append((y+1, x+1))
            mins.append(m[y+1, x+1])
        index = np.argmin(np.array(mins))
        parent = directions[index]
        row = row + 1
        new_image[row] = np.delete(copy_img[row], parent[1], axis=0)
    copy_img = new_image
    return copy_img


def seam_carve(img, y, x):

    rows = img.shape[0]
    cols = img.shape[1]
    num_iterations_x = cols - x
    num_iterations_y = rows - y
    for i in range(num_iterations_x):
        g = getGradient(img)
        m = computeM(g)
        img = constructImage(img, m)
        print(img.shape)
    if (num_iterations_y > 0):
        img = img.transpose(1, 0, 2)

    for i in range(num_iterations_y):
        g = getGradient(img)
        m = computeM(g)
        img = constructImage(img, m)
        print(img.shape)

    if (num_iterations_y > 0):
        img = img.transpose(1, 0, 2)

    return img


img = cv2.imread("./ex3.jpg")
carved_img = seam_carve(img, 861, 1200)
height = carved_img.shape[0]
width = carved_img.shape[1]
im = cv2.cvtColor(carved_img, cv2.COLOR_BGR2RGB)

im = Image.fromarray(im, "RGB")
im.save("ex3_carved.png")
