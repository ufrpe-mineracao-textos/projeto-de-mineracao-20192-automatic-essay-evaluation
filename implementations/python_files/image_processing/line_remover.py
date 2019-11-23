"""
    Authors: Lucas Correia e Wilson Neto
    This module is dedicated to the procedures implemented by Lucas and Wilson to remove lines
    in an image of an essay
"""
import cv2
import numpy as np


def remove_lines_horizontal(img, i):
    if (i):
        kernel = np.array([[-2, 4, -2], [-4, 7.55, -4], [-2, 4, -2]])
    else:
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    leftBorder = cv2.filter2D(img, -1, kernel)
    rightBorder = cv2.filter2D(img, -1, -kernel)

    borders = np.bitwise_or(leftBorder, rightBorder)
    pivot = 128
    borders[borders < pivot] = 0
    borders[borders >= pivot] = 255

    kernel2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    dilate = cv2.dilate(borders, kernel2, iterations=1)
    # erode = cv2.erode(dilate, kernel2, iterations=1)

    # dilate = cv2.dilate(borders, kernel2, iterations=1)
    # dilate = cv2.dilate(borders, kernel2, iterations=1)
    # erode = cv2.erode(dilate, kernel2, iterations=1)
    return dilate


def remove_lines_vertical(img, i):

    if (i):
        kernel = np.array([[-2, -4, -2], [4, 7.55, 4], [-2, -4, -2]])
    else:
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    topBorder = cv2.filter2D(img, -1, kernel)
    bottomBorder = cv2.filter2D(img, -1, -kernel)

    borders = np.bitwise_or(topBorder, bottomBorder)

    pivot = 100
    borders[borders < pivot] = 0
    borders[borders >= pivot] = 255

    kernel2 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

    dilate = cv2.dilate(borders, kernel2, iterations=1)

    erode = cv2.erode(dilate, kernel2, iterations=1)

    # res = np.bitwise_and(dilate, img)

    # res = cv2.dilate(erode, kernel2, iterations=1)

    return erode


def bining_image(img, pivot=255):
    """
    Thresholds a copy of an image using a pivot as a paarmeter
    :param img: Image
    :param pivot: Pixel value that determines how the image is going to be limiarized
    :return: Binarized image
    """
    bining = img.copy()
    bining[bining < pivot] = 0
    bining[bining >= pivot] = 255
    return bining


def remove_lines(img, iterations=10):
    #return remove_lines_vertical(remove_lines_horizontal(img))
    #img = cv2.imread('captura.PNG', 0)
    #cv2_imshow(img)
    pivot = 160
    binimg = bining_image(img, pivot=pivot)
    kernel=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    new_img = remove_lines_horizontal(binimg, True)
    new_img = remove_lines_vertical(new_img, True)
    new_img = cv2.GaussianBlur(new_img, (3, 3), 5)
    pivot = 32
    #new_img[new_img < pivot] = 0
    #new_img[new_img >= pivot] = 255
    new_img = bining_image(new_img, pivot=pivot)
    new_img = cv2.erode(new_img, kernel, iterations=1)
    for i in range(iterations):
        new_img = cv2.GaussianBlur(new_img, (5, 5), 5)
        pivot = 32
        # new_img[new_img < pivot] = 0
        # new_img[new_img >= pivot] = 255
        new_img = bining_image(new_img, pivot=pivot)
        new_img = cv2.erode(new_img, kernel, iterations=1)
        new_img = np.bitwise_and(new_img, binimg)

    return new_img


def get_labeled_connected_components(img, connectivity=8):
    """
    Takes a binarized image, calculates its connected components and creates a new image with connected components
    labeled by color
    :param img: Image
    :param connectivity: Connectivity level to be used in the connected components
    :return:
    """
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    label_hue = np.uint8(100 * ret * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return labeled_img
