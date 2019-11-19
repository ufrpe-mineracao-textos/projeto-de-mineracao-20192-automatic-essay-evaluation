"""
    This module is dedicated to the implementation of handwritten words in an image
    the pretrained machine learning model for text detection has been downloaded in
    this link: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
"""
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from data_procedures import create_iam_dataset_path_list

user_folder = "user_folder"
datasets_folder = "folder_where_the_datasets_are_located"

NIST = "/home/" + user_folder + "/" + datasets_folder + "/NIST"
IAM = "/home/" + user_folder + "/" + datasets_folder + "/IAM-dataset/lineImages/"
IAM_TEXT = "/home/"+user_folder+"/"+datasets_folder+"/IAM-dataset/ascii/"
EMNIST = "/home/" + user_folder + "/" + datasets_folder + "/EMNIST"


def preprocess_image(image, verbose=True):
    """
    Perform preprocessing techniques, of adaptive thresholding,
    and calculation of gradient, as well as boarder detection using canny
    :param image:
    :param verbose:
    :return: the pre processing of the image using different channels
    """
    if verbose:
        print("[INFO] Applying gaussian blur onto image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 5)

    if verbose:
        print("[INFO] Applying Adaptive thresholding onto image, using Adptive thresholdin Gaussian")
    th_binary = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 25)

    if verbose:
        print("[INFO] Applying Rotated laplacian to the image")

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    rot_laplacian = cv2.filter2D(th_binary, -1, kernel)

    if verbose:
        print("[INFO] Making connected components of image ")

    connectec_comps, comps_labels = cv2.connectedComponents(th_binary)

    comps_labels = comps_labels - 1

    rows = image.shape[0]
    cols = image.shape[1]
    new_img = []
    for i in range(rows):
        new_img.append([])
        for j in range(cols):
            val_rot_lpl = rot_laplacian[i, j]
            val_th_binary = th_binary[i, j]
            val_can_eroded = comps_labels[i, j]
            new_img[i].append([val_rot_lpl, val_th_binary, val_can_eroded])

    new_img = np.array(new_img, dtype=np.uint8)

    return new_img


def word_detection(image, verbose=True,
                   new_dimentions=(320, 320),
                   min_confidence=0.5,
                   preprocess=False,
                   model_path="frozen_east_text_detection.pb"):
    """
    Applying the machine learning model to perform the text detection in the image.
    To perform text detection a pre-trained machine learning model is being used.
    The model is called EAST: Efficient and accurate scene text detector.
    :param image: the image to be loaded.
    :param verbose: the option indicating verbosity
    :return: the list of vertices of possible rectangles containing the text.
    """

    p_image = image.copy()

    if preprocess:
        if verbose:
            print("[INFO] Pre-processing Image")
        p_image = preprocess_image(p_image, verbose=verbose)

    mean_c1 = p_image[:, :, 0].mean()
    mean_c2 = p_image[:, :, 1].mean()
    mean_c3 = p_image[:, :, 2].mean()

    (W, H) = p_image.shape[:2]

    (newW, newH) = new_dimentions

    rW = W / newW
    rH = W / newH

    p_image = cv2.resize(p_image, (newW, newH))

    (W, H) = p_image.shape[:2]

    if verbose:
        print("[INFO] Loading EAST Machine learning model")

    model = cv2.dnn.readNet(model_path)

    image_blob = cv2.dnn.blobFromImage(p_image, 1.0, (W, H),
                                       (123.68, 116.78, 103.94),
                                       swapRB=True, crop=False)

    model.setInput(image_blob)

    if verbose:
        print("[INFO] Making foward pass of an image")

    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]

    (scores, geometry) = model.forward(layerNames)

    if verbose:
        print("[INFO] Scores shape: ", scores.shape)
        print("[INFO] Geometry shape: ", geometry.shape)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []


    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for i, (startX, startY, endX, endY) in enumerate(boxes):
        boxes[i][0] = int(startX * rW)
        boxes[i][1] = int(startY * rH)
        boxes[i][2] = int(endX * rW)
        boxes[i][3] = int(endY * rH)

    return boxes


def draw_rectangles(image, boxes):
    """
    Draw rectangles onto the image

    :return: the image with the rectangles drawed
    """

    for (startX, startY, endX, endY) in boxes:
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    images_paths, texts_paths = create_iam_dataset_path_list(IAM, IAM_TEXT)
    choice = 3
    chosen_path = images_paths[choice][0] + images_paths[choice][1][0]

    image = cv2.imread(chosen_path)

    boxes = word_detection(image, preprocess=False, verbose=True)

    detection = draw_rectangles(image.copy(), boxes)

    cv2.imshow("Original: ", image)
    cv2.imshow("Detection: ", detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
