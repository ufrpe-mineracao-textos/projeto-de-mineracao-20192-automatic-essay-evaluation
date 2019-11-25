"""
    This module is dedicated to the implementation of handwritten words in an image
"""
import cv2
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from data_procedures import create_iam_dataset_path_list, load_essays
from imutils.object_detection import non_max_suppression
from line_remover import bining_image, findLinesInEssays, show_textLines
user_folder = "user_folder"
datasets_folder = "folder_where_the_datasets_are_located"

NIST = "/home/" + user_folder + "/" + datasets_folder + "/NIST"
IAM = "/home/" + user_folder + "/" + datasets_folder + "/IAM-dataset/lineImages/"
IAM_TEXT = "/home/"+user_folder+"/"+datasets_folder+"/IAM-dataset/ascii/"
EMNIST = "/home/" + user_folder + "/" + datasets_folder + "/EMNIST"


def word_segmentation(image, verbose=False, thresh_distance=.001, distance='euclidean'):

    distance_f = None
    if distance == 'euclidean':
        distance_f = euclidean
    elif distance == 'cosine':
        distance_f = cosine
    else:
        raise Exception("Undifined distance type: "+distance)

    gray = image
    if len(image.shape) == 3:
        if verbose:
            print("[INFO] Converting image to gray scale")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if verbose:
        print("[INFO] Applying gaussian blur to image")
    blur_gray = cv2.GaussianBlur(gray, (3, 3), 25)

    if verbose:
        print("[INFO] Applying adaptive thresholding to image")
    th_binary = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 27)

    if verbose:
        print("[INFO] Calculating connected components with stats")
    n_comps, labels, stats, centroids = cv2.connectedComponentsWithStats(th_binary)

    # Removing background stats and centroids
    stats = stats[1:]
    centroids = centroids[1:]

    if verbose:
        print("[INFO] Sorting centroids according to the x axis")
    centroids = np.array([[i, c[0], c[1]] for i, c in enumerate(centroids)])
    centroids = np.array(sorted(centroids, key=lambda x: x[1]))

    # Cluster connnected closer connected components together
    # Adding a visited variable, to avoid centroids in the same cluster to be added twice
    if verbose:
        print("[INFO] Clustering centroids of connected components")
    marked_centroids = [[-1, c] for c in centroids]
    clusters = []

    for i, (cluster_lbl1, c1) in enumerate(marked_centroids):

        clusters.append([])

        for j, (cluster_lbl2, c2) in enumerate(marked_centroids):

            if i == j:
                if cluster_lbl1 == -1:
                    marked_centroids[j][0] = i
                    cluster_lbl1 = i
                    clusters[i].append(c2)
            elif cluster_lbl1 != -1:
                dist = distance_f(c1, c2)
                if dist <= thresh_distance:
                    if cluster_lbl2 == -1:
                        marked_centroids[j][0] = cluster_lbl1
                        clusters[i].append(c2)

    # Getting rid of empty centroid clusters
    clusters = [c for c in clusters if len(c) != 0]
    if verbose:
        print("[INFO] Sorting centroids by the x axis, and creating vertices for bounding box, word rectangles")

    rects = []
    for c in clusters:
        # clusters[i] = np.array(sorted(clusters[i], key=lambda x: x[1]))
        c = np.array(sorted(c, key=lambda x: x[1]))

        # Creating bounding boxes based on the indexes of the rectangles
        if len(c) == 1:
            idx = int(c[0][0])
            start_x, start_y, w, h = stats[idx][:-1]
            end_x, end_y = start_x + w, start_y + h
            rects.append((start_x, start_y, end_x, end_y))
        else:
            first_idx = int(c[0][0])
            start_f_x, start_f_y, _w, _h = stats[first_idx][:-1]

            last_idx = int(c[-1][0])
            start_l_x, start_l_y, w, h = stats[last_idx][:-1]
            end_x, end_y = start_l_x + w, start_l_y + h
            rects.append((start_f_x, start_f_y, end_x, end_y))

    rects = non_max_suppression(np.array(rects))
    return rects


def show_images(*args):
    for i, a in enumerate(args):
        #cv2.imshow("Output "+str(i), a)
        cv2.imwrite("Output "+str(i)+".png", a)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    essays = load_essays(path="essay_pictures_folder", gray_scale=True)

    lines = findLinesInEssays(essays)

    essays_keys = list(essays.keys())
    e = essays_keys[0]

    segmented_lines = show_textLines(essays[e], lines[e])

    for i, s in enumerate(segmented_lines):
        cv2.imwrite("Line "+str(i)+".png", s)

    """
    images_paths, texts_paths = create_iam_dataset_path_list(IAM, IAM_TEXT)
    choice = 0
    # choice = 0
    chosen_path = images_paths[choice][0] + images_paths[choice][1][0]

    image = cv2.imread(chosen_path)

    # Euclidean 400, 900, cosine:.059, 0.29
    rectangles = word_segmentation(image, thresh_distance=0.029, distance='cosine')

    detection = image.copy()

    for vertices in rectangles:
        cv2.rectangle(detection, (vertices[0], vertices[1]), (vertices[2], vertices[3]), (0, 255, 0), 10)

    show_images(image, detection)
    """
