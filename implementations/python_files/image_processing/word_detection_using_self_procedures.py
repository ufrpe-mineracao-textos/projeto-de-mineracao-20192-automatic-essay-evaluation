"""
    This module is dedicated to the implementation of handwritten words in an image
"""
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from scipy.spatial.distance import cosine, euclidean

from data_procedures import create_iam_dataset_path_list

user_folder = "ismael"
datasets_folder = "Downloads"

NIST = "/home/" + user_folder + "/" + datasets_folder + "/NIST"
IAM = "/home/" + user_folder + "/" + datasets_folder + "/IAM-dataset/lineImages/"
IAM_TEXT = "/home/"+user_folder+"/"+datasets_folder+"/IAM-dataset/ascii/"
EMNIST = "/home/" + user_folder + "/" + datasets_folder + "/EMNIST"




def show_images(*args):
    for i, a in enumerate(args):
        cv2.imshow("Output "+str(i), a)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_paths, texts_paths = create_iam_dataset_path_list(IAM, IAM_TEXT)
    choice = 0
    chosen_path = images_paths[choice][0] + images_paths[choice][1][0]

    image = cv2.imread(chosen_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (3, 3), 25)

    th_binary = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 27)

    n_comps, labels, stats, centroids = cv2.connectedComponentsWithStats(th_binary)

    # Removing background stats and centroids
    stats = stats[1:]
    centroids = centroids[1:]

    chosen_stats = 2
    start_x, start_y = stats[chosen_stats][:2]
    end_x, end_y = stats[chosen_stats][2]+start_x, stats[chosen_stats][3]+start_y

    # Cluster connnected closer connected components together
    # Adding a visited variable, to avoid centroids in the same cluster to be added twice
    marked_centroids = [[False, c] for c in centroids]
    thresh_distance = 0.001
    clusters = []
    for i, (v_1, c1) in enumerate(marked_centroids):
        # Compares each centroid with every other centroid and clusterize the closest ones
        marked_centroids[i][0] = True
        clusters.append([])
        for j, (v_2, c2) in enumerate(marked_centroids):
            # ignore same centroid comparisons
            if i == j:
                continue

            dist = cosine(c1, c2)
            if dist <= thresh_distance:
                if not v_2:
                    # Mark centroid as added to a cluster
                    marked_centroids[j][0] = True
                    c2 = c2.tolist()
                    clusters[i].append([j, c2[0], c2[1]])

    # Getting rid of empty centroid clusters
    clusters = [c for c in clusters if len(c) != 0]
    # Sorting centroid clusters by x axis
    for i, c in enumerate(clusters):
        clusters[i] = np.sort(clusters[i], axis=1)

    print(len(clusters))
