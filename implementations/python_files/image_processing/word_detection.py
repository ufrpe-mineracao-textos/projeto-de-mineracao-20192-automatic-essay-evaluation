"""
    This module is dedicated to the implementation of handwritten words in an image
"""
import cv2
from data_procedures import create_iam_dataset_path_list

user_folder = "user"
datasets_folder = "datasets_folders"

NIST = "/home/" + user_folder + "/" + datasets_folder + "/NIST"
IAM = "/home/" + user_folder + "/" + datasets_folder + "/IAM-dataset/lineImages/"
IAM_TEXT = "/home/"+user_folder+"/"+datasets_folder+"/IAM-dataset/ascii/"
EMNIST = "/home/" + user_folder + "/" + datasets_folder + "/EMNIST"


def word_detection(image, verbose=True):
    """
    This procedure call is destined to the implementation of the word detection algorithm

    :param image: loaded image
    :param v: Option to activate verbosity of procedure
    :return:
    """
    if verbose:
        print("[INFO] Binarizing image using adaptive threshold com o método gaussiano")

    binary = cv2.adaptiveThreshold(image, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 25)

    cv2.imshow("Binarized", 255 - binary)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_paths, texts_paths = create_iam_dataset_path_list(IAM, IAM_TEXT)
    choice = 4
    chosen_path = images_paths[choice][0]+images_paths[choice][1][0]

    image = cv2.imread(chosen_path, 0)

    word_detection(image, verbose=True)

