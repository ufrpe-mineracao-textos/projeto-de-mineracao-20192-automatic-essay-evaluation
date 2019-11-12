"""
    This module is dedicated to the implementation of handwritten words in an image
"""
import cv2
from data_procedures import create_iam_dataset_path_list

user_folder = "user"
datasets_folder = "ds_folder"

NIST = "/home/" + user_folder + "/" + datasets_folder + "/NIST"
IAM = "/home/" + user_folder + "/" + datasets_folder + "/IAM-dataset/lineImages/"
IAM_TEXT = "/home/"+user_folder+"/"+datasets_folder+"/IAM-dataset/ascii/"
EMNIST = "/home/" + user_folder + "/" + datasets_folder + "/EMNIST"

if __name__ == '__main__':
    images_paths, texts_paths = create_iam_dataset_path_list(IAM, IAM_TEXT)

    chosen_path = images_paths[0][0]+images_paths[0][1][4]

    image = cv2.imread(chosen_path)
    cv2.imshow("Teste", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

