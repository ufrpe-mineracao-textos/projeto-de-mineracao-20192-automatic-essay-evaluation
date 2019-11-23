"""
    This module is dedicated to the loading images related to hand written words and
    handwritten characters to be processed by the implemented algorightms.

    The datasets used both in the text detection and hand written word segmentation are the following:
    The handwritten characters NIST dataset, made available here:
    The handwritten texts IAM dataset, made available here:
    The handwritten characters and digits EMNIST dataset, made available here:


    We strongly recommend for you to download each dataset before running the experiments contained in the image
    processing folder.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd


def set_datset_directories(IAM_DIR = "", IAM_TEXT_DIR = "", NIST_DIR = "", EMNIST_DIR = ""):
    """
    Appends the path of each dataset to the system so they can be accessed no
    matter where they are in the computer.
    """
    # Appending path of the directories in to the system
    global NIST, IAM, IAM_TEXT, EMNIST
    NIST = NIST_DIR
    IAM = IAM_DIR
    IAM_TEXT = IAM_TEXT_DIR
    EMNIST = EMNIST_DIR
    sys.path.append(IAM)
    sys.path.append(IAM_TEXT)
    sys.path.append(NIST)
    sys.path.append(EMNIST)


def create_iam_dataset_path_list(custom_iam_path='IAM-dataset/inlineImages/',
                                 custom_iam_text_path='IAM-dataset/ascii/'):
    """
    Each parent directory of this dataset has a set of sub directories, each subdirectory has its corresponding
    sub subdirectory which it contains two or more images, which when put together they form a phrase.

    Instead of loading all the images in to memory, this procedure creates a list of tuples of paths,
    so that each image can be loaded on demand.

    :return: A list of tuples, with the first element being the folder where the images can be found, and
            the second element of the tuple is the images' names
            Another list of tuples with the resulting desired text from the texts detection.
            All the texts is united by phrase, which means a certain set of images in an especific folder of
            IAM dataset will actually compose the phrase of the text file in the IAM_TEXT path
    """
    #global IAM, IAM_TEXT
    phrases_list = []
    text_paths_list = []
    for subdirs, dirs, files in os.walk(custom_iam_path):
        if len(files) != 0:
            phrases_list.append((subdirs+'/', files))

    for subdirs, dirs, files in os.walk(custom_iam_text_path):
        if len(files) != 0:
            text_paths_list.append((subdirs+'/', files))

    return phrases_list, text_paths_list


def create_nist_dataset_path_list(custom_nist_path='NIST/'):
    """
    Each parent directory of this dataset has a set of sub directories, each subdirectory has its corresponding
    sub subdirectory which it contains two or more images, which when put together they form a phrase.

    Instead of loading all the images in to memory, this procedure creates a list of tuples of paths,
    so that each image can be loaded on demand.

    :return: A list of tuples, with the first element being the folder where the images can be found, and
            the second element of the tuple is the images' names
    """
    paths_list = []
    for subdir, dir, files in os.walk(custom_nist_path):

        if len(files) != 0 and all('.mit' in w for w in files):
            images_paths = []
            # Removing the folder with the train name
            relevant_dirs = [new_d for new_d in dir if 'train' not in new_d]
            for mit, mit_dir in zip(files, relevant_dirs):
                paths_df = pd.read_csv(subdir+'/'+mit, delimiter=' ', skiprows=1)
                columns = list(paths_df.columns)
                data = paths_df[columns[0]].tolist()
                data = columns + data
                # Appending path to image with name of the subdirectory
                images_paths += [mit_dir + '/' + d for d in data if mit_dir in d]

            paths_list.append((subdir+'/', images_paths))

    return paths_list


def load_essays(path=".../../../data/"):

    essays = []
    for sub_dir, dir, files in os.walk(path):
        for file in files:
            file_name = file.replace('.png', '')
            essays.append(cv2.imread(sub_dir+'/'+file))

    return essays
