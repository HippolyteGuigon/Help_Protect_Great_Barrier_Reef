import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import random
import shutil
import glob
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib import tzip
from Help_protect_great_barrier_reef.preprocessing.preprocessing import preprocessing_yolo, clean_all_files
from Help_protect_great_barrier_reef.logs.logs import main

main()

class yolo_model:

    def __init__(self):
        self.df=pd.read_csv("train.csv")
        preprocess=preprocessing_yolo(self.df)
        preprocess.full_conversion()
        preprocess.saving_result()

    def get_split(self)->None:
        """
        The goal of this function
        is to get the different sets 
        for the training of the yolo
        model
        
        Arguments: 
            -None
        Returns:
            -None
        """
        annotations = glob.glob('train_images/*/*.txt')
        images = glob.glob("train_images/*/*.jpg")

        annotations.sort()
        images.sort()

        self.train_images, self.val_images, self.train_annotations, self.val_annotations = \
        train_test_split(images, annotations, test_size=0.2)

        self.val_images, self.test_images, self.val_annotations, self.test_annotations =\
        train_test_split(self.val_images, self.val_annotations, test_size=0.5)

    def split_files(self)->None:
        if not all([hasattr(self, attr) for attr in ["train_images",
        "val_images","train_annotations","val_annotations", "test_annotations"]]):
            raise AssertionError("Definition of split sets was not done,\
                                 please call the get_split method")
        
        for path_set in ["train_set", "test_set", "valid_set"]:
            if not os.path.exists(path_set):
                os.mkdir(path_set)
            else:
                shutil.rmtree(path_set)
                os.mkdir(path_set)     

        logging.info("Splitting the files between the different sets...")

        for train_image_path, test_image_path, valid_image_path\
              in tzip(self.train_images, self.test_images, self.val_images):
            shutil.copy(train_image_path, "train_set")
            shutil.copy(test_image_path, "test_set")
            shutil.copy(valid_image_path, "valid_set")

            shutil.copy(train_image_path.replace(".jpg", ".txt"), "train_set")
            shutil.copy(test_image_path.replace(".jpg", ".txt"), "test_set")
            shutil.copy(valid_image_path.replace(".jpg", ".txt"), "valid_set")

if __name__ == '__main__':
    test=yolo_model()
    test.get_split()
    test.split_files() 