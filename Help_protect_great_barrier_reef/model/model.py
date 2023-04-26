import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import random
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Help_protect_great_barrier_reef.preprocessing.preprocessing import preprocessing_yolo, clean_all_files

class yolo_model:

    def __init__(self):
        self.df=pd.read_csv("train.csv")
        preprocess=preprocessing_yolo(df)
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
        train_test_split(images, annotations, test_size=0.2, random_state=42)

        self.val_images, self.test_images, self.val_annotations, self.test_annotations =\
        train_test_split(self.val_images, self.val_annotations, test_size=0.5, random_state=42)

    def split_files(self)->None:
        if not all([hasattr(self, attr) for attr in ["train_images",
        "val_images","train_annotations","val_annotations", "test_annotations"]]):
            raise AssertionError("Definition of split sets was not done,\
                                 please call the get_split method")
        
        pass        