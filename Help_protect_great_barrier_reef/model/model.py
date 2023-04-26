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

df=pd.read_csv("train.csv")

preprocess=preprocessing_yolo(df)
preprocess.full_conversion()
preprocess.saving_result()

annotations = glob.glob('train_images/*/*.txt')
images = glob.glob("train_images/*/*.jpg")

annotations.sort()
images.sort()

train_images, val_images, train_annotations, val_annotations = \
train_test_split(images, annotations, test_size=0.2, random_state=42)

val_images, test_images, val_annotations, test_annotations =\
train_test_split(val_images, val_annotations, test_size=0.5, random_state=42)

print(val_images)