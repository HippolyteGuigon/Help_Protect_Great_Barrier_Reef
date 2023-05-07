#Ici, on veut implémenter un modèle MaskRCNN

import glob
import shutil
import numpy as np
import logging
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Help_protect_great_barrier_reef.logs.logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

class MaskRCNN:
    """
    The goal of this class
    is the implementation of
    the MaskRCNN model
    """

    def __init__(self)->None:
        logging.warning("MASK RCNN Model launched !")

    def split_images(self, train_size:float=0.9)->None:
        """
        The goal of this function
        is to split the images between
        train and validation sets
        
        Arguments:
            -train_size: float: The
            proportion of images allocated
            to the train set
            
        Returns:
            -None
        """

        logging.info("Allocating images in train/val set...")

        train_image_path="Help_protect_great_barrier_reef/model/Custom_MaskRCNN/samples/custom/dataset/train"
        val_image_path="Help_protect_great_barrier_reef/model/Custom_MaskRCNN/samples/custom/dataset/val"

        for file in os.listdir(train_image_path):
            os.remove(os.path.join(train_image_path,file))
        for file in os.listdir(val_image_path):
            os.remove(os.path.join(val_image_path,file))

        all_images=glob.glob("train_images/*/*.jpg")

        train_set, val_set = train_test_split(all_images, test_size=0.1)

        for image_train_path in tqdm(train_set):
            shutil.copy(image_train_path,train_image_path)
            entire_name=image_train_path.split("/")[-2]+"_"+image_train_path.split("/")[-1]
            os.rename(os.path.join(train_image_path,
                                   image_train_path.split("/")[-1]), os.path.join(
                train_image_path,entire_name))

        for image_val_path in tqdm(val_set):
            shutil.copy(image_val_path,val_image_path)
            entire_name=image_val_path.split("/")[-2]+"_"+image_val_path.split("/")[-1]
            os.rename(os.path.join(val_image_path,
                                   image_val_path.split("/")[-1]), os.path.join(
                val_image_path,entire_name))
            
        logging.info("Images allocated !")

if __name__ == '__main__':
    main()
    test=MaskRCNN()
    test.split_images()
