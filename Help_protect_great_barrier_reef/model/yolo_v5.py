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
from PIL import Image
from typing import List
from Help_protect_great_barrier_reef.preprocessing.preprocessing import preprocessing_yolo, clean_all_files
from Help_protect_great_barrier_reef.logs.logs import main
from Help_protect_great_barrier_reef.configs.confs import load_conf, clean_params, Loader

class yolo_model:
    """
    The goal of this class is
    the implementation of the
    yolov5 model
    
    Arguments:
        -preprocessing: bool: Determines
        wheter train files should be preprocessed
    Returs:
        -None
    """
    def __init__(self, preprocessing=True):
        main_params = load_conf("configs/main.yml", include=True)
        main_params = clean_params(main_params)
        split_path = main_params["split_path"]
        train_file_path = main_params["train_file_path"]
        fitted_model_path = main_params["fitted_model_path"]
        
        self.df=pd.read_csv("train.csv")
        self.split_path=split_path
        self.train_file_path=train_file_path
        self.fitted_model_path=fitted_model_path

        if preprocessing:
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

        print(len(images))
        print(len(annotations))
        
        self.train_images, self.val_images, self.train_annotations, self.val_annotations = \
        train_test_split(images, annotations, test_size=0.2)
        
        self.val_images, self.test_images, self.val_annotations, self.test_annotations =\
        train_test_split(self.val_images, self.val_annotations, test_size=0.5)


    def split_files(self)->None:
        if not all([hasattr(self, attr) for attr in ["train_images",
        "val_images","train_annotations","val_annotations", "test_annotations"]]):
            raise AssertionError("Definition of split sets was not done,\
                                 please call the get_split method")
        
        if not os.path.exists(self.split_path):
            self.split_path=os.path.join(os.getcwd(),self.split_path)
            
        for path_set in ["train_set", "test_set", "valid_set"]:
            full_path=os.path.join(self.split_path,path_set)
            if not os.path.exists(full_path):
                logging.info("Parsed here 1")
                os.mkdir(full_path)
                logging.info("Success 1")
            else:
                logging.info("Parsed here 2")
                shutil.rmtree(full_path)
                os.mkdir(full_path)     
                logging.info("Success 2")

        logging.info("Splitting the files between the different sets...")

        logging.info("Allocating test set image...")
        for test_image_path in tqdm(self.test_images):
            video_name=test_image_path.split("/")[1]
            image_name=test_image_path.split("/")[2]
            shutil.copy(test_image_path, os.path.join(self.split_path,"test_set"))
            shutil.copy(test_image_path.replace(".jpg", ".txt"),  
                        os.path.join(self.split_path,"test_set"))
            os.rename(os.path.join(self.split_path,"test_set",image_name),
                      os.path.join(self.split_path,"test_set",video_name+"_"+image_name))
            os.rename(os.path.join(self.split_path,"test_set",
                                   image_name.replace(".jpg", ".txt")),os.path.join(self.split_path,"test_set",video_name+"_"+image_name.replace(".jpg", ".txt")))
                
        logging.info("Allocating validation set image...")
        for valid_image_path in tqdm(self.val_images):
            video_name=valid_image_path.split("/")[1]
            image_name=valid_image_path.split("/")[2]
            shutil.copy(valid_image_path,
                        os.path.join(split_path,"valid_set"))
            shutil.copy(valid_image_path.replace(".jpg", ".txt"),
                        os.path.join(split_path,"valid_set"))
            os.rename(os.path.join(split_path,
                                   "valid_set",image_name),os.path.join(split_path,"valid_set",video_name+"_"+image_name))
            os.rename(os.path.join(split_path,
                                   "valid_set",image_name.replace(".jpg", ".txt")),os.path.join(split_path,"valid_set",video_name+"_"+image_name.replace(".jpg", ".txt")))

        logging.info("Allocating train set image...")
        for train_image_path in tqdm(self.train_images):
            video_name=train_image_path.split("/")[1]
            image_name=train_image_path.split("/")[2]
            shutil.copy(train_image_path, os.path.join(split_path,"train_set"))
            shutil.copy(train_image_path.replace(".jpg", ".txt"),  
                        os.path.join(split_path,"train_set"))
            os.rename(os.path.join(split_path,"train_set",
                                   image_name),os.path.join(split_path,"train_set",video_name+"_"+image_name))
            os.rename(os.path.join(split_path,"train_set",
                                   image_name.replace(".jpg", ".txt")),os.path.join(split_path,"train_set",video_name+"_"+image_name.replace(".jpg", ".txt")))
        
        logging.info("Split done !")

    def fit(self)->None:
        """
        The goal of this
        function is to launch
        the fitting of the Yolo
        model
        
        Arguments:
            -None
        
        Returns:
            -None
        """
        
        logging.warning("Fitting of the model has begun")
        os.system("python3 "+self.train_file_path)
        logging.warning("Fitting of the model has ended")

    def model_loading(self)->None:
        """
        The goal of this function is to
        load the model 
        
        Arguments:
            -None
            
        Returns:
            -None
        """

        if not os.path.exists(self.fitted_model_path):
            raise ValueError("The model needs to be fitted before predicting")

        model = torch.hub.load(
                "ultralytics/yolov5", "custom", self.fitted_model_path
                )
        
        self.model=model


    def predict(self, image_path: str)->List[dict]:
        """
        The goal of this function is
        to predict if objects are detected
        on a given  image
        
        Arguments:
            -image_path: str: The path  of the
            image to be predicted
            
        Returns:
            -prediction: List[dict]: The prediction for
            a given image
        """

        if not os.path.exists(self.fitted_model_path):
            raise ValueError("The model needs to be fitted before predicting")
        
        if not hasattr(self,"model"):
            self.model_loading()
        
        image=Image.open(image_path)

        results=self.model(image)
        results=results.pandas().xyxy[0]

        predictions=[]

        if results.shape[0]==0:
            return dict()
        else:
            for index in range(results.shape[0]):
                pred=results.loc[index,["xmin","ymin","xmax","ymax"]].to_dict()
                predictions.append(pred)

        return predictions

if __name__ == '__main__':
    test=yolo_model()
    test.get_split()
    test.split_files() 
    test.fit()