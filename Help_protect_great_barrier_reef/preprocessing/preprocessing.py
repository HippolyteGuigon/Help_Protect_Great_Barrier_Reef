import pandas as pd 
import ast 
import os
import logging
from typing import List
from PIL import Image
from Help_protect_great_barrier_reef.configs.confs import load_conf, clean_params, Loader
from Help_protect_great_barrier_reef.logs.logs import main
from tqdm import tqdm

tqdm.pandas()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

tqdm.pandas()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
image_width=main_params["width"]
image_height=main_params["height"]

class preprocessing_yolo:
    """
    The goal of this class is
    to preprocess images to match
    with a Yolo v5 pipeline
    
    Arguments:
        -df: pd.DataFrame: The
        Dataframe with image coordinates
        annotations to be modified
        
    Returns:
        -None
    """

    def __init__(self, df: pd.DataFrame)->None:
        self.df=df
        self.df["annotations"]=self.df["annotations"].apply(lambda x: list() if x=="[]" else ast.literal_eval(x))
        self.df["annotations"]=self.df["annotations"]+self.df["image_id"].apply(lambda x: [x])

    def convert_coordinates(self, list_dict: List[dict])->List[dict]:
        """
        The goal of this function is to
        convert the coordinates of a given 
        starfish in the DataFrame so that
        it matches with Yolo_v5 pipeline
        
        Arguments:
            -list_dict: List[dict]: The dictionnary
            in which coordianates are stored
        Returns:
            -list_dict: List[dict]: The dictionnary list
            after it was properly converted
        """

        image_id=list_dict[-1].split("-")
        image_path=os.path.join("train_images", "video_"+image_id[0], image_id[1]+".jpg")

        # Transform the bbox co-ordinates as per the format required by YOLO v5

        if len(list_dict)==1:
            return [image_path]

        list_dict.remove(list_dict[-1])

        for i, dict_coordinates in enumerate(list_dict):
            dict_coordinates["x"] += dict_coordinates["width"]/2
            dict_coordinates["y"] += dict_coordinates["height"]/2
            dict_coordinates["x"] /= image_width
            dict_coordinates["y"] /= image_height
            dict_coordinates["width"] /= image_width
            dict_coordinates["height"] /= image_height
            dict_coordinates={**{"class": 0}, **dict_coordinates}
            list_dict[i]="{} {:.3f} {:.3f} {:.3f} {:.3f}".format(0, dict_coordinates["x"], dict_coordinates["y"], dict_coordinates["width"], dict_coordinates["height"])
            
        list_dict.append(image_path)

        return list_dict

    def full_conversion(self)->None:
        """
        The goal of this function
        is to convert the coordinates
        in the DataFrame so that they
        match the Yolo algorithm requirements
        
        Arguments:
            -None
        Returns:
            -None
        """

        logging.info("Conveting annotations in yolo format...")

        self.df["annotations"]=self.df["annotations"].progress_apply(lambda x: self.convert_coordinates(x))
    
    def saving_result(self)->None:
        """
        The goal of this function
        is saving the results under
        txt files readable by the Yolo
        model
        
        Arguments:
            -None
        Returns:
            -None
        """
        
        logging.info("Saving results under Yolo format...")
        
        for elements in self.df["annotations"]:
            path_save=elements[-1].replace(".jpg",".txt")
            elements.pop(-1)

            if os.path.exists(path_save):
                continue
            print("\n".join(elements), file= open(path_save, "w"))

def clean_all_files():
    for file in os.listdir("train_images"):
        if file != ".DS_Store":
            for subfile in os.listdir(os.path.join("train_images",file)):
                if ".txt" in subfile:
                    os.remove(os.path.join("train_images", file, subfile))