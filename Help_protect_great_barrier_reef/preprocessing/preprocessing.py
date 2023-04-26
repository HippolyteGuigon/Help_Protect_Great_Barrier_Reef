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

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        
        for i, dict_coordinates in enumerate(list_dict):
            dict_coordinates["x"] += dict_coordinates["width"]/2
            dict_coordinates["y"] += dict_coordinates["height"]/2
            dict_coordinates["x"] /= image_width
            dict_coordinates["y"] /= image_height
            dict_coordinates["width"] /= image_width
            dict_coordinates["height"] /= image_height
            dict_coordinates={**{"class": 0}, **dict_coordinates}
            list_dict[i]=dict_coordinates

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

        self.df["annotations"]=self.df["annotations"].progress_apply(lambda x: 
                            list() if x=="[]" else self.convert_coordinates(ast.literal_eval(x)))
      