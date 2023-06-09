import pandas as pd
import ast
import os
import logging
import glob
import shutil
from typing import List
from PIL import Image
from Help_protect_great_barrier_reef.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)
from Help_protect_great_barrier_reef.logs.logs import main
from tqdm import tqdm

tqdm.pandas()

main()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
image_width = main_params["width"]
image_height = main_params["height"]
faster_cnn_model_path = main_params["faster_cnn_model_path"]


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

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.df["annotations"] = self.df["annotations"].apply(
            lambda x: list() if x == "[]" else ast.literal_eval(x)
        )
        self.df["annotations"] = self.df["annotations"] + self.df["image_id"].apply(
            lambda x: [x]
        )

    def convert_coordinates(self, list_dict: List[dict]) -> List[dict]:
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

        image_id = list_dict[-1].split("-")
        image_path = os.path.join(
            "train_images", "video_" + image_id[0], image_id[1] + ".jpg"
        )

        # Transform the bbox co-ordinates as per the format required by YOLO v5

        if len(list_dict) == 1:
            return [image_path]

        list_dict.remove(list_dict[-1])

        for i, dict_coordinates in enumerate(list_dict):
            dict_coordinates["x"] += dict_coordinates["width"] / 2
            dict_coordinates["y"] += dict_coordinates["height"] / 2
            dict_coordinates["x"] /= image_width
            dict_coordinates["y"] /= image_height
            dict_coordinates["width"] /= image_width
            dict_coordinates["height"] /= image_height
            dict_coordinates = {**{"class": 0}, **dict_coordinates}
            list_dict[i] = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                0,
                dict_coordinates["x"],
                dict_coordinates["y"],
                dict_coordinates["width"],
                dict_coordinates["height"],
            )

        list_dict.append(image_path)

        return list_dict

    def full_conversion(self) -> None:
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

        self.df["annotations"] = self.df["annotations"].progress_apply(
            lambda x: self.convert_coordinates(x)
        )

    def saving_result(self) -> None:
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
            path_save = elements[-1].replace(".jpg", ".txt")
            elements.pop(-1)

            if os.path.exists(path_save):
                continue
            print("\n".join(elements), file=open(path_save, "w"))


class preprocessing_faster_rcnn:
    """
    The goal of this class is
    to preprocess data for the
    implementation of the faster
    rcnn method

    Arguments:
        -df: pd.DataFrame: The
        original train DataFrame
        from which image information
        will be retreived
    Returns:
        -None
    """

    def __init__(self) -> None:
        self.df = pd.read_csv("train.csv")

    def dataframe_preprocessing(self) -> None:
        """
        The goal of this function
        is to preprocess the original
        train DataFrame so that it
        corresponds to faster cnn train
        format.

        Arguments:
            -None
        Returns:
            -None
        """

        self.df["image_path"] = self.df["image_id"].apply(
            lambda x: "train_images/video_" + x[0] + "/" + x.split("-")[-1] + ".jpg"
        )
        self.df = self.df[
            self.df["image_path"].apply(lambda path: os.path.exists(path))
        ]
        self.df["annotations"] = self.df["annotations"].apply(
            lambda x: ast.literal_eval(x)
        )
        self.df = self.df[self.df["annotations"].apply(lambda x: len(x)) > 0]
        self.df = self.df.explode(column="annotations")
        self.df = self.df[["annotations", "image_path"]]
        self.df["class_name"] = "starfish"
        self.df["x1"], self.df["y1"], self.df["x2"], self.df["y2"] = (
            self.df["annotations"],
            self.df["annotations"],
            self.df["annotations"],
            self.df["annotations"],
        )

        self.df.drop("annotations", axis=1, inplace=True)
        self.df["x1"] = self.df["x1"].apply(
            lambda dict_coordinates: dict_coordinates["x"]
        )
        self.df["y1"] = self.df["y1"].apply(
            lambda dict_coordinates: dict_coordinates["y"]
        )
        self.df["x2"] = self.df["x2"].apply(
            lambda dict_coordinates: dict_coordinates["x"] + dict_coordinates["width"]
        )
        self.df["y2"] = self.df["y2"].apply(
            lambda dict_coordinates: dict_coordinates["y"] + dict_coordinates["height"]
        )
        self.df = self.df[["image_path", "x1", "y1", "x2", "y2", "class_name"]]
        self.df.to_csv(os.path.join(faster_cnn_model_path, "annotate.txt"), index=False)

    def transfer_image(self)->None:
        """
        The goal of this function
        is to transfer all appropriate
        images to the faster cnn model
        to be able to launch the training
        
        Arguments:
            -None
        Returns:
            -None
        """

        annotation_path=os.path.join(faster_cnn_model_path, "annotate.txt")
        if not os.path.exists(annotation_path):
            raise ValueError("The annotation file was not found ! If you haven't already,\
                             run the dataframe preprocessing method first.")

        if not os.path.exists(os.path.join(faster_cnn_model_path,"train_images")):
            os.makedirs(os.path.join(faster_cnn_model_path,"train_images"))
            os.makedirs(os.path.join(faster_cnn_model_path,"train_images","video_0"))
            os.makedirs(os.path.join(faster_cnn_model_path,"train_images","video_1"))
            os.makedirs(os.path.join(faster_cnn_model_path,"train_images","video_2"))
            
            with open(annotation_path, 'r') as f:
                for line in tqdm(f):
                    line_split = line.strip().split(',')
                    filename = line_split[0]

                    if filename=="image_path":
                        continue
                    shutil.copy(filename,os.path.join(faster_cnn_model_path,filename))



def clean_all_files() -> None:
    """
    The goal of this funcion
    is to remove all txt annotation
    files

    Arguments:
        -None
    Returns:
        -None
    """
    to_delete = glob.glob("train_images/*/*.txt")
    for file_path in to_delete:
        os.remove(file_path)
