import unittest
import logging
import glob
import os
import numpy as np
from Help_protect_great_barrier_reef.logs.logs import main
from Help_protect_great_barrier_reef.model.yolo_v5 import (
    yolo_model,
    get_last_model_path,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


def copy_yolo_file() -> None:
    if not os.path.exists("Help_protect_great_barrier_reef/model/yolov5_ws"):
        os.chdir("Help_protect_great_barrier_reef/model")
        os.mkdir("yolov5_ws")
        os.system("cd yolov5_ws")
        os.system("git clone https://github.com/ultralytics/yolov5")
        os.system("cd yolov5")
        os.system("pip install -r requirements.txt")
        os.chdir("../../")


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    logging.info("Performing unitest...")

    def test_yolo_repartition(self) -> None:
        """
        The goal of this function is to check
        if the model preprocessing works well
        when allocating the images and annotations
        between the different sets

        Arguments:
            -None

        Returns:
            -None
        """

        copy_yolo_file()

        model = yolo_model(preprocessing=False)
        model.get_split()
        model.split_files()

        nb_images = len(glob.glob("train_images/*/*.jpg"))
        nb_annotation = len(glob.glob("train_images/*/*.txt"))

        nb_test_images = len(glob.glob(model.split_path + "/test_set/*.jpg"))
        nb_train_images = len(glob.glob(model.split_path + "/train_set/*.jpg"))
        nb_valid_images = len(glob.glob(model.split_path + "/valid_set/*.jpg"))

        nb_test_annotation = len(glob.glob(model.split_path + "/test_set/*.txt"))
        nb_train_annotation = len(glob.glob(model.split_path + "/train_set/*.txt"))
        nb_valid_annotation = len(glob.glob(model.split_path + "/valid_set/*.txt"))

        self.assertEqual(nb_images, nb_test_images + nb_train_images + nb_valid_images)
        self.assertEqual(
            nb_annotation,
            nb_test_annotation + nb_train_annotation + nb_valid_annotation,
        )

    def test_yolo_fit(self) -> None:
        """
        The goal of this function
        is to check if the model is well
        fitted when called

        Arguments:
            -None
        Returns:
            -None
        """

        copy_yolo_file()
        model = yolo_model(preprocessing=False)
        model.get_split(train_size=0.5)
        model.split_files()

        try:
            model.fit(nb_epochs=1)
        except:
            raise Exception("Fitting of the model has failed")

    def test_main(self) -> None:

        """
         The goal of this
         test is to check wheter
         the main file works

         Arguments:
            -None

        Returns:
            -None
        """
        try:
            os.system("python3 main.py")
        except:
            raise Exception("main file pipeline failed")


if __name__ == "__main__":
    main()
    unittest.main()
