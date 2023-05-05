import unittest
import logging
import glob
import os
from Help_protect_great_barrier_reef.logs.logs import main
from Help_protect_great_barrier_reef.model.yolo_v5 import yolo_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

def get_all_files()->None:
    logging.info("Function parsed")
    if not os.path.exists("train_images/"):
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        logging.info("Copying all files")
        os.system("kaggle competitions download -c tensorflow-great-barrier-reef")
        logging.info("Extracting files...")
        os.system("unzip tensorflow-great-barrier-reef.zip")
        logging.info("Files succesfully unzipped...")

def copy_yolo_file()->None:
    if not os.path.exists("Help_protect_great_barrier_reef/model/yolov5_ws"):
            os.chdir("Help_protect_great_barrier_reef/model")
            os.mkdir("yolov5_ws")
            os.system("cd yolov5_ws")
            os.system("git clone https://github.com/ultralytics/yolov5")
            os.system("cd yolov5")
            os.system("pip install -r requirements.txt")
            os.chdir("../../../../../")

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    logging.info("Performing unitest")

    def test_yolo_repartition(self) -> None:
        """
        The goal of this function is to check
        if the yolo_v5 model fits well when 
        asked to

        Arguments:
            -None

        Returns:
            -None
        """
        get_all_files()
        copy_yolo_file()
        
        model=yolo_model()
        model.get_split()
        model.split_files()

        nb_images=len(glob.glob("train_images/*/*.jpg"))
        nb_test_images=len(glob.glob("Help_protect_great_barrier_reef/model/yolov5_ws/yolov5/test_set/*.jpg"))
        nb_train_images=len(glob.glob("Help_protect_great_barrier_reef/model/yolov5_ws/yolov5/train_set/*.jpg"))
        nb_valid_images=len(glob.glob("Help_protect_great_barrier_reef/model/yolov5_ws/yolov5/valid_set/*.jpg"))

        self.assertEqual(nb_images, nb_test_images+nb_train_images+nb_valid_images)
        
    def test_yolo_predict(self)->None:
        pass


if __name__ == "__main__":
    main()
    get_all_files()
    copy_yolo_file()
    unittest.main()