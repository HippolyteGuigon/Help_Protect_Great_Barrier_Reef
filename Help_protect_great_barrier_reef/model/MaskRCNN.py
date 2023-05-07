#Ici, on veut implémenter un modèle MaskRCNN

import glob
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Help_protect_great_barrier_reef.logs.logs import main


class MaskRCNN:
    """
    The goal of this class
    is the implementation of
    the MaskRCNN model
    """

    def __init__(self)->None:
        pass

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

        all_images=glob.glob("train_images/*/*.jpg")

        train_set=np.random.choice(all_images, 
                                   size=np.floor(0.9*len(all_images)), replace=False)
        
        val_set=[image_path for image_path in all_images if image_path not in train_set]


if __name__ == '__main__':
    main()
    test=MaskRCNN()
    test.split_images()
