#Ici, on veut implémenter un modèle MaskRCNN

import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
