import numpy as np
import glob
import os
import pandas as pd
import ast
from tqdm import tqdm
from PIL import Image
from math import sqrt
from numpy import zeros, ones, expand_dims, asarray
from numpy.random import randn, randint
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import LeakyReLU, Dropout, Embedding
from keras.layers import BatchNormalization, Activation
from keras import initializers
from matplotlib import pyplot


class GAN_model:
    def __init__(self) -> None:
        self.X_train = []
        self.df = pd.read_csv("train.csv")

    def convert_images(self, only_starfish: bool = True) -> None:
        """
        The goal of this function
        is converting all images to
        proper npy format in order
        to use the GAN model

        Arguments:
            -only_starfish: bool: Whether
            or not only images with starfish
            should be considered

        Returns:
            -None
        """

        if only_starfish:
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
            images_path = self.df["image_path"]
        else:
            images_path = glob.glob("train_images/*/*.jpg")

        if not os.path.exists("X_train.npy"):
            for image in tqdm(images_path):
                img = Image.open(image)
                numpydata = asarray(img)
                self.X_train.append(numpydata)
            self.X_train = np.array(self.X_train)
            np.save("X_train.npy", self.X_train)

        else:
            self.X_train = np.load("X_train.npy")


if __name__ == "__main__":
    test = GAN_model()
    test.convert_images()
    print(test.X_train.shape)
    print(test.X_train)
