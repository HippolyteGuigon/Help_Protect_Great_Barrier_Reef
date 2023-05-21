import numpy as np
import glob
import os
import pandas as pd
import ast
from tqdm import tqdm
from PIL import Image
from math import sqrt
from keras.optimizers import Adam, SGD, Adamax
from keras.initializers import RandomNormal
from numpy import zeros, ones, expand_dims, asarray
from numpy.random import randn, randint
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import LeakyReLU, Dropout, Embedding
from keras.layers import BatchNormalization, Activation
from keras import initializers
from matplotlib import pyplot
from Help_protect_great_barrier_reef.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
nb_epochs=main_params["nb_epochs"]
batch_size=main_params["batch_size"]

class GAN_model:
    """
    The goal of this class is the
    implementation of a generative
    adversarial network (GAN) to perform
    data augmentation

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self) -> None:
        self.X_train = []
        self.df = pd.read_csv("train.csv")

    def convert_images(self, only_starfish: bool = True, reload: bool = False) -> None:
        """
        The goal of this function
        is converting all images to
        proper npy format in order
        to use the GAN model

        Arguments:
            -only_starfish: bool: Whether
            or not only images with starfish
            should be considered
            -reload: bool: Whether images should
            be reloaded in the folder
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

        if not os.path.exists("X_train.npy") or reload:
            for image in tqdm(images_path):
                img = Image.open(image)
                img = img.resize((200, 200))
                numpydata = asarray(img)
                self.X_train.append(numpydata)
            self.X_train = np.array(self.X_train)
            np.save("X_train.npy", self.X_train)

        else:
            self.X_train = np.load("X_train.npy")

    def generate_latent_points(self, latent_dim, n_samples):
        """
        The goal of this function
        is to generate a certain
        number of random points that
        will then be used to generate
        fake images

        Arguments:
            -latent_dim: int: The dimensions
            in which the final images will
            be
            -n_samples: int: The number of fake
            images to generate

        Returns:
            -z_input: np.array: The random points
            generated in the correct dimensions
        """

        x_input = randn(latent_dim * n_samples)
        z_input = x_input.reshape(n_samples, latent_dim)
        return z_input

    def generate_real_samples(self, X_train, n_samples):
        ix = randint(0, X_train.shape[0], n_samples)
        X = X_train[ix]
        y = ones((n_samples, 1))
        return X, y

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        z_input = self.generate_latent_points(latent_dim, n_samples)
        images = generator.predict(z_input)
        y = zeros((n_samples, 1))
        return images, y

    def summarize_performance(self, step, g_model, latent_dim, n_samples=100):
        X, _ = self.generate_fake_samples(g_model, latent_dim, n_samples)
        X = (X + 1) / 2.0
        for i in range(100):
            pyplot.subplot(10, 10, 1 + i)
            pyplot.axis("off")
            pyplot.imshow(X[i, :, :, 0], cmap="gray_r")
        last_model = glob.glob("*.h5")
        if len(last_model) > 0:
            os.remove(last_model[0])
        filename2 = "model_%04d.h5" % (step + 1)
        g_model.save(filename2)
        print(">Saved: %s" % (filename2))

    def define_discriminator(self, in_shape=(200, 200, 3)):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=in_shape)
        fe = Flatten()(in_image)
        fe = Dense(1024)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        fe = Dense(512)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        fe = Dense(256)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.3)(fe)
        out = Dense(1, activation="sigmoid")(fe)
        model = Model(in_image, out)
        opt = Adamax(lr=0.02)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def define_generator(self, latent_dim):
        init = RandomNormal(stddev=0.02)
        in_lat = Input(shape=(latent_dim,))
        gen = Dense(256, kernel_initializer=init)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(512, kernel_initializer=init)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(1024, kernel_initializer=init)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(200 * 200 * 3, kernel_initializer=init)(gen)
        out_layer = Activation("tanh")(gen)
        out_layer = Reshape((200, 200, 3))(gen)
        model = Model(in_lat, out_layer)
        return model

    def define_gan(self, g_model, d_model):
        d_model.trainable = False
        gan_output = d_model(g_model.output)
        model = Model(g_model.input, gan_output)
        opt = Adamax(lr=0.02)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def train(
        self, g_model, d_model, gan_model, X_train, latent_dim, n_epochs=nb_epochs, n_batch=64
    ):
        bat_per_epo = int(X_train.shape[0] / n_batch)
        n_steps = bat_per_epo * n_epochs
        for i in range(n_steps):
            X_real, y_real = self.generate_real_samples(X_train, n_batch)
            d_loss_r, d_acc_r = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, n_batch)
            d_loss_f, d_acc_f = d_model.train_on_batch(X_fake, y_fake)
            z_input = self.generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss, g_acc = gan_model.train_on_batch(z_input, y_gan)
            print(
                ">%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]"
                % (i + 1, d_loss_r, d_acc_r, d_loss_f, d_acc_f, g_loss, g_acc)
            )
            if (i + 1) % (bat_per_epo * 1) == 0:
                self.summarize_performance(i, g_model, latent_dim)

    def save_plot(self, examples, n_examples):
        for i in range(n_examples):
            pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
            pyplot.axis("off")
            pyplot.imshow(examples[i, :, :, 0], cmap="gray_r")
        pyplot.show()


if __name__ == "__main__":
    test = GAN_model()
    test.convert_images(only_starfish=False)
    discriminator = test.define_discriminator()
    latent_dim = 100
    generator = test.define_generator(latent_dim)
    gan_model = test.define_gan(generator, discriminator)
    test.train(
        generator,
        discriminator,
        gan_model,
        test.X_train,
        latent_dim,
        n_epochs=nb_epochs,
        n_batch=batch_size,
    )
    last_model = glob.glob("*.h5")[0]
    model = load_model(last_model)
    n_examples = 9
    latent_points = test.generate_latent_points(latent_dim, n_examples)
    X = model.predict(latent_points)
    X = (X + 1) / 2
    test.save_plot(X, n_examples)
