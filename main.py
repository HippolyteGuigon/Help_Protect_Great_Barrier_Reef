import os
import argparse
import logging
import torch
import glob
from Help_protect_great_barrier_reef.model.yolo_v5 import (
    yolo_model,
    get_last_model_path,
)
from Help_protect_great_barrier_reef.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)
from PIL import Image
from keras.models import load_model
from Help_protect_great_barrier_reef.logs.logs import main
from Help_protect_great_barrier_reef.data_augmentation.gan_model import GAN_model

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
all_fitted_model_path = main_params["all_fitted_model_path"]
latent_dim=main_params["latent_dim"]
nb_epochs=main_params["nb_epochs"]
batch_size=main_params["batch_size"]
nb_image_to_generate=main_params["nb_image_to_generate"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, default="Yolo", help="Model chosen for the training"
)
parser.add_argument(
    "--fitting",
    type=str,
    choices=["fit", "load"],
    default="fit",
    help="Wheter the model should\
                     be trained or loaded",
)
parser.add_argument(
    "--data_augmentation",
    type=str,
    default="no",
    choices=["yes", "no"],
    help="Whether or not data augmentation with a GAN should be used",
)
parser.add_argument(
    "--train_size",
    type=float,
    default=0.8,
    help="Size of the train\
                    set for the model fitting",
)

args = parser.parse_args()

if args.model == "Yolo":
    logging.info("You have chosen the Yolo model")
    model = yolo_model()

if args.data_augmentation == "yes":
    logging.info("You have chosen the data augmentation with a GAN")
    gan=GAN_model()
    gan.convert_images(only_starfish=False)
    discriminator = gan.define_discriminator()
    generator = gan.define_generator(latent_dim)
    gan_model = gan.define_gan(generator, discriminator)
    gan.train(
        generator,
        discriminator,
        gan_model,
        gan.X_train,
        latent_dim,
        n_epochs=nb_epochs,
        n_batch=batch_size,
    )
    last_model = glob.glob("*.h5")[0]
    model = load_model(last_model)
    n_examples = 9
    latent_points = gan.generate_latent_points(latent_dim, n_examples)
    X = model.predict(latent_points)

    for image_number, image in enumerate(X):
        image_array=Image.fromarray(image)
        image_path=os.path.join(
            "train_images/video_0",
            str(image_number*100000)+".jpg")
        image_array.save(image_path)
        file = open(image_path.replace(".jpg",".txt"), 'w')
        file.close()

if args.fitting == "fit":
    logging.info("You have chosen to fit the model")
    model.get_split(train_size=args.train_size)
    model.split_files()
    model.fit()
    model.model_loading()

elif args.fitting == "load":
    logging.info("You have chosen to load the model")

    last_model_path = get_last_model_path()

    model = torch.hub.load("ultralytics/yolov5", "custom", last_model_path)

    logging.info("Model successfully loaded !")
