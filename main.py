#Ici, on veut lancer l'intégralité du Pipeline, 
#C'est à dire à la fois le fit ainsi que le predict

import os
import argparse
import logging
import torch
from Help_protect_great_barrier_reef.model.yolo_v5 import yolo_model
from Help_protect_great_barrier_reef.configs.confs import load_conf, clean_params, Loader
from Help_protect_great_barrier_reef.logs.logs import main

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
all_fitted_model_path=main_params["all_fitted_model_path"]

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="Yolo", help="Model chosen")
parser.add_argument("--fitting", type=str,choices=['fit', 'load'], default="fit", help="Wheter the model should\
                     be trained or loaded")
parser.add_argument("--train_size", type=float, default=0.8, help="Size of the train\
                    set for the model fitting")

args=parser.parse_args()

if args.model=="Yolo":
    logging.info("You have chosen the Yolo model")
    model=yolo_model()

if args.fitting=="fit":
    logging.info("You have chosen to fit the model")
    model.get_split(train_size=args.train_size)
    model.split_files()
    model.fit()

elif args.fitting=="load":
    logging.info("You have chosen to load the model")
    all_models=os.listdir(all_fitted_model_path)
    all_models=[path for path in all_models if path.startswith("exp")]
    all_models=sorted(all_models, key=lambda x: int(x[len("exp"):]))
    last_model=all_models[-1]
    last_model_path=os.path.join(all_fitted_model_path,last_model,"weights/best.pt")

    model = torch.hub.load(
                "ultralytics/yolov5", "custom", last_model_path
                )
    logging.info("Model successfully loaded !")