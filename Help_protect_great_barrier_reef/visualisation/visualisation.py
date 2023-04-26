import pandas as pd 
import ast
import os
from PIL import Image, ImageDraw

df_train=pd.read_csv("train.csv")

def visualise(path: str)->Image:
    """
    The goal of this function
    is the visualisation of the
    image with annotated position 
    of the starfish position if 
    there is any
    
    Arguments:
        -path: The path to the 
        image to display
        
    Returns:
        -Image: The visualisation
        desired
    """
    
    if not os.path.exists(path):
        raise AssertionError("The path you entered does not exist, \
                             check all existing path by going into train_images file")
    
    image = Image.open(path)

    splitted_path=path.split("/")
    video_number, image_number = splitted_path[-2].replace("video_",""),\
          splitted_path[-1].replace(".jpg","")
    image_id=video_number+"-"+image_number
    annotation=df_train[df_train.image_id==image_id]["annotations"].values[0]
    
    if annotation=="[]":
        pass
    else:
        image_annotation=ast.literal_eval(annotation)
        
        for dict_coordinate in image_annotation:
            x0, y0, x1, y1 = dict_coordinate["x"], dict_coordinate["y"],\
            dict_coordinate["x"]+dict_coordinate["width"],\
                dict_coordinate["y"]+ dict_coordinate["height"]
            draw = ImageDraw.Draw(image)
            draw.rectangle((x0, y0, x1, y1), outline="red", width=3)

    image.show()

