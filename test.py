import os
# import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# import project_utils as utils
# import settings
# import streamlink
# import main as main
from main import LaneSegmentationModel
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import time
# import tensorflow as tf
# import core.utils as utilsYolo
# from core.yolov3 import YOLOv3, decode
# from PIL import Image

torch.cuda.is_available()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def overlay_image(image, overlay):
    # set the transparency level for overlay
    alpha = 0.3
    
    zeros = np.uint8(np.zeros_like(overlay))

    # convert the output to 3 channels
    overlay = np.uint8(overlay)
    overlay = np.dstack((overlay, overlay, zeros))

    # apply the overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image
    

def main(source_path=None):

    # load in video capture for source
    if source_path is None:
        vid = cv2.VideoCapture("videos/city15.mov")
    else:
        vid = cv2.VideoCapture(source_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    MODEL_PATH = "models/v1/newmodel2.pth"
    
    print(f"Using device: {DEVICE}")

    # load in model
    model = LaneSegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # loop to process each frame of video
    while vid.isOpened():
        _, im1 = vid.read()

        # im1 = cv2.rotate(im1, cv2.cv2.ROTATE_90_CLOCKWISE)
        im1 = cv2.rotate(im1, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # convert to PIL image 
        im1PIL = Image.fromarray(im1)
        
        image = transform(im1PIL)
        
        output = model(image.unsqueeze(0).to(DEVICE))
        output = torch.sigmoid(output)
        # output = (output > 0.7).float()
        # print(output)
        output = output.squeeze(0).detach().cpu().numpy()
        
        output_new = np.zeros_like(output)
        output_new[output > 0.7] = 255
        
        # overlay output on original image
        # processed_image = overlay_image(im1, output.detach().numpy())
        # cv2.imshow("results", output_new)
        
        # processed_image = output
        
        # # resize im1 to match output
        # im1 = cv2.resize(im1, (output.shape[1], output.shape[0]))
        
        # resize output to match im1
        output_new = cv2.resize(output_new, (im1.shape[1], im1.shape[0]))
        
        # print(im1)
        
        # output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        
        # convert im1 and output to np arrays
        # im1 = np.array(im1)
        # output = np.array(output)
        
        # processed_image = cv2.addWeighted(output,0.5,im1,0.7,0)
        processed_image = overlay_image(im1, output_new)
        
        # print(im1.shape)

        

        # display final results
        cv2.imshow("results", processed_image)

        # close window on q button
        if cv2.waitKey(1) & 0xFF == ord('q'):      
            break

    # close the video file
    vid.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()