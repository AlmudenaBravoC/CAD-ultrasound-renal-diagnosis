
# A TEST WITH A HEALTHY IMAGE


#%% LIBRARIES

from skimage import io
import matplotlib.pyplot as plt
from torchvision import models
import torch
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

## our functions
import allProcess


#%% read image
file = 'name.jpg'
root_model = 'ResNet50.3_lr0.001_th0.6_m0.95_batch16.pt'
root_model_paren = 'weights_parenquima2.pt'

# %% Caluclate distances

initial_img = io.imread(file)
points_img = allProcess.crop_image(initial_img)
points = allProcess.find_points(points_img)
px_cm = allProcess.get_distance_x(points)


# %% Preprocess the image

crop_img = allProcess.preprocess_img(file)

# %% CLASSIFICATION
model_class = models.resnet50(pretrained=True)
model_class.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),
                              torch.nn.Linear(in_features=model_class.fc.in_features, out_features=2, bias=True))

model_class.load_state_dict(torch.load(root_model))

pred_class = allProcess.predictionRESNET(model_class, crop_img, threshold=0.6)
print(pred_class) #predicted correctly as healthy

# %% SEGMENTATION PARENCHYMA

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.classifier = DeepLabHead(2048, 1)
model.load_state_dict(torch.load(root_model_paren))
model.eval()

mask = allProcess.predictionDEEPLABV(model, crop_img, threshold=0.6, save=True)

## get number of pixels (=1)
pixels_paren = np.sum(mask==1)
print(pixels_paren/px_cm) #parenchyma area --> 882.59 cm2
