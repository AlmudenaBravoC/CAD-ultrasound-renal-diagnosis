### FUNCTIONS

import skimage
import numpy as np

from math import sqrt
from skimage.feature import blob_dog, blob_log

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

### 1 GET DISTANCE
def crop_x(image, x1=100,x2=250, y1=38,y2=48):
  """
  Return the cropped image at the x1, x2, y1, y2 coordinates
  """
  return image[y1:y2 , x1:x2, :]
  
def crop_x2D(image, x1=100,x2=250, y1=38,y2=48):
  """
  Return the cropped image at the x1, x2, y1, y2 coordinates
  """
  return image[y1:y2 , x1:x2]

def find_points(image, laplacian=True):
	image_gray = skimage.color.rgb2gray(image)
	if laplacian == True:
		blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
		blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
		return blobs_log

	else:
		blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
		blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
		return blobs_dog

def crop_image(img):
  if len(np.shape(img)) == 3:
    img1 = crop_x(img,100,300,0,60)
  else:
    img1 = crop_x2D(img,100,300,0,60)
  points = find_points(img1,laplacian=False)
  for p in points:
    if p[2] <1.5:
      x = int(p[1])
      y = int(p[0])
      break
  try:
    if y <3:
      y=3
  except:
    img = second_crop(img)
    return img
  if x<49:
    x=50
  img = crop_x(img, x1=(x-10),x2=(x+130),y1=(y-2),y2=(y+2))
  return img


def second_crop(img):
  if len(np.shape(img)) == 3:
    img1 = crop_x(img,100,300,0,60)
  else:
    img1 = crop_x2D(img,100,300,0,60)
  points = find_points(img1,laplacian=True)
  for p in points:
    if p[2] <1.5:
      x = int(p[1])
      y = int(p[0])
      break
  try:
    if y <3:
      y=3
  except UnboundLocalError:
    img = crop_x(img,y1=0)
    return img
  if x<49:
    x=50
  img = crop_x(img, x1=(x-10),x2=(x+130),y1=(y-2),y2=(y+2))
  return img

def get_distance_x(points):
  all =[]
  for p in points:
    if p[2] < 1.5:
      all.append(p[1])
  all =np.sort(all)
  distances = [abs(all[i] - all[i+1]) for i in range(len(all)-1)]
  return np.mean(distances)


### 2 PREPROCESSING

def preprocess_img(file):
  """
  Crop the images to 375x375 without considering where is the kidney.
  It read the image file and the matlab files where the masks where saved
  
  features:
  - file: list of name files
  - s-lesions: dict with the indx and the sum of the lesions each kidney has
  - poly: list of names with the poly mask
  """

  im = Image.open(file) #IMAGE

  im_dim= np.shape(im)[:2] #we get the dimension of the image (and the masks)
  min_s = 375
  
  #getting the number of pixels we need to remove
  h = im_dim[0] - min_s
  w = im_dim[1] - min_s

  left = int(w/2); right = min_s+int(w/2)
  top = int(h/4); bottom = min_s+int(h/4)

 
  ######################################## CROP AND RETURN ####################################################
  #IMAGE
  im2 = im.crop((left, top, right, bottom))

  return im2


### 3 CLASIFICATION
def probs_to_prediction(probs, threshold):
    pred=[]
    for x in probs[:,1]: #check the probabilities of the class 1
        if x>threshold: #[0.3, 0.7] --> 0.7 > 0.6 (th)
            pred.append(1) #pathological
        else:
            pred.append(0) #health
    return pred

def predictionRESNET(model, image_test, threshold = 0.5):
  """
    model: model to be used
  """
  image_test = preprocess_for_model(image_test)

  with torch.no_grad():
    model.eval()
    image_test = image_test.reshape((1,3,375,375))
    outputs_t = model(image_test)

    ## PREDICTIONS
    prob_t = torch.nn.functional.softmax(outputs_t, dim=1)
    pred_t = probs_to_prediction(prob_t, threshold)
         
    return pred_t[0]


def preprocess_for_model(img, type_model = 1):
    """
    type_model = 1 (classification) 2 (segmentation)
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])
    if type_model == 2:
        # new_img = np.moveaxis(new_img, 2,0)
        # return DataLoader(torch.tensor(new_img)).dataset[0]
        return data_transforms(img)
    else:
      new_img = np.array(img)
      return torch.tensor(new_img, dtype=torch.float)

### 4 SEGMENTATION

def predictionDEEPLABV(model, image_test, threshold = 0.6, save=False):
  """
    model: model to be used
  """
  image_test = preprocess_for_model(image_test, 2)
  
  with torch.no_grad():
    model.eval()
    mask = model(image_test.unsqueeze_(0))['out']
    mask = mask.detach().numpy()
    new_mask = (mask>= threshold) *1

    if save:
      plt.imshow(new_mask.reshape((375, 375,1)))
      plt.savefig('test.jpg')
         
    return new_mask
