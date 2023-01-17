# CAD Ultrasound Renal Diagnosis

Project in collaboration with the hospital Ramon y Cajal of Madrid

## Abstract of the work 

n this project we propose the development of a renal ultrasound computer aided tool using deep learning techniques. We introduce three main tools as part of this system: a classifier that uses convolutional neural networks (CNNs) to distinguish between healthy and pathological kidneys, a segmentation tool substantiated on semantic segmentation based CNNs (deeplabv3) to segmentate the kidney and its parenchyma, and lastly a tool for estimating the area of the parenchyma and the kidney size using blob detection with a pixel-to-centimeter conversion. We demonstrate the effectiveness of these tools using ResNets for the classification task and performing visual evaluation of the segmentation and area estimation tools. Additionally we develop an application programming interface (API) for local usage and testing. Our results suggest that the proposed system has the potential to improve the efficiency of renal ultrasound diagnosis.

$\textbf{Keywords:}$ Artificial intelligence, deep learning, classification, segmentation, parenchyma, kidney

## PREPROCESSING

Before starting, you will need to download and install the environment in conda.

In our case we needed to convert all the information so it was easiser to read and use. Also, we needed to do some crops to the images and analysis to get the distance information to have the pixel-cm relation.
All the process has been included.

## CLASIFICATION

This model is based on the ResNet-50 pretained model of PyTorch.
To run an example:

```ruby
python Classification/classification_model.py --n_epochs=30 --m=0.95 --root_img='cropped_images'
```
Some other values can be changed, for more check -help


## SEGMENTATION

This model is based on the Deeplabv3-ResNet50 pretained model of PyTorch.
To run an example:

```ruby
python Segmentation/segmentation_model.py --n_epochs=30 --root_fold='normal_crop' --fold_img='good_img' --fold_masks='mask_parenquima' --batch=8
```
Some other values can be changed, for more check -help

## TestImage

This is a folder that contains a file with all the functions necessary and a example that uses the different functions with our models and a test image. 

All is included, the preprocessing necessary for the image, the calculation of the number of pixels that represent one centimeter and the prediction using the different models. Moreover, the functions to calculate the areas, the axis and the parenchyma thickness are also included.


Images are not available
