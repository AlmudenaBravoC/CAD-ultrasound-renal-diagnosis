# CAD Ultrasound Renal Diagnosis

## Abstract of the work 

In this project we propose the development of a renal ultrasound computer aided tool using deep learning techniques. We introduce three main tools as part of this system: a classifier that uses convolutional neural networks (CNNs) to distinguish between healthy and pathological kidneys, a segmentation tool substantiated on mask region based CNNs >>(mask-RCNNs) to locate the kidney, and a tool for estimating the area of the parenchyma, using another segmentation model and blob detection, while also returning the kidney size using blob detection to find a pixel-to-centimiter conversion. We demonstrate the effectiveness of these tools using efficientNets<< (REESCRIBIR) and ResNets for the classification task and performing visual evaluation of the segmentation and area estimation tools. Additionally we develop an application programming interface (API) for local usage and testing. Our results suggest that the proposed system has the potential to improve the efficiency of renal ultrasound diagnosis.

$\textbf{Keywords:}$ Artificial intelligence, deep learning, classification, segmentation, parenchyma, kidney

## PREPROCESSING

In our case we needed to convert all the information so it was easiser to read and use. Also, we needed to do some crops to the images and analysis to get the distance information to have the pixel-cm relation.
All the process has been included.

## CLASIFICATION

This model is based on the ResNet-50 pretained model of PyTorch.
To run it you just need to write in the terminal:

```ruby
python classification_model.py 
```
