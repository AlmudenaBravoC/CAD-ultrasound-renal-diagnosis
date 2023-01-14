## CLASSIFICATION MODEL USING RESNET FROM PYTHON

# %% Libraries
from torchvision import models
from torchmetrics import AUROC
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from skimage import io, util
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from multiprocessing import Pool
import multiprocessing as mp
import time
import os
from collections import Counter
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc



# %% Plot function (acc+auc)
def plot_acc(t, v, at, av, tit="Train-Validation Accuracy",name: str='', saveFig=False):
  """
  Plot the evolution of the ACC and AUC of the validation and train
  
  t: train acc values 
  v: validation acc values
  at: auc values of the train
  av: auc values of the validation
  tit: title of the plot
  
  name: name for the tile in case we want to save it
  saveFig: Boolean parameter if we want to save the plot or not
  """
  fig = plt.figure(figsize=(10,4))
  plt.title(tit)
  plt.plot(t, label='ACC train')
  plt.plot(v, label='ACC validation')
  plt.plot(at, linestyle='--', label='AUC train')
  plt.plot(av, linestyle='--', label='AUC validation')
  plt.xlabel('num_epochs', fontsize=12)
  plt.ylabel('accuracy', fontsize=12)
  plt.legend(loc='best')
  plt.show()

  if saveFig:
    plt.savefig(f'{name}_acc.png')


# %% Plot function (loss)
def plot_loss(loss_t, loss_v, tit="Loss", name:str ='', saveFig=False):
  """
  Plot the evolution of the loss
  
  loss_t: loss values during train
  loss_v: loss value during validation
  tit: title of the plot
  
  name: name for the tile in case we want to save it
  saveFig: Boolean parameter if we want to save the plot or not
  """
  fig = plt.figure(figsize=(10,4))
  plt.title(tit)
  plt.plot(loss_t, label='train')
  plt.plot(loss_v, label='validation')
  plt.xlabel('num_epochs', fontsize=12)
  plt.ylabel('loss', fontsize=12)
  plt.legend(loc='best')
  plt.show()

  if saveFig:
    plt.savefig(f'{name}_loss.png')

# %% Confusion matrix

def print_cm(pred, target, tit='Train',name: str='', saveFig=False):
  """
  Plot confusion matrix of validation
  
  pred: predicted values
  target: target values
  tit: title of the plot
  
  name: name for the tile in case we want to save it
  saveFig: Boolean parameter if we want to save the plot or not
  """
  cf_matrix = confusion_matrix(target, pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)

  disp.plot()
  plt.title(tit)
  plt.show()

  if saveFig:
    plt.savefig(f'{name}_cm.png')



# %% Probability to pred

# from probability to class value
def probs_to_prediction(probs, threshold):
    pred=[]
    for x in probs[:,1]: #check the probabilities of the class 1
        if x>threshold: #[0.3, 0.7] --> 0.7 > 0.6 (th)
            pred.append(1) #pathological
        else:
            pred.append(0) #health
    return pred

# %% Train-test function
def trainloop(model, name,
                  trans_tensor, weights_loss = list() , root_img = 'cropped_images', proportion_train = 0.8, batch_size = 16, 
                  n_epochs=5, print_every=20, train_m = True, learning_rate = 1e-05, threshold = 0.5, momentum = 0.9):
  """
    model: model to be used
    name: name of the model, for the title of the final plot

      FOR DATALOADERS
    root_img: the root of the folder where the images are
    trans_tensor: the tensor with the transformations
    proportion_train: the proportion we want to have the train and velidation
    batch_size: number for the batch 

      FOR THE MODEL
    n_epochs: number of epochs to run the cnn
    print_every: how often print the loss of the model 
    train_m: if we want to train the model or just use the test set to check the loss and accuracy
    
      OTHER VALUES
    weights_loss: weights (if any) for the loss function
    learning_rate: learning rate hyperparameter 
    threshold: for the prediction of the label
    momentum: for the optimazer
  """

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  ##CREATING DATALOADER_______________________________________________________
  print('Creating dataloaders...')
  orig_set = datasets.ImageFolder(root=root_img, transform=trans_tensor)

  train_size = int(proportion_train * len(orig_set))
  test_size = len(orig_set) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(orig_set, [train_size, test_size])
  train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


  ##CRITERION AND OPTIMIZER____________________________________________________
  weights = torch.tensor(weights_loss)
  weights = weights.to(device)
  criterion = torch.nn.CrossEntropyLoss(weight = weights)
  optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)


  ##MODEL______________________________________________________________________
  
  model = model.to(device)
  auroc = AUROC(task="binary")

  valid_loss_min = np.Inf
  best_acc = None; best_auc= None
  val_loss = []
  val_acc = []
  train_loss = []
  train_acc = []
    #aucs
  val_auc= []
  train_auc =[]
  total_step = len(train_set)

  prev_loss = 0
  alpha = 0.1
  epoch = 0
  has_learned = False


  if not train_m:
    n_epochs = 1
    print('Computing only test...')

  while has_learned == False:
      ## TRAIN ___________________________________
      if train_m:
        print('Training...')
        running_loss = 0.0
        correct = 0
        total=0
        prob_tr = torch.tensor([], device = device)

          #for the confusion matrix
        pred_cf = []
        target_cf = torch.tensor([], device = device)

        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_set):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            ## PREDICTION
            prob = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.tensor(probs_to_prediction(prob, threshold)).to(device)

            prob_tr = torch.cat((prob_tr, prob[:,1]))
            target_cf = torch.cat((target_cf, target_))
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)

            if (batch_idx) % print_every == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        acc = 100 * (correct / total)
        auc = auroc(prob_tr, target_cf) * 100
        train_acc.append(acc)
        train_auc.append(auc.item())
        train_loss.append(running_loss/total_step)

        #plot confusion Matrix
        # print_cm( sum(pred_cf, []), sum(target_cf, []))
      
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(acc):.4f}, train-auc: {(train_auc[-1]):.4f}')
      
      ## EVALUATION _______________________________
      batch_loss = 0
      total_t = 0
      correct_t = 0
      auc_v = []
      outputs_cf_t = torch.tensor([], device = device)
      targets_cf_t = torch.tensor([], device = device)
      prob_tst = torch.tensor([], device = device)
      with torch.no_grad():
          model.eval()
          for data_t, target_t in (test_set):
              data_t, target_t = data_t.to(device), target_t.to(device)
              outputs_t = model(data_t)
              loss_t = criterion(outputs_t, target_t)
              batch_loss += loss_t.item()

              ## PREDICTIONS
              prob_t = torch.nn.functional.softmax(outputs_t, dim=1)
              pred_t = torch.tensor(probs_to_prediction(prob_t, threshold)).to(device)

              prob_tst = torch.cat((prob_tst, prob_t[:,1]))

              outputs_cf_t = torch.cat((outputs_cf_t, pred_t))
              targets_cf_t = torch.cat((targets_cf_t, target_t))
              correct_t += torch.sum(pred_t==target_t).item()
              total_t += target_t.size(0)

          acc_t = 100 * (correct_t / total_t)
          auc_t = auroc(prob_tst, targets_cf_t)*100
          val_acc.append(acc_t)
          val_loss.append(batch_loss/len(test_set))
          val_auc.append(auc_t.item())
          network_learned = batch_loss < valid_loss_min

          print(f'validation-loss: {np.mean(val_loss):.4f}, validation-acc: {(acc_t):.4f}, validation-auc: {(val_auc[-1]):.4f}\n')
          
          if network_learned and train_m:
              #plot confusion Matrix
              print_cm( outputs_cf_t.cpu().tolist(), targets_cf_t.cpu().tolist(), tit='Validation')

              valid_loss_min = batch_loss
              best_acc = acc_t
              best_accTrain = acc
              best_auc = val_auc[-1]
              
              name_model = f'{name}_lr{learning_rate}_th{threshold}_m{momentum}_batch{batch_size}'
              torch.save(model.state_dict(), f'{name_model}.pt') #save the model
              print('Improvement-Detected, save-model')
              print_cm( outputs_cf_t.cpu().tolist(), targets_cf_t.cpu().tolist(), tit='Validation', name=name, saveFig=True)e
      
      ## Check if it has lerned
      if epoch >= n_epochs:
          if batch_loss - prev_loss <= alpha:
            has_learned = True
      prev_loss = batch_loss
      
      model.train()
      epoch += 1

  #plot fucntions
  plot_loss(train_loss, val_loss, tit=f"{name}_Loss", name=name, saveFig = sf)
  plot_acc(train_acc, val_acc, train_auc, val_auc, tit=f"{name}_Bacth{batch_size}: acc {round(best_acc,2)}", name=name, saveFig = sf)
  


  return [valid_loss_min, best_accTrain, best_acc, best_auc]




######################################################################################################################################################
# %% WEIGHTS and transform tensors

torch.manual_seed(6)

### CALCULATING WEIGHTS
health = 450
path = 1535

#weights Opt2
n_samples = health+path
n_classes =2
w1 = n_samples / (n_classes * health)
w2 = n_samples / (n_classes *path)



##TRANSFORMS
pixel_mean=[0.485, 0.456, 0.406]
pixel_std=[0.229, 0.224, 0.225]

t_tensor = transforms.Compose([transforms.ColorJitter(0.3, 0.3, 0.3),
                               transforms.RandomHorizontalFlip(p=0.7),
                               transforms.GaussianBlur(np.random.choice([3,5,7]),
                                sigma = (0.5,2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=pixel_mean,std=pixel_std)])



t_tensor_simple = transforms.ToTensor()


# %% ResNet 50

##### DEFINE PARAMETERS
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--n_epochs', type=float, default=100.0,
                    help='number of epochs')
parser.add_argument('--m', type=float, default=0.9,
                    help='momentum for the optimizer')
parser.add_argument('--th', type=float, default=0.6,
                    help='threshold for the prediction of the class')
parser.add_argument('--batch', type=int, default=16,
                    help='bacth size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout')
parser.add_argument('--root_img', type=str, default='cropped_images',
                    help='root of where the images are save')
parser.add_argument('--pt', type=float, default=0.8,
                    help='proportion train - validation')
args = parser.parse_args()
print(args)


model_resnet50 = models.resnet101(pretrained=True)
model_resnet50.fc = torch.nn.Sequential(torch.nn.Dropout(args.dropout),
                              torch.nn.Linear(in_features=model_resnet50.fc.in_features, out_features=2, bias=True))

start_time = time.time()
name = 'ResNet101_transformations'
vl , acc_tr, acc_v, auc = trainloop(model_resnet50, name = name,
                  root_img = args.root_img,
                  weights_loss = [w1, w2] , trans_tensor = t_tensor, proportion_train = args.pt, batch_size = args.batch, 
                  n_epochs=args.n_epochs, print_every=50, train_m = True, 
                  learning_rate = args.lr, threshold = args.th, momentum = arsg.m)
t = time.time() - start_time

print(f'\n{name} -------- ACC:{acc_v} /// AUC: {auc}')


torch.cuda.empty_cache()
