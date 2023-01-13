
#%% Import libraries + functions
from torchvision import models
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import time
import os
from sklearn.metrics import f1_score, roc_auc_score
import time


from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import transforms

import copy
from tqdm import tqdm
import gc

from metrics import mean_iou, IoU, wpintersect

#% DATALOADER FUNCTIONS
def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'Images',
                                 mask_folder: str = 'Masks',
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    """
    based on : [link](https://github.com/msminhas93/DeepLabv3FineTuning)
    
    Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()]) #meter aqui las transfromaciones que veamos apropiadas

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders


class SegmentationDataset(VisionDataset):
    """
    based on : [link](https://github.com/msminhas93/DeepLabv3FineTuning)
    
    A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample

          
#%% SAVE PREDICTION COMPARATION IMAGE
def save_img(model,dataloader, info='0', root='results', times=1):
    for time in range(times):
        fig, ax = plt.subplots(3,4, figsize=(20,10))
        ax = ax.flatten()

        i=0

        with torch.no_grad():
            for im in range(time*4, 4 + time*4):
                img = dataloader['Test'].dataset[im]['image'].numpy()
                m_p = model(torch.from_numpy(img).unsqueeze_(0))['out']
                
                #plot images
                img = np.moveaxis(img, 0,2)
                ax[i].imshow(m_p.reshape((375, 375,1)))
                ax[i].set_title('PREDICTION')

                j = i+4
                ax[j].imshow(np.moveaxis(dataloader['Test'].dataset[im]['mask'].numpy(), 0,2))
                ax[j].set_title('GROUND TRUTH')

                j = i+8
                ax[j].imshow(img)
                ax[j].set_title('IMAGE')

                i+=1

        # fig.savefig('segmentation_imageK/segmentation_kidney1.jpg')
        fig.savefig(root+ str(info)+'_'+str(time)+ '.jpg')


#%% TRAIN LOOP FUNCTION
def trainloop_seg(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, lr, b):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch','lr', 'batch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

        #write the loss information in a csv
    with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        pred_mask = torch.tensor([], device= device)

        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().ravel() 

                 
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            try:
                                batchsummary[f'{phase}_{name}'].append(metric(y_pred.numpy(), y_true.astype('uint8').detach().numpy()))
                            except:
                                batchsummary[f'{phase}_{name}'].append(0.5)
                    

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            # thresholded = iou_pytorch(outputs,y_pred)
            
            print('{} Loss: {:.4f}'.format(phase, loss))

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            batchsummary['lr'] = lr
            batchsummary['batch']=b
            writer.writerow(batchsummary)
            
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss)) 

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#%% DEFINE PARAMETERS
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--n_epochs', type=float, default=100.0,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=16,
                    help='bacth size')
parser.add_argument('--root_fold', type=str, default='normal_crop',
                    help='root of where the folders with the images and maks are save')
parser.add_argument('--fold_img', type=str, default='good_img',
                    help='folder name where the images are')
parser.add_argument('--fold_masks', type=str, default='mask_parenquima',
                    help='folder name where the masks are')
parser.add_argument('--pt', type=float, default=0.2,
                    help='proportion validation - train')
args = parser.parse_args()
print(args)


#%% TRAIN MODEL

#1. create dataloader
data_dir= args.root_fold

dataloader_parenquima = get_dataloader_single_folder(data_dir = data_dir, 
                                                    image_folder = args.fold_img, mask_folder = args.fold_masks, fraction = args.pt , batch_size= args.batch)

#2. the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
#modify the clasification layer (1 output --> parenchyma)
model.classifier = DeepLabHead(2048, 1)


#3. The criterion and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#4, The metrics to evaluate the model
metrics = {'f1_score': f1_score, 'iou': mean_iou, 'wpintersect': wpintersect}

#5. Create the experiment directory (to save the information of the models etc)
exp_directory = "results"
exp_directory = Path(data_dir+exp_directory)
if not exp_directory.exists(): #create the directory if it does not exits
    exp_directory.mkdir()

#6. train the model
model_best = train_model(model = model, criterion = criterion, dataloaders= dataloader_parenquima, optimizer = optimizer, num_epochs = args.n_epochs, metrics=metrics, bpath=exp_directory, lr=args.lr, b=args.batch)

  # Save the trained model
print('model saved')
torch.save(model_best.state_dict(), exp_directory / f'weights_seg_lr{args.lr}_numBatch{args.batch}.pt')

model.cpu()
model.load_state_dict(torch.load(exp_directory / f'weights_seg_lr{args.lr}_numBatch{args.batch}.pt'))
model.eval()
save_img(model,dataloader_parenquima, info=f'lr{args.lr}_numBatch{args.batch}', root='results', times=3)

torch.cuda.empty_cache()

