from skimage.io import imread,imsave
import pandas as pd
import numpy as np 
import torch
import os
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision

def one_hot_encoding(label,num_cat):
    temp=np.zeros(num_cat)
    temp[label]=1
    return temp


class Dataset(Dataset):

    def __init__(self,csv_file='fashion-mnist_train.csv',transform=None):
       
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.num_cat=len(self.df.label.unique())

        self.img_indexes=np.arange(len(self.df))

 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img=self.df.loc[idx].values
        label=one_hot_encoding(img[0],self.num_cat)
        img=np.reshape(img[1:],(28,28))
        sample={'img':img,'label':label}
        

        if self.transform:
            sample = self.transform(sample)

        return sample


class Scale(object):
    def __call__(self, sample):
        
        img,label=sample['img'],sample['label']
        img=np.pad(img, ((2, 2), (2, 2)), 'minimum')
        
        return {'img':img/255,'label':label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        img,label=sample['img'],sample['label']
        
        img_shape=img.shape
        
        if len(img_shape)!=3:
            img=np.expand_dims(img,axis=2)
        
        img = img.transpose((2, 0, 1))
        img= torch.from_numpy(img).type(torch.FloatTensor)
        label=torch.from_numpy(label).type(torch.IntTensor)
        
        return {'img':img,'label':label}


class Normalize(object):
    def __init__(self,mean, std,):
      self.mean=mean
      self.std=std

    def __call__(self, sample):
        img,label=sample['img'],sample['label']
        
        for i in range(img.shape[0]):
          
          img[i,:,:]=(img[i,:,:]-self.mean[i])/self.std[i]
        return {'img':img,'label':label}
    
    
    
def visualize_loader(loader,index):
    for i,sample in enumerate(loader):
        #print(sample['image'].shape)
        if i==index:
            img,label=sample['img'][index],sample['label'][index]

            img=img.numpy()
            
            img=img.transpose(1,2,0)
            img=np.squeeze(img,2)
            print("Image shape : {}\nImage maximum val : {}\nImage minimum val ; {}\nImage label : {}"\
                  .format(img.shape,np.amax(img),np.min(img),label.numpy()))

            return img.astype(np.uint8)
            