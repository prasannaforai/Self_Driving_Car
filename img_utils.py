import numpy as np
import pandas as pd
from keras.utils import Sequence
import os
from skimage.io import imread
from skimage.transform import resize



class ImageDataGenerator(Sequence):
    
    def __init__(self, data_dir, target_shape, batch_size, df, x_col, y_col, shuffle=False):
        self.data_dir = None
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.filenames = df[x_col].values
        self.targets = df[y_col].values
        self.num_samples = len(self.filenames)
        self.num_batches = (self.num_samples  // self.batch_size) + 1
        self.shuffle = shuffle
        self.indices = np.arange(self.num_samples)
        
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError("Invalid FileName")

                 
    def __len__(self): 
        return self.num_batches
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
        
    #infinite images generator
    def __getitem__(self, batch_index):
            
        batch_idxs = self.indices[batch_index*self.batch_size : (batch_index+1)*self.batch_size]
        batch_imgs = np.array([resize(imread(self.data_dir+f)[-150:], self.target_shape) for f in self.filenames[batch_idxs]])
        targets = np.reshape(self.targets[batch_idxs], (-1, 1))
        
        return (batch_imgs, targets)
        #yield (batch_imgs, targets)
        #return  iter([batch_idxs])
            
            