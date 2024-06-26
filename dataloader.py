import os, sys
import torch 
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

# 3d shapes dataset
class ShapesDataset(Dataset):
    def __init__(self, x, y, args, shuffle=True, label_ratio=0.01):
        self.device = args.device
        self.shuffle = shuffle

        # swap s.t. 4th col is 0th col
        # this is so 0 - shape, 1 - floor hue, 2 - wall hue, 3 - obj hue, 4 - scale, 5 - orientation
        self.x = x
        self.y = y[:, [4, 0, 1, 2, 3, 5]]
        self.y[:, 5] = (self.y[:, 5] + 30) / 60 # scale orientation to 0 - 1

        # only label a subset of the data (rest set to nan)
        blank_ids = np.random.choice(np.arange(x.shape[0]), int(x.shape[0] * (1-label_ratio)))
        self.y[blank_ids] *= float('nan')

        self.batch_size = args.batch_size
        self.n_batch = int(np.ceil(self.x.shape[0] / self.batch_size))
        if self.shuffle:
            self.ind = np.random.permutation(np.arange(x.shape[0]))
        else:
            self.ind = np.arange(x.shape[0])

    def get_batch(self, i):
        if i != self.n_batch - 1:
            ind = self.ind[i * self.batch_size : (i + 1) * self.batch_size]
        else:
            ind = self.ind[i * self.batch_size : ]
            if self.shuffle: self.shuffle_ind()

        x_ = self.x[ind]
        y_ = self.y[ind]

        return x_, y_

    # call this at end of get_batch
    def shuffle_ind(self):
        self.ind = np.random.permutation(self.ind)

# return dataset to main train and test framework
def celeba_dataset(batch_size, num_workers=4, size=64):
    celeba_transform =  transforms.Compose(
        [transforms.ToTensor(),
        transforms.CenterCrop((178, 178)), # square > rectangle
        transforms.Resize((size, size))]
    )

    # dataloader 
    celeba_set = CelebaDataset(img_path='/drive2/celeba/imgs', label_path='/drive2/celeba/attr.txt', transform=celeba_transform)
    celeba_loader = data.DataLoader(celeba_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return celeba_loader

# custom dataloader for celeba
class CelebaDataset(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.transform = transform

        # label map
        with open('/drive2/celeba/labels.txt', 'r') as f:
            labels = f.read()

        # load labels
        self.y = np.loadtxt(label_path, dtype=str)
        self.img_names = self.y[:,0]
        self.img_names = [name.replace('.jpg', '.png') for name in self.img_names]

    def __getitem__(self, index):
        # get images, then label
        img = Image.open(os.path.join(self.img_path, self.img_names[index]))
        label = self.y[index, 1:]

        # try using float16
        img = self.transform(img).type(torch.float32)
        label = torch.from_numpy(label.astype(np.float32))
        return img, label

    def __len__(self):
        return 202599


