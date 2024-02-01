import h5py
import torch
import time

'''
converts hd5 dataset to torch tensors
the dataset is less than a Gb, so we can just load it at once

one might be tempted to parallelize this,
however using concurrent.futures.ProcessPoolExecutor
results in errors...
'''

if __name__ == '__main__':
    n = 480000

    # load full dataset
    print('loading full dataset')
    dataset = h5py.File('data/3dshapes.h5', 'r')
    imgs = dataset['images']
    labels = dataset['labels']

    # get and convert to torch objects
    print('converting to torch')
    print('this will take a minute...')
    print(n / len(imgs))

    t0 = time.time()

    imgs = dataset['images'][:n]
    labels = dataset['labels'][:n]

    # save torch reps as imgs as uint8 and labels as float64
    imgs = torch.tensor(imgs, dtype=torch.uint8)
    labels = torch.tensor(labels, dtype=torch.float64)

    # save
    print('saving...')
    torch.save(imgs, f'data/imgs.pt')
    torch.save(labels, f'data/labels.pt')

    print(time.time() - t0)
