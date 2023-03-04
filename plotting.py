import os, sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

plt.style.use('seaborn')

# visualize images
def save_vis(x, x_out, path, n=4):
    # take first n images
    x = x[:n]; x_out = x_out[:n]

    # x in top row, x_out in bottom row
    # stitch using einops
    x = rearrange(x, 'b c h w -> h (b w) c', b=n)
    x_out = rearrange(x_out, 'b c h w -> h (b w) c', b=n)
    img = torch.stack((x, x_out))
    img = rearrange(img, 'b h w c -> (b h) w c', b=2)

    # convert to numpy and save
    img = img.detach().cpu().numpy()
    img = Image.fromarray(np.uint8(img*255))
    img.save(os.path.join(path, 'sample.png'))

# save loss
def save_loss(loss, path='loss'):
    loss = np.array(loss)
    np.save(os.path.join(path, 'loss.npy'), loss)

    # plot loss
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(loss[:, 0], label='recon', color='blue')
    ax2.plot(loss[:, 1], label='kld', color='black')

    ax1.set_title('Reconstruction Loss')
    ax2.set_title('KLD Loss')

    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()

