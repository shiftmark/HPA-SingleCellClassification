import matplotlib.pyplot as plt
import numpy as np
import os

def plot_images(images, fig_size=(10,10), show_axes=True):
    '''
    Plot 9 images (3x3) or less. If more are provided, the firtst 9 are returned.
    Args:
        images (list of np.arrays, paths): images to plot. A list of numpy arrays and/or local path strings.
        fig_size (tuple): the figure size, default (10, 10)
        show_axes (bool): if True, show axes
    Returns:
        Plot of 9 or less images
    '''
    images = images[:9]
    def plt_img(imgs):
        plt.figure(figsize=fig_size)
        for idx, img in enumerate(imgs):
            plt.subplot(3, 3, idx+1)
            plt.imshow(img)
            plt.axis(show_axes)
        plt.tight_layout()
        plt.show()
     
    for idx, img in enumerate(images):
        if not isinstance(img, (np.ndarray, str)):
            raise TypeError(f'The input type must be a list of numpy arrays or strings but got {type(img)}')

        if isinstance(img, str):
            if not os.path.exists(img):
                raise ValueError(f'The string "{img}" is not a valid local path.')
            else:
                images[idx] = plt.imread(img)

    
    return plt_img(images)


def plot_history(history, metrics=['loss'], grid=True, legend=True):
    '''
    Plot train history metrics.
    Args:
        history (list): the history list name
        metrics (list of strings): the metrics to plot. Default is 'loss'
        grid (bool): if True, show grid. Default is True
        legend (bool): if True, show legend. Default is True
    '''
    
    plt.figure(figsize=(12, 12))
    for idx, metric in enumerate(metrics):
        plt.subplot(9, 1, idx+1)
        plt.xlabel('Epochs', fontweight='bold')
        plt.ylabel(metric)
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.grid(grid)
        plt.legend(legend)
        plt.tight_layout()
        plt.show()

    return
