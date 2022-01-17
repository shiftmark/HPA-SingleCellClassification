import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def plot_images(images, fig_size=(10,10), show_axes=True):
    """Plot 9 images (3x3) or less. If more are provided, the firtst 9 are returned.
    Args:
        images (list of np.arrays, paths): images to plot. A list of numpy arrays and/or local path strings.
        fig_size (tuple): the figure size, default (10, 10)
        show_axes (bool): if True, show axes
    Returns:
        Plot of 9 or fewer images"""
    images = images[:9]

    def plt_img(imgs):
        plt.figure(figsize=fig_size)

        for i, j in enumerate(imgs):
            shape_two = len(j.shape) == 2
            plt.subplot(3, 3, i + 1)
            plt.imshow(j, cmap='gray') if shape_two else plt.imshow(j)
            plt.axis(show_axes)

        plt.tight_layout()
        plt.show()
     
    for idx, img in enumerate(images):
        if not isinstance(img, (np.ndarray, str, tf.Tensor)):
            raise TypeError(f'The input type must be a list of numpy arrays, strings or tf tensors but got {type(img)}')

        if isinstance(img, str):
            if not os.path.exists(img):
                raise ValueError(f'The string "{img}" is not a valid local path.')
            else:
                images[idx] = plt.imread(img)
        if isinstance(img, tf.Tensor):
            images[idx] = img.numpy()
                
    return plt_img(images)


def plot_history(history, metrics=None, grid=True, legend=True):
    """Plot train history metrics.
    Args:
        history (list): the history list name
        metrics (list of strings): the metrics to plot. Default is 'loss'
        grid (bool): if True, show grid. Default is True
        legend (bool): if True, show legend. Default is True"""

    if metrics is None:
        metrics = ['loss']
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
