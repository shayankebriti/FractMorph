from datetime import datetime
import matplotlib.pyplot as plt
from plot.utils import save_plot


def plot_acdc_slices(dataset, index, output_name='plot_acdc_slices'):
    """
    Generate and save a grid of ED/ES images and their labels for one sample.

    Parameters
    ----------
    dataset : ACDCDataset
        The dataset to sample from.
    index : int
        Index of the sample to plot.
    output_name : str, optional
        Filename (without extension) under which to save the figure.
    """
    moving, fixed, moving_label, fixed_label = dataset[index]

    moving = moving.numpy()
    fixed = fixed.numpy()
    moving_label = moving_label.squeeze(0).numpy()
    fixed_label = fixed_label.squeeze(0).numpy()

    num_slices = moving.shape[0]
    fig, axes = plt.subplots(
        nrows=4,
        ncols=num_slices,
        figsize=(num_slices * 2, 8),
        constrained_layout=True
    )

    titles = ['Moving', 'Fixed', 'Moving Label', 'Fixed Label']
    arrays = [moving, fixed, moving_label, fixed_label]

    for row, (title, arr) in enumerate(zip(titles, arrays)):
        for col in range(num_slices):
            ax = axes[row, col]
            ax.imshow(arr[col], cmap='gray')
            ax.axis('off')
            ax.set_title(f"{title} {col + 1}")

    save_plot(fig, output_name)
