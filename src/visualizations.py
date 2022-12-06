import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from e3nn import o3


def plot_image(fig, ax,
               image: torch.tensor,
               vpad=60,
              ):
    image = torch.nn.functional.pad(image, (0, 0, vpad, vpad), 'constant', 1)
    ax.imshow(image.permute(1,2,0))
    ax.axis('off')


def plot_to_image(fig):
    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    plt.close(fig)
    return rgb_array


def plot_so3_distribution(probs: torch.Tensor,
                          rots: torch.Tensor,
                          gt_rotation=None,
                          fig=None,
                          ax=None,
                          display_threshold_probability=0.000005,
                          prob_threshold: float=0.00001,
                          show_color_wheel: bool=True,
                          canonical_rotation=torch.eye(3),
                         ):
    '''
    Taken from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py
    '''
    cmap = plt.cm.hsv

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        alpha, beta, gamma = o3.matrix_to_angles(rotation)
        color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
        ax.scatter(alpha, beta-np.pi/2, s=2000, edgecolors=color, facecolors='none', marker=marker, linewidth=5)
        ax.scatter(alpha, beta-np.pi/2, s=1500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)
        ax.scatter(alpha, beta-np.pi/2, s=2500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=400)
        fig.subplots_adjust(0.01, 0.08, 0.90, 0.95)
        ax = fig.add_subplot(111, projection='mollweide')

    rots = rots @ canonical_rotation
    scatterpoint_scaling = 3e3
    alpha, beta, gamma = o3.matrix_to_angles(rots)

    # offset alpha and beta so different gammas are visible
    R = 0.02
    alpha += R * np.cos(gamma)
    beta += R * np.sin(gamma)

    which_to_display = (probs > display_threshold_probability)


    # Display the distribution
    ax.scatter(alpha[which_to_display],
               beta[which_to_display]-np.pi/2,
               s=scatterpoint_scaling * probs[which_to_display],
               c=cmap(0.5 + gamma[which_to_display] / 2. / np.pi))
    if gt_rotation is not None:
        if len(gt_rotation.shape) == 2:
            gt_rotation = gt_rotation.unsqueeze(0)
        gt_rotation = gt_rotation @ canonical_rotation
        _show_single_marker(ax, gt_rotation, 'o')
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    img = plot_to_image(fig)
    plt.close(fig)
    return img


def plot_predictions(images, probs, rots, gt_rots, num=4, path=None):
    images = images.cpu()
    probs = probs.detach().cpu()
    rots = rots.cpu()
    gt_rots = gt_rots.cpu()

    fig = plt.figure(figsize=(4.8, np.ceil(num/2)), dpi=300)
    gs = GridSpec(int(np.ceil(num/2)), 4, width_ratios=[1,3,1,3], wspace=0, left=0, top=1, bottom=0, right=1)

    for i in range(num):
        ax0 = fig.add_subplot(gs[2*i])
        plot_image(fig, ax0, images[i])

        ax1 = fig.add_subplot(gs[2*i+1])
        img = plot_so3_distribution(probs[i], rots, gt_rotation=gt_rots[i],
                                    show_color_wheel=i==0)
        ax1.imshow(img)
        ax1.axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
