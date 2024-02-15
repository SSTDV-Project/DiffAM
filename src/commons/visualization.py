import numpy as np

import matplotlib.colors as colors

def sdf_cmap(N = 256):

    cdict = {
        'red': [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        'green': [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        'blue': [
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    }

    clist = colors.LinearSegmentedColormap('', cdict)(np.linspace(0, 1, N))
    M = N // 2
    clist[M-1:M+1, :] = 1
    cmap = colors.LinearSegmentedColormap.from_list('sdf_cmap', clist)
    return cmap

def sdf_divnorm(vmin, vcenter, vmax):
    return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

def center_norm():
    return colors.CenteredNorm()

def linear_norm(vmin, vmax):
    return colors.Normalize(vmin, vmax)

def slice_animation(plt, img, slice_axis=2, x_axis=0, y_axis=1, **imshow_kwargs):
    import matplotlib.animation as anim

    img = img.transpose(slice_axis, x_axis, y_axis)
    N = img.shape[0]
    slices = []
    fig = plt.figure()
    for i in range(N):
        img_slice = img[i,:,:]
        ax = plt.imshow(img_slice, animated=True, **imshow_kwargs)
        slices.append([ax])
    ani = anim.ArtistAnimation(fig, slices, blit=True)
    return ani
