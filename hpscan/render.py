"""Rasterisation of scatterplots.

Used for the input/target images of the image-based U-Net baseline and for visualising
predictions. The rendering settings (figure size, axis limits, marker) are the ones used
to train the released baseline checkpoint, do not change them.
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLEAU_20 = ['#000000', '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7',
              '#9c755f', '#bab0ab', '#a0cbe8', '#ffbe7d', '#8cd17d', '#b6992d', '#f1ce63', '#499894', '#86bcb6',
              '#ff9d9a', '#79706e', '#d37295']


def _relabel(labels):
    """Map arbitrary non-negative labels to 0..K (0 stays noise)."""
    unique = np.unique(labels).tolist()
    if 0 not in unique:
        unique = [0] + unique
    index_map = np.array([(unique.index(n) if n in unique else -1) for n in np.arange(1 + np.max(unique))])
    return index_map[labels.tolist()]


def _figure_to_array(fig):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def draw_scatterplot(points, labels=None, point_size=2, border=True, text=None):
    """Render points in [-1, 1]^2 to a 1000x1000 RGB image (origin bottom-left).

    labels: optional per-point labels (0 = noise, drawn black).
    """
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    if labels is None:
        plt.scatter(x=points[:, 0], y=points[:, 1], marker='.', s=point_size, c="black")
    else:
        labels = _relabel(np.asarray(labels).astype(int))
        colors = TABLEAU_20 if np.max(labels) < len(TABLEAU_20) else \
            [TABLEAU_20[0]] + [matplotlib.colors.to_hex(c) for c in plt.colormaps['prism'](np.linspace(0, 1, np.max(labels) + 1))[1:]]
        plt.scatter(x=points[:, 0], y=points[:, 1], marker='.', s=point_size, c=[colors[i] for i in labels])
    fig.tight_layout(pad=0)
    image = _figure_to_array(fig)
    if border:
        image[0:2, ...] = 0
        image[-2:, ...] = 0
        image[:, -2:, ...] = 0
        image[:, 0:2, ...] = 0
    image = cv2.flip(image, 0)
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.putText(image, text, (int((image.shape[1] - textsize[0]) / 2), 30), font, 1, (0, 0, 0), 2)
    return image


def render_input(points, image_size=128):
    """Grayscale (H, W) uint8 input image of the image baseline. points: (N, 2) in [-1, 1]."""
    image = draw_scatterplot(points, point_size=100)
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (image_size, image_size))


def project_points(image, points, kernel_size=3):
    """Pixel coordinates (x, y) of points in [-1, 1] on `image`, snapped to the darkest pixel in a
    `kernel_size` neighbourhood."""
    image_size = image.shape[0]
    projected = np.round((points + 1.0) * 0.5 * image_size).astype(np.int32)
    dim = kernel_size // 2
    snapped = []
    for (x, y) in projected[:, 0:2]:
        neighbors = image[y - dim:y + dim + 1, x - dim:x + dim + 1]
        idx = np.argmin(neighbors)
        snapped.append([x + (idx % kernel_size) - 1, y + (idx // kernel_size) - 1])
    return np.array(snapped).astype(np.int32)
