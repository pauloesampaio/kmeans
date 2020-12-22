import numpy as np
from sklearn.cluster import KMeans
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import altair as alt
import pandas as pd


def open_image(path, remote=False):
    """Helper to open image. If remote, downloads it, if local, simply opens it.

    Args:
        path (str): Path to local image or url to remote
        remote (bool, optional): If true, downloads instead of opens. Defaults to False.

    Returns:
        pil.Image: Pil image
    """
    if remote:
        return download_image(path)
    else:
        return Image.open(path)


def download_image(image_url):
    """Downloads image from an url and returns PIL image

    Args:
        image_url (str): url of the desires image

    Returns:
        PIL Image: downloaded image
    """
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = Image.open(im_bytes)
    return image


def resize_image(image, dest_size=None, max_side=None):
    """Function to resize image to a specified size or so the largest side has a defined size

    Args:
        image (np.array): Image as a numpy array
        dest_size (int tuple, optional): destination size of the image (width, height)
        max_side (int, optional): [description]. Size of the largest side of the image

    Returns:
        np.array: Array representation of the resized image
    """
    if dest_size:
        return image.resize(dest_size, Image.LANCZOS)
    elif max_side:
        scale_factor = max(image.size) / max_side
        (width, height) = (
            image.width // scale_factor,
            image.height // scale_factor,
        )
        resized_image = image.resize((int(width), int(height)), Image.LANCZOS)
        return resized_image
    else:
        return image


def get_elbow_curve(data, min_k=1, max_k=10, random_state=12345):
    """Function to iteratively run k-means with k from 1 to 10 and get the inertia.

    Args:
        data (np.array): Array to be clustered
        min_k (int, optional): Min value of k. Defaults to 1.
        max_k (int, optional): Max value of k. Defaults to 10.
        random_state (int, optional): Random seed. Defaults to 12345.

    Returns:
        List: List with inertia value for each k
    """
    elbow = []
    for i in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        kmeans.fit(data)
        elbow.append((i, kmeans.inertia_))
    return elbow


def plot_elbow_curve(elbow):
    """Helper to build elbow plot

    Args:
        elbow (List): List with inertia value for each k

    Returns:
        altair.chart: Altair chart
    """
    initial_inertia = elbow[0][1]
    elbow_df = pd.DataFrame(elbow, columns=["k", "Relative inertia (%)"])
    elbow_df["Relative inertia (%)"] = (
        elbow_df["Relative inertia (%)"] / initial_inertia * 100
    )
    chart = (
        alt.Chart(elbow_df)
        .mark_line()
        .encode(
            alt.X("k", axis=alt.Axis(values=elbow_df["k"].values)),
            alt.Y("Relative inertia (%)"),
        )
    )
    return chart


def img_to_list(image):
    """Reshape image to a list

    Args:
        image (pil.Image): Pil Image

    Returns:
        [np.array]: Array with all the pixels as rows and the R,G,B values as column
    """
    return np.array(image).reshape((-1, 3))


def create_cluster_image(kmeans, shape, add_legend=False):
    """Given a fitted k-means model, reshape it into an image format and generates a visualization.

    Args:
        kmeans (sklearn.Kmeans): Fitted k-means model
        shape (tuple(int, int)): height and width of image
        add_legend (bool, optional): If true, adds legend with cluster names. Defaults to False.

    Returns:
        plt.imshow: matplotlib Figure
    """
    k = kmeans.n_clusters
    cluster_centers_hsv = kmeans.cluster_centers_.astype("uint8").reshape(1, k, 3)
    cluster_centers_rgb = np.array(
        Image.fromarray(cluster_centers_hsv, mode="HSV").convert("RGB")
    )
    cluster_mask = kmeans.labels_.reshape(shape)
    cmap = ListedColormap(cluster_centers_rgb.reshape((k, 3)) / 256)
    fig = plt.figure()
    if add_legend:
        patches = [
            Patch(color=c, label="Cluster {}".format(i))
            for i, c in enumerate(cmap.colors)
        ]
        plt.legend(
            handles=patches, frameon=True, bbox_to_anchor=(1.05, 1), loc="upper left"
        )
    plt.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    plt.grid(False)
    plt.imshow(cluster_mask, cmap=cmap)
    return fig
