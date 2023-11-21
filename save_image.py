"""
This file is used to save the image from the camera
"""

import numpy as np
import PIL.Image


def save_image(path: str, data: np.ndarray):
    """
    Saves the image to the given path.
    :param path: The path to save the image to
    :param data: The image data to save
    """
    image = PIL.Image.fromarray(data)
    image.save(path)