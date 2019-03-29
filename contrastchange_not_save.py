from PIL import Image, ImageEnhance
import os
import cv2
import numpy as np

def change(image_cv, uprate=1.5):

    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_cv)

    con_upper = ImageEnhance.Contrast(image)
    image = con_upper.enhance(uprate)

    return_image = np.asarray(image)
    return_image = cv2.cvtColor(return_image, cv2.COLOR_RGB2BGR)

    return return_image
