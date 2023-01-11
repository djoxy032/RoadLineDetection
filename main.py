import cv2 as cv
import numpy as np
from PIL import Image

from detect_lines import detect_lines, rgb_to_grayscale

if __name__ == "__main__":

    img_path = r"C:\Users\16693\Downloads\advanced-road-materials-ue4-3d-model-low-poly-uasset.jpg"
    output_path = 'out_image.png'
    image = Image.open(img_path)

    np_img = np.array(image)

    img = detect_lines(np_img, 30)

    cv.imwrite(output_path, img)
