from random import seed, randint

import numpy as np
import pandas as pd
import cv2
from matplotlib import colors
import matplotlib.pyplot as pt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def read_data_file():
    # Read in the textual dataset
    dataset = pd.read_csv('dataset/images.txt', delimiter='\t')
    # Remove the segmented path
    dataset.drop('segmented_path', axis=1, inplace=True)
    # Remove images from the 'lab'
    dataset.set_index('source', inplace=True)
    dataset.drop('lab', axis=0, inplace=True)
    # Return data
    return dataset


def graph_image(img):
    pt.subplot(1, 2, 1)
    pt.imshow(img)
    pt.show()


def graph_colour(c):
    pt.subplot(1, 2, 1)
    pt.imshow(rgb_to_hsv(c))
    pt.show()


def graph_hsv(img):
    h, s, v = cv2.split(img)
    fig = pt.figure()
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker='.')
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    pt.show()


def image_pre_processing(df: pd.DataFrame):
    # Display the process of a random image in the data set
    seed(0)
    selected = 384  # randint(0, len(df))
    print('Selected Image Number = ', selected)
    # preprocess the images
    c = 0
    for index, row in df.iterrows():
        print('Pre processing ', c, ' ', row['species'])
        # Original image
        original = cv2.imread(row['image_path'])
        # Resize the image
        original = cv2.resize(original, (400, 400), interpolation=cv2.INTER_AREA)
        # Convert to RGB
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        # Change the colour space
        hue = cv2.cvtColor(original, cv2.COLOR_RGB2HSV_FULL)
        # Set upper and lower bounds for segmentation
        lg = (10, 50, 10)
        dg = (128, 255, 128)
        # Threshold segmentation
        mask = cv2.inRange(hue, lg, dg)
        # Filtering noise
        filtering = cv2.medianBlur(mask, 5)
        # Display a random image
        if selected == c:
            # Horizontal display
            row = np.hstack((mask, filtering))
            # Display RGB image
            graph_image(original)
            # Graph of HSV space
            graph_hsv(hue)
            # Graph of threshold colours
            light_square = np.full((10, 10, 3), lg, dtype=np.uint8) / 255.0
            dark_square = np.full((10, 10, 3), dg, dtype=np.uint8) / 255.0
            graph_colour(light_square)
            graph_colour(dark_square)
            # Display HSV Image
            graph_image(hue)
            # Display
            cv2.imshow('Pre Processing', row)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # Counter
        c = c + 1


def main():
    data = read_data_file()
    image_pre_processing(data)
    pass


if __name__ == '__main__':
    main()
