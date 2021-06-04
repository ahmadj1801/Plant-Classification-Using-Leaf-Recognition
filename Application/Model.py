from random import seed, randint

import numpy as np
import pandas as pd
import cv2
from matplotlib import colors
import matplotlib.pyplot as pt
from matplotlib.colors import rgb_to_hsv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import  KNeighborsClassifier


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
    # Features
    c = ['0', '1', '2', '3', '4', '5', '6', 'label']
    f = pd.DataFrame(columns=c)
    # Display the process of a random image in the data set
    seed(0)
    selected = 300  # randint(0, len(df))
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
        filtering_1 = cv2.medianBlur(mask, 5)
        # Opening -> remove stem
        opening = cv2.morphologyEx(filtering_1, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        filtering_2 = cv2.medianBlur(opening, 15)
        # Display a random image
        if selected == c:
            # Horizontal display
            row1 = np.hstack((mask, filtering_1))
            row2 = np.hstack((opening, filtering_2))
            display = np.vstack((row1, row2))
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
            cv2.imshow('Pre Processing', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return f
        # Conduct feature extraction
        f = feature_extraction(f, filtering_2, row['species'])
        # Counter
        c = c + 1
    return f


def feature_extraction(f, img, lbl):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
    f.loc[len(f.index)] = [hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
                           hu_moments[4], hu_moments[5], hu_moments[6], lbl]
    return f


def main():
    data = read_data_file()
    features = image_pre_processing(data)
    print(features)

    lbl_df = features['label']
    lbl_df.apply(LabelEncoder().fit_transform)

    X = features.drop('label', axis=1)
    Y = ''
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


if __name__ == '__main__':
    main()
