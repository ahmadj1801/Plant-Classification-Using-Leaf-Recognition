from random import seed, randint

import numpy as np
import pandas as pd
import cv2
from matplotlib import colors
import matplotlib.pyplot as pt
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import mahotas as mt
from sklearn.neighbors import KNeighborsClassifier


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
    c = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
         '16', '17', '18', '19', 'label']
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
        mask = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)  # cv2.inRange(hue, lg, dg)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print(mask.shape)
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
    textures = mt.features.haralick(img)
    mean = textures.mean(axis=0)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        if not hu_moments[i] == 0:
            hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
        else:
            hu_moments[i] = 0
    f.loc[len(f.index)] = [hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
                           hu_moments[4], hu_moments[5], hu_moments[6], mean[0], mean[1],
                           mean[2], mean[3], mean[4], mean[5], mean[6], mean[7], mean[8],
                           mean[9], mean[10], mean[11], mean[12], lbl]
    return f


def main():
    data = read_data_file()
    features = image_pre_processing(data)
    print(features)

    le = LabelEncoder()
    features['label'] = le.fit_transform(features['label'])
    print(features)

    X = features.drop('label', axis=1)
    Y = features['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
    '''neural_classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    neural_classifier.fit(x_train, y_train)
    classifications = neural_classifier.predict(x_test)'''
    knn = KNeighborsClassifier(5, weights='uniform')
    knn.fit(x_train, y_train)
    classifications = knn.predict(x_test)
    print(classifications)
    print(classification_report(y_test, classifications))
    print(accuracy_score(y_test, classifications))


if __name__ == '__main__':
    main()
