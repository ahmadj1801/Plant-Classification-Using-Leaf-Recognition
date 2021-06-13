from random import seed

import cv2
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
# keras imports for the dataset and building our neural network
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib import colors
from matplotlib.colors import rgb_to_hsv
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def read_data_file():
    # Read in the textual dataset
    dataset = pd.read_csv('../dataset/images.txt', delimiter='\t')
    # Remove the segmented path
    dataset.drop('segmented_path', axis=1, inplace=True)
    # Remove images from the 'lab'
    dataset.set_index('source', inplace=True)
    dataset.drop('field', axis=0, inplace=True)
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
    c = ['image', 'label']
    imgs = []
    lbls = []
    f = pd.DataFrame(columns=c)
    # Display the process of a random image in the data set
    seed(0)
    selected =600  # randint(0, len(df))
    print('Selected Image Number = ', selected)
    # preprocess the images
    c = 0
    for index, row in df.iterrows():
        print('Pre processing ', c, ' ', row['species'])
        # Original image
        original = cv2.imread('../' + row['image_path'])
        crop = original[0:550, 0:550]
        # Resize the image
        original = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
        # Convert to Gray
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        T, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        opening = filtering = np.ones((5, 5))
        num_white = np.sum(thresh == 255)
        if num_white > 2000:
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            # filtering = cv2.medianBlur(thresh, 7)
            final_image = opening
        else:
            final_image = thresh
        final_image = cv2.resize(final_image, (128, 128), interpolation=cv2.INTER_AREA)
        final_image = np.array(final_image, dtype=np.float)
        f.loc[len(f.index)] = [final_image, row['species']]
        imgs.append(final_image)
        lbls.append(row['species'])
        # Display a random image
        if selected == c:
            cv2.imshow('Original', original)
            cv2.imshow('Grayscale', gray)
            cv2.imshow('Threshold', thresh)
            if num_white > 2000:
                cv2.imshow('Opening', opening)
                cv2.imshow('Filtering', filtering)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            imgs = np.array(imgs, dtype=float)
            return f, imgs, lbls
        # Counter
        c = c + 1
    return f


def main():
    print('starting...')
    data = read_data_file()
    features, imgs, lbls = image_pre_processing(data)
    print(features)
    le = LabelEncoder()
    features['label'] = le.fit_transform(features['label'])
    Y = features['label']
    Y = np.array(Y)
    print(Y.dtype)
    X = imgs
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=43)
    # y_train = np.asarray(y_train).astype(np.int)
    # normalizing the data to help with the training
    # building the input vector from the 28x28 pixels
    x_train /= 255
    x_test /= 255

    # building a linear stack of layers with the sequential model
    model = Sequential()
    # convolutional layer
    model.add(
        Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(1, 1)))
    # flatten output of conv
    model.add(Flatten())
    # hidden layer
    model.add(Dense(100, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
    classifications = model.predict(x_test)
    '''neural_classifier = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(100,), activation='logistic',
                                      random_state=1)
    neural_classifier.fit(x_train, y_train)
    classifications = neural_classifier.predict(x_test)'''

    print(classifications)
    print(y_test)
    print(classification_report(y_test, classifications))
    print(accuracy_score(y_test, classifications))


if __name__ == '__main__':
    main()
