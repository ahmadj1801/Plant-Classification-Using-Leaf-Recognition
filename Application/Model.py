from random import seed, randint

import numpy as np
import pandas as pd
import cv2
import skimage


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


def image_pre_processing(df: pd.DataFrame):
    # local variables for feeding pre processing
    # original, grey, equalization, median, threshold = ''
    global display
    o1, o2, o3 = '', '', ''
    # Display the process of a random image in the data set
    seed(0)
    selected = 0  # randint(0, len(df))
    print('Selected Image Number = ', selected)
    # preprocess the images
    c = 0
    for index, row in df.iterrows():
        print('Pre processing ', c)
        # Original image
        # print(df['image_path'][image])
        original = cv2.imread(row['image_path'])
        # Convert to gray scale
        grey = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        # Median Filtering
        median = cv2.medianBlur(grey, 5) #cv2.GaussianBlur(grey, (5,5), 0) #cv2.medianBlur(grey, 5)
        # Threshold
        T, threshold = cv2.threshold(median, 100, 255, type=cv2.THRESH_BINARY_INV)# cv2.threshold(median, 100, 255, type=cv2.THRESH_BINARY_INV)
        # Resize
        # cv2.resize(threshold, (400, 400), interpolation=cv2.INTER_AREA)
        if selected == c:
            # cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR, threshold)
            # row1 = np.hstack((grey, equalization))
            # row2 = np.hstack(median)
            # display = np.vstack((row1, row2))
            cv2.imshow('Original Image', original)
            cv2.imshow('Grey Image', grey)
            cv2.imshow('Smoothed Image', median)
            cv2.imshow('Threshold Image', threshold)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        c = c + 1


def main():
    data = read_data_file()
    image_pre_processing(data)
    pass


if __name__ == '__main__':
    main()
