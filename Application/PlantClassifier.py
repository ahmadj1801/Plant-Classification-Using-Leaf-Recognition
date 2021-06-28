from random import seed, randint
import numpy as np
import pandas as pd
import cv2
from matplotlib import colors
import matplotlib.pyplot as pt
from matplotlib.colors import rgb_to_hsv
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, hamming_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skimage.measure import shannon_entropy
from tabulate import tabulate


def read_data_file():
    # Read in the textual dataset
    dataset = pd.read_csv('dataset/images.txt', delimiter='\t')
    # Remove the segmented path
    dataset.drop('segmented_path', axis=1, inplace=True)
    # Remove images from the 'field'
    dataset.set_index('source', inplace=True)
    dataset.drop('field', axis=0, inplace=True)
    # Return data
    return dataset


# Draw the image with a title
def graph_image(img, title):
    pt.subplot(1, 2, 1)
    pt.title(title)
    pt.imshow(img)
    pt.show()


# Draw colours - used initially. Not anymore.
def graph_colour(c):
    pt.subplot(1, 2, 1)
    pt.imshow(rgb_to_hsv(c))
    pt.show()


# graph colour distribution
def graph_hsv(img):
    b, g, r = cv2.split(img)
    fig = pt.figure()
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    axis.scatter(b.flatten(), g.flatten(), r.flatten(), facecolors=pixel_colors, marker='.')
    axis.set_xlabel("Blue")
    axis.set_ylabel("Green")
    axis.set_zlabel("Red")
    pt.show()


# Preprocess images
def image_pre_processing(df: pd.DataFrame):
    # Features
    c = ['0', '1', '2', '3', '4', '5', '6', 'area', 'perimeter', 'compactness','convex', 'entropy',
         'hcontrast', 'hdissimilarity', 'hhomogeneity', 'henergy', 'hcorrelation',
         'vcontrast', 'vdissimilarity', 'vhomogeneity', 'venergy', 'vcorrelation', 'label']
    f = pd.DataFrame(columns=c)
    # Display the process of a random image in the data set
    # seed(0)
    # input from user
    n = int(input('How many images would you like to train the model on.'
              '\n1. < 5000\n2. >5000 and <10000\n3. Full dataset\n'))
    selected = 500
    if n == 1:
        selected = randint(500, 5000)
    elif n == 2:
        selected = randint(5000, 10000)
    else:
        selected = len(df)
    print('Total Number of Images Being Used = ', selected)
    # preprocess the images
    c = 0
    for index, row in df.iterrows():
        # Processing the ith image which belongs to a class
        print('Pre processing ', c, '/', selected, row['species'])
        # Original image
        original = cv2.imread(row['image_path'])
        # Find region of interest
        x = original.shape[1] - 165
        y = original.shape[0]
        # Obtain a good height to crop at
        if y < 700:
            y = y - 140
        elif y < 770:
            y = y - 155
        else:
            y = y - 180
        # Crop the image
        crop = original[0:y, 0:x]
        # Resize the image for uniformity
        crop = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
        # Convert to Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Initial threshold
        T, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        # Count the amount of white pixels
        num_white = np.sum(thresh == 255)
        if num_white > 2000:
            # Threshold the image
            T, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        opening = filtering = []
        final_image = []
        # Conduct filtering and opening if there are more than 2000 white pixels
        if num_white > 2000:
            # Filtering
            filtering = cv2.medianBlur(thresh, 3)
            # Opening
            morph = cv2.morphologyEx(filtering, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            # Preprocessed image
            final_image = morph
        else:
            # Conduct dilation - make it more visible
            morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((3, 3)))
            final_image = morph
        # Display a last image
        if selected == c:
            plant_name = row['species']
            graph_image(original, plant_name + ' Original Image')
            graph_image(crop, plant_name + ' Cropped Image')
            graph_hsv(crop)
            graph_image(gray, plant_name + ' Grayscale Image')
            graph_image(thresh, plant_name + ' Binary Image')
            if num_white > 2000:
                graph_image(filtering, plant_name + ' Filtered Image')
            graph_image(morph, plant_name + ' after Morphological Operations')
            graph_image(final_image, plant_name + ' Final Image')
            return f
        # Conduct feature extraction
        f = feature_extraction(f, final_image, gray, row['species'])
        # Counter
        c = c + 1
    return f


# GLCM and extracting features
def glcm_features(img, distance, angle):
    glcm = greycomatrix(img, [distance], [angle], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, prop='contrast')
    dissimilarity = greycoprops(glcm, prop='dissimilarity')
    homogeneity = greycoprops(glcm, prop='homogeneity')
    energy = greycoprops(glcm, prop='energy')
    correlation = greycoprops(glcm, prop='correlation')
    # Array of GLCM features
    feat = [contrast[0][0], dissimilarity[0][0], homogeneity[0][0],
            energy[0][0], correlation[0][0]]
    return feat


# Feature Extraction for an image
def feature_extraction(f, img, gray, lbl):
    # HU moments
    moments = cv2.moments(img)
    # Count white pixels - area
    area = np.sum(img == 255)
    # Handling exception for one image - cant div by 0
    if area == 0:
        area = 1
    cnt = 1
    perimeter = 1
    convex = False
    contour, hierachy = cv2.findContours(img, 1, 2)
    if len(contour) > 0:
        cnt = contour[0]
        # Calculate perimeter
        perimeter = cv2.arcLength(cnt, True)
        # Convex
        convex = cv2.isContourConvex(cnt)
    else:
        perimeter = f['perimeter'].iloc[-1]
        convex = f['convex'].iloc[-1]
    # convex is initially boolean
    if convex or convex == 1:
        convex = 1
    else:
        convex = 0
    # Calculate compactness
    compactness = (perimeter**2) / area
    # Calculating entropy
    entropy = shannon_entropy(img)
    # Horizontal GLCM features
    h_glcm_features = glcm_features(gray, 1, 0)
    # Vertical GLCM features
    v_glcm_features = glcm_features(gray, 1, 90)
    # Hu Moments
    hu_moments = cv2.HuMoments(moments)
    for i in range(7):
        if not hu_moments[i] == 0:
            hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
        else:
            hu_moments[i] = 0
    # Add to feature matrix
    f.loc[len(f.index)] = [hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3],
                           hu_moments[4], hu_moments[5], hu_moments[6], area, perimeter,
                           compactness, convex, entropy, h_glcm_features[0], h_glcm_features[1],
                           h_glcm_features[2], h_glcm_features[3], h_glcm_features[4],
                           v_glcm_features[0], v_glcm_features[1], v_glcm_features[2],
                           v_glcm_features[3], v_glcm_features[4], lbl]
    return f


# Train and evaluate models
def train_and_evaluate(x_train, y_train, x_test, y_test):
    # MLP Model
    neural_classifier = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(100,), activation='relu',
                                      max_iter=500, random_state=1)
    neural_classifier.fit(x_train, y_train)
    classifications = neural_classifier.predict(x_test)
    mlp_metrics = ['Multi-Layer Perceptron'] + evaluation(classifications, y_test)
    print('MLP Classification Report\n', classification_report(y_test, classifications, zero_division=1))

    # SVC Model
    support_vector_classifier = svm.SVC(kernel='poly')
    support_vector_classifier.fit(x_train, y_train)
    classifications = support_vector_classifier.predict(x_test)
    svm_metrics = ['Support Vector Classifier'] + evaluation(classifications, y_test)
    print('SVC Classification Report\n', classification_report(y_test, classifications, zero_division=1))

    # Linear SVC Model
    linear_support_vector_classifier = svm.LinearSVC()
    linear_support_vector_classifier.fit(x_train, y_train)
    classifications = linear_support_vector_classifier.predict(x_test)
    linear_svm_metrics = ['Linear Support Vector Classifier'] + evaluation(classifications, y_test)
    print('Linear SVC Classification Report\n', classification_report(y_test, classifications, zero_division=1))

    # Construct neat display
    table = [mlp_metrics, svm_metrics, linear_svm_metrics]
    headings = ['Classifier', 'Accuracy', 'Recall', 'Precision', 'F1-Score', 'Hamming Loss']
    print(tabulate(table, headers=headings))


# Method for evaluation
def evaluation(classifications, truth):
    # Calculate metrics
    accuracy = round(accuracy_score(truth, classifications), 4)
    recall = round(recall_score(truth, classifications, average='micro'), 4)
    precision = round(precision_score(truth, classifications, average='micro', zero_division=1), 4)
    f1_score_value = round(f1_score(truth, classifications, average='micro', zero_division=1), 4)
    hamming_loss_value = round(hamming_loss(truth, classifications), 4)
    # return metrics
    return [accuracy, recall, precision, f1_score_value, hamming_loss_value]


# Normalising Feature Matrix - Min Max - rescaled to fit [0, 1]
def normalise_feature_matrix(f: pd.DataFrame):
    # Scaler Object
    scaler = MinMaxScaler()
    labels = f['label']
    f = f.drop('label', axis=1)
    # Normalization
    normalised = pd.DataFrame(scaler.fit_transform(f), columns=f.columns)
    normalised['label'] = labels
    return normalised


# Main Method
def main():
    # Read datafile
    data = read_data_file()
    # Get features
    features = image_pre_processing(data)
    # Normalise feature
    features = normalise_feature_matrix(features)
    # Encode the target labels
    le = LabelEncoder()
    features['label'] = le.fit_transform(features['label'])
    # X(features) and Y(Labels) to be split for training and testing
    Y = features['label']
    X = features.drop('label', axis=1)
    # Get train and test splits. 90% for training.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=43)
    # Train and evaluate model
    train_and_evaluate(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
