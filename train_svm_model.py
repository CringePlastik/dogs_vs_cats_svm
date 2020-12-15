from image_loader import ImageLoader, ImagePreprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import cv2
import matplotlib.pyplot as plt

img_size = (100, 100)
train_dir = "images/train/train"
test_dir = "images/test/test"
classes = {"cat": 0, "dog": 1}
SZ = img_size[0]


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def load_images(n, img_dir, image_size, class_labels=None):
    data = []
    img_labels = []
    image_loader = ImageLoader(dir_name=img_dir)
    image_preprocessor = ImagePreprocessing(img_size=image_size)
    for i in range(n):
        img = image_loader.get_one_image(labels=class_labels, new_size=image_size)
        gray_scaled = image_loader.rgb2gray(img["data"])
        sobeled = image_loader.sobel_image(gray_scaled)
        deskewed = deskew(sobeled)
        hog = image_preprocessor.get_hog(deskewed)
        data.append(hog)
        img_labels.append(img["target"])
    return np.array(data), np.array(img_labels)


model = SVC(C=10.5, gamma=0.5)
clf = make_pipeline(StandardScaler(), PCA(n_components=42), model)


def learn(n_epochs, n_samples, clf):
    for i in range(n_epochs):
        hog_descriptors, labels = load_images(n_samples, image_size=img_size, class_labels=classes, img_dir=train_dir)
        train_n = int(0.9 * len(hog_descriptors))
        hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
        clf.fit(hog_descriptors_train, labels_train)
        print(f'epoch: {i} \t Accuracy: {accuracy_score(clf.predict(hog_descriptors_test), labels_test)*100}')
        del hog_descriptors, labels

