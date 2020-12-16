from image_loader import ImageLoader, ImagePreprocessing
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import cv2


def deskew(img, SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def load_images(n, img_dir, image_size=None, class_labels=None):
    data = []
    img_labels = []
    image_loader = ImageLoader(dir_name=img_dir)
    image_preprocessor = ImagePreprocessing(img_size=image_size)
    for i in range(n):
        img = image_loader.get_one_image(labels=class_labels, new_size=image_size)
        gray_scaled = image_loader.rgb2gray(img["data"])
        # sobeled = image_loader.sobel_image(gray_scaled)
        # deskewed = deskew(sobeled, image_size[0])
        hog = image_preprocessor.get_hog(np.uint8(gray_scaled))
        data.append(hog)
        img_labels.append(img["target"])
    return np.array(data), np.array(img_labels)


# model = SVC(C=8.5, gamma=0.5)
# clf = make_pipeline(StandardScaler(), PCA(n_components=2000), model)


def learn(n_epochs, n_samples, clf, train_dir, img_size=None, classes=None):
    for i in range(n_epochs):
        hog_descriptors, labels = load_images(n_samples, image_size=img_size, class_labels=classes, img_dir=train_dir)
        train_n = int(0.9 * len(hog_descriptors))
        hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
        clf.fit(hog_descriptors_train, labels_train)
        print(f'epoch: {i} \t Accuracy: {accuracy_score(clf.predict(hog_descriptors_test), labels_test)*100}')
        del hog_descriptors, labels


def predict(img_name, model, dir_name, new_size=None):
    image_loader = ImageLoader(dir_name=dir_name)
    img = image_loader.get_one_image(new_size=new_size, path=img_name)
    image_preprocessor = ImagePreprocessing(img_size=new_size)
    gray_scaled = image_loader.rgb2gray(img["data"])
    # sobeled = image_loader.sobel_image(gray_scaled)
    # deskewed = deskew(sobeled, new_size[0])
    hog = image_preprocessor.get_hog(np.uint8(gray_scaled))
    return model.predict([hog])




# learn(n_epochs=1, n_samples=25000, clf=clf)
