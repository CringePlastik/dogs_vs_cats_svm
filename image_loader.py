import numpy as np
import os
import random
import cv2


class ImageLoader:
    def __init__(self, dir_name, img_ext=".jpg"):
        self.dir_name = dir_name
        self.img_ext = img_ext

    def get_one_image(self, labels=None, new_size=None, path=None):
        if not path:
            file_names = [os.path.join(self.dir_name, img) for img in os.listdir(self.dir_name)]
            file_names = [file for file in file_names if os.path.splitext(file)[1] == self.img_ext]  # get only images
            path_to_img = random.choice(file_names)
            real_image_name = path_to_img.split("/")[-1]
        else:
            path_to_img = os.path.join(self.dir_name, path)
            real_image_name = path_to_img.split("/")[-1]
        label = None
        if labels:  # we need a train image with a label, otherwise, we need a test image
            for class_name in labels:
                if class_name in real_image_name:
                    label = labels[class_name]
            if label is None:
                raise Exception("no class specified for %s" % real_image_name)
        img_pixels = cv2.imread(path_to_img)
        if not new_size:
            return {"data": img_pixels, "target": label}
        # if the resized image is needed
        img_pixels = cv2.resize(img_pixels, new_size, interpolation=cv2.INTER_CUBIC)
        return {"data": img_pixels, "target": label}

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def sobel_image(self, im):
        im = np.float32(im)
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        return np.uint8(mag)

    def get_batch(self, start, stop, labels=None, new_size=None, shuffle=None):
        file_names = [os.path.join(self.dir_name, img) for img in os.listdir(self.dir_name)]
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(file_names)
        needed_files = file_names[start:stop+1]
        print(needed_files[0:10])
        images = []
        img_labels = []
        for file in needed_files:
            real_image_name = file.split("/")[-1]
            label = None
            if labels:
                for class_name in labels:
                    if class_name in real_image_name:
                        label = labels[class_name]
                if label is None:
                    raise Exception("no class specified for %s" % real_image_name)
            img_pixels = cv2.imread(file)
            if not new_size:
                images.append(img_pixels)
                img_labels.append(label)
            else:
                img_pixels = cv2.resize(img_pixels, new_size, interpolation=cv2.INTER_CUBIC)
                images.append(img_pixels)
                img_labels.append(label)
        return {"data": np.array(images), "target": np.array(img_labels)}










class ImagePreprocessing:
    def __init__(self, img_size):
        self.img_size = img_size

    def get_hog(self, img):
        winSize = self.img_size
        blockSize = (20, 20)
        blockStride = (10, 10)
        cellSize = (10, 10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
        return self.resize_hog(hog.compute(img))

    def resize_hog(self, hog):
        return np.reshape(hog, (hog.shape[0], ))

    def simple_preproc(self, img):
        return np.resize(img, (img.shape[0]*img.shape[1], ))/255.0







