import os
import numpy as np
import random
import cv2


class ImageLoader:
    """
    Class for loading images from directories.

    ...

    Attributes
    ----------
    dir_name : str
        A directory with images.

    Methods
    -------
    get_one_image(labels=None, new_size=None, path=None):
        Loads an image from self.dir_name.
    get_batch(start, stop, labels=None, new_size=None, shuffle=None):
        Loads a batch of images.
    rgb2gray(rgb):
       Returns a gray scaled image from rgb.
    sobel_image(im):
       Returns gradients of an image

    """
    def __init__(self, dir_name: str, img_ext=".jpg"):
        """
        Image loader initialization

        Parameters
        ----------
        dir_name: str
            Default directory name.
        img_ext: str, optional
            Defaults to ".jpg".

        """
        self.dir_name = dir_name
        self.img_ext = img_ext

    def get_one_image(self, labels: dict = None, new_size: tuple = None, path: str = None) -> dict:
        """
        Returns an image and it's label, if path is set, returns an image by this path,
        relative to self.dir_name, else performs random choice. If new_size is set, returns a resized image,
        if labels is set, returns an image with a label.

        Parameters
        ----------
        labels: dict, optional
            Class labels, where keys are label names and values are labels.
        new_size: tuple, optional
            Size of the returned image.
        path: str, optional
            Path of the needed image.

        Returns
        -------
        dict
            Output image is stored by the key "data", it's label is stored by the key "target".

        Raises
        ------
        Exception
            When cannot assign a label to image.

        """
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

    def rgb2gray(self, rgb: np.array) -> np.array:
        """
        Returns a gray scaled version of rgb image rgb.

        Parameters
        ----------
        rgb : np.array
            Input rgb image.

        Returns
        -------
        np.array
            Gray scaled image.

        """
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def sobel_image(self, im: np.array) -> np.array:
        """
        Returns gradients of image.

        Parameters
        ----------
        im: np.array
            Input gray scaled image.

        Returns
        -------
        np.array
            Gradients of the input image.

        """
        im = np.float32(im)
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        return np.uint8(mag)

    def get_batch(self, start: int, stop: int, labels: dict = None, new_size: tuple = None, shuffle: bool = None) -> \
            dict:
        """
        Returns a batch of images, which size is defined by start and stop with their labels,
        If new_size is set, returns resized images, if labels is set,
        returns the images with labels, if shuffle is set to True, shuffles files in a directory in certain order.

        Parameters
        ----------
        start: int
            Starting index of given files.
        stop: int
            Ending index of given files.
        labels: dict, optional
            Class labels, where keys are label names and values are labels.
        new_size: tuple, optional
            Size of the returned image.
        shuffle: bool
            If set to True, shuffle files in a directory according to random seed.

        Returns
        -------
        dict
            Output images are stored by the key "data", their labels are stored by the key "target".

        Raises
        ------
        Exception
            When cannot assign a label to image.

        """
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
    """
    A class for preprocessing images.

    ...

    Attributes
    ----------
    img_size: tuple
        Default image size for any preprocessing.

    Methods
    ------
    get_hog(img)
        Returns HOG of image.
    resize_hog(hog)
        Resizes HOG of image to a needed size.
    simple_preproc(img)
        Converts image to a vector, then normalizes.
    """
    def __init__(self, img_size: tuple):
        """
        Initializes image size for preprocessing

        Parameters
        ----------
        img_size : tuple
            Sets default image size to img_size.
        """
        self.img_size = img_size

    def get_hog(self, img: np.array) -> np.array:
        """
        Performs HOG on input image.

        Parameters
        ----------
        img: np.array
            Input image.

        Returns
        -------
        np.array
            HOG of img.
        """
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

    def resize_hog(self, hog: np.array) -> np.array:
        return np.reshape(hog, (hog.shape[0], ))

    def simple_preproc(self, img: np.array) -> np.array:
        """
        Flattens an input image to a vector with values from 0 to 1.

        Parameters
        ----------
        img : np.array
            Gray scaled image.

        Returns
        -------
        np.array
            A vector with size of (img.shape[0]*img.shape[1], ) and float values in range (0, 1).

        """
        return np.resize(img, (img.shape[0]*img.shape[1], ))/255.0







