import numpy as np
from numpy.linalg import norm
import cv2
from StringIO import StringIO

def LoadSVM (filename):
    svm = cv2.SVM()
    svm.load(filename)
    return svm

def RunDumbDetector (imbuf, svm):
    try:
        img = np.float32(cv2.imdecode(np.asarray(bytearray(imbuf), dtype = np.uint8),\
                                        cv2.CV_LOAD_IMAGE_GRAYSCALE).flatten())
        return svm.predict(img)
    except:
        return -1

def ComputeHog (img):
    # Do edge detection.
    height = len(img)
    width = len(img[0])
    features = []
    window_size = 8
    stride = 4
    for y in xrange(0, height - window_size, stride):
        for x in xrange(0, width - window_size, stride):
            wnd = img[y:y+window_size, x:x+window_size]
            # Do edge detection.
            gx = cv2.Sobel(wnd, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(wnd, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            # Bin the angles.
            # TODO: we might want to try different rotations of the image.
            bin_n = 9
            bin = np.int32(bin_n*ang/(2*np.pi))
            # the magnitudes are used as weights for the gradient values.
            hist = np.bincount(bin.ravel(), mag.ravel(), bin_n)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps
            features.extend(hist)
    return np.float32(features)

def RunHogDetector (imbuf, svm):
    try:
        img = cv2.imdecode(np.asarray(bytearray(imbuf), dtype = np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        features = ComputeHog(img)
        return svm.predict(features)
    except:
        return -1
