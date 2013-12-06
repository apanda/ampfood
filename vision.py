'''
VM and KNearest digit recognition.

Borrowed from the sample file here: https://github.com/Itseez/opencv/blob/master/samples/python2/digits.py and modified for our purposes.

Sample loads a dataset of handwritten digits from 'digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   digits.py
'''

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import numpy as np
from numpy.linalg import norm
import os

# local modules
from common import clock, mosaic

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
DIGITS_FN = 'data/digits.png'
TRAINING_DIR = '/mnt/images/training'
LABELS = '/mnt/images/labeled_uptp1386267400'

def load_training(training_dir):
    print 'loading files from "%s" ...' % training_dir
    filenames = os.listdir(TRAINING_DIR)

    full_filenames = [os.path.join(TRAINING_DIR, f) for f in filenames]
    # the second parameter of imread tells it to read the image in grayscale
    images = [cv2.imread(f, 0) for f in full_filenames if os.path.isfile(f)]
    print 'found %d image files' % len(images)

    positive_labels = set()
    RANGE_REGEX = re.compile('(\d*) - (\d*)')
    SINGLE_REGEX = re.compile('\d*')
    for line in open(LABELS, 'r'):
        match = RANGE_REGEX.match(line)
        if match:
            start = match.group(1)
            end = match.group(2)
            for i in range(start, end + 1):
                positive_labels.add('%s.png' % i)
        else:
            single_match = SINGLE_REGEX.match(line)
            if single_match:
                positive_labels.add('%s.png' % single_match.group())
# TODO: do this in a diferent way! just linearly add to labels; add for gap between this
                start and the previous end.

    print 'num positive labels: %s' % positive_labels.size()
    labels = numpy.zeros((len(images)))
    total_pos = 0
    for i, name in enumerate(filenames):
        if name in positive_labels:
            labels[i] = 1
            total_pos += 1
    print 'positive labels added (should be same as above!!): %s' % positive_labels.size()

    return numpy.array(images), labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, images, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

    vis = []
    for img, flag in zip(images, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(images):
    return np.float32(images).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(images):
    samples = []
    for img in images:
        # Do edge detection.
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        # Bin the angles.
        # TODO: we might want to try different rotations of the image.
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        # the magnitudes are used as weights for the gradient values.
        hist = np.bincount(bin.ravel(), mag.ravel(), bin_n)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print __doc__

    images, labels = load_training(DIGITS_FN)

    print 'preprocessing...'
    # shuffle images
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(images))
    images, labels = images[shuffle], labels[shuffle]

    images2 = map(deskew, images)
    samples = preprocess_hog(images2)

    train_n = int(0.9*len(samples))
    cv2.imshow('test set', mosaic(25, images[train_n:]))
    images_train, images_test = np.split(images2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print 'training KNearest...'
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, images_test, samples_test, labels_test)
    cv2.imshow('KNearest test', vis)

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, images_test, samples_test, labels_test)
    cv2.imshow('SVM test', vis)
    print 'saving SVM as "images_svm.dat"...'
    model.save('images_svm.dat')

    cv2.waitKey(0)
