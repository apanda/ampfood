import numpy as np
from numpy.linalg import norm
import cv2
import sys
if len(sys.argv) != 4:
    print >>sys.stderr, "%s tagfile model mode"%(sys.argv[0])
    sys.exit(1)

def TrainDumbImages ():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    for k, v in tags.iteritems():
        try:
            images.append(cv2.imread(k, cv2.CV_LOAD_IMAGE_GRAYSCALE).flatten())
            labels.append([0 if v == '-' else 1])
        except:
            print "Failed on %s"%(k)
            pass
    images = np.float32(images)
    labels = np.float32(labels)
    svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=0.001) # Should cross validate
    svm = cv2.SVM()
    svm.train(images, labels, params = svm_params)
    svm.save(sys.argv[2])


def TestDumbImages():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    imgatidx = []
    for k, v in tags.iteritems():
        imgatidx.append(k)
        images.append(cv2.imread(k, cv2.CV_LOAD_IMAGE_GRAYSCALE).flatten())
        labels.append([0 if v == '-' else 1])
    images = np.float32(images)
    labels = np.float32(labels)
    svm = cv2.SVM()
    svm.load(sys.argv[2])
    results = svm.predict_all(images)
    wrong = 0
    total = 0
    for i in xrange(0, len(results)):
        total += 1
        if results[i] != labels[i]:
            wrong += 1
            print "Did not match %s is marked as %f I said %f"%(imgatidx[i], results[i][0], labels[i][0])
    print "Got %d of %d wrong"%(wrong, total)

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

def TrainHogImages ():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    hog = cv2.HOGDescriptor()
    done = 0
    for k, v in tags.iteritems():
        try:
            im = cv2.imread(k, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            #features = hog.compute(im).reshape(-1)
            features = ComputeHog (im)
            done += 1
            images.append(features)
            labels.append([0 if v == '-' else 1])
            if done % 50 == 0:
                print "Size of vector %d"%(len(features))
                print "Done with %d"%(done)
        except:
            print "Failed on %s"%(k)
            pass
    print "Converting to np array"
    images = np.float32(images)
    print images
    labels = np.float32(labels)
    print "Training SVM"
    svm_params = dict(kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, C=0.001) # Should cross validate
    svm = cv2.SVM()
    svm.train(images, labels, params = svm_params)
    svm.save(sys.argv[2])
        

def TestHogImages():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    imgatidx = []
    hog = cv2.HOGDescriptor()
    done = 0
    for k, v in tags.iteritems():
        imgatidx.append(k)
        im = cv2.imread(k, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #features = hog.compute(im).reshape(-1)
        features = ComputeHog (im)
        done += 1
        images.append(features)
        labels.append([0 if v == '-' else 1])
        if done % 50 == 0:
            print "Size of vector %d"%(len(features))
            print "Done with %d"%(done)
    images = np.float32(images)
    labels = np.float32(labels)
    svm = cv2.SVM()
    svm.load(sys.argv[2])
    results = svm.predict_all(images)
    wrong = 0
    total = 0
    for i in xrange(0, len(results)):
        total += 1
        if results[i] != labels[i]:
            wrong += 1
            print "Did not match %s is marked as %f I said %f"%(imgatidx[i], results[i][0], labels[i][0])
    print "Got %d of %d wrong"%(wrong, total)
mode = sys.argv[3]
if mode == 'train_dumb':
    TrainDumbImages ()
elif mode == 'validate_dumb':
    TestDumbImages ()
elif mode == 'train_hog':
    TrainHogImages ()
elif mode == 'validate_hog':
    TestHogImages ()
else:
    print "Unknown mode must be train_dumb or validate_dumb or train_hog or validate_hog"
