import numpy as np
from numpy.linalg import norm
import cv2
import sys
import skimage.feature
if len(sys.argv) != 4:
    print >>sys.stderr, "%s tagfile model mode"%(sys.argv[0])
    sys.exit(1)

def TrainDumbImages ():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    for k, v in tags.iteritems():
        try:
            images.append(cv2.imread(k, 0).flatten())
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
    svm = cv2.SVM()
    svm.load(sys.argv[2])
    wrong = 0
    total = 0
    for k, v in tags.iteritems():
        image = np.float32(cv2.imread(k, 0).flatten())
        label = 0.0 if v == '-' else 1.0
        result = svm.predict(image)
        total += 1
        if result != label:
            wrong += 1
            print "%s SVM is %f GTRUTH is %f"%(k, result, label)
    print "Got %d of %d wrong"%(wrong, total)

def TrainHogImages ():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    hog = cv2.HOGDescriptor()
    done = 0
    for k, v in tags.iteritems():
        try:
            im = cv2.imread(k, 0)
            #features = hog.compute(im).reshape(-1)
            features = skimage.feature.hog(im, normalise=False)
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
    svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=0.001) # Should cross validate
    svm = cv2.SVM()
    svm.train(images, labels, params = svm_params)
    svm.save(sys.argv[2])
        

def TestHogImages():
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    hog = cv2.HOGDescriptor()
    done = 0
    print >>sys.stderr, "Loading SVM"
    svm = cv2.SVM()
    svm.load(sys.argv[2])
    print >>sys.stderr, "Done Loading SVM"
    wrong = 0
    total = 0
    for k, v in tags.iteritems():
        print >>sys.stderr, "Processing %s"%(k)
        im = cv2.imread(k, 0)
        #features = hog.compute(im).reshape(-1)
        features = np.float32(skimage.feature.hog(im, normalise=False))
        done += 1
        label = 0 if v == '-' else 1
        result = svm.predict(features)
        total += 1
        if result != label:
            wrong += 1
            print "Did not match %s SVM is %f GTRUTH is %f"%(k, result, label)
        elif done % 50 == 0:
            print >>sys.stderr, "Done with %d"%(done)
        print >>sys.stderr, "Done with %s"%(k)
    print "Got %d of %d wrong"%(wrong, total)

def TrainUsingFeatureImages (feature_func):
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    done = 0
    for k, v in tags.iteritems():
        try:
            im = cv2.imread(k, 0)
            #features = hog.compute(im).reshape(-1)
            features = np.float32(feature_func(im))
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
    #print images
    labels = np.float32(labels)
    print "Training SVM"
    svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=0.001) # Should cross validate
    svm = cv2.SVM()
    svm.train(images, labels, params = svm_params)
    svm.save(sys.argv[2])
        

def TestUsingFeatureImages (feature_func):
    tags = dict(map(str.split, map(str.strip, open(sys.argv[1]).readlines())))
    images = []
    labels = []
    done = 0
    print >>sys.stderr, "Loading SVM"
    svm = cv2.SVM()
    svm.load(sys.argv[2])
    print >>sys.stderr, "Done Loading SVM"
    wrong = 0
    total = 0
    for k, v in tags.iteritems():
        print >>sys.stderr, "Processing %s"%(k)
        im = cv2.imread(k, 0)
        #features = hog.compute(im).reshape(-1)
        features = np.float32(feature_func(im))
        done += 1
        label = 0 if v == '-' else 1
        result = svm.predict(features)
        total += 1
        if result != label:
            wrong += 1
            print "Did not match %s SVM is %f GTRUTH is %f"%(k, result, label)
        elif done % 50 == 0:
            print >>sys.stderr, "Done with %d"%(done)
        print >>sys.stderr, "Done with %s"%(k)
    print "Got %d of %d wrong"%(wrong, total)

mode = sys.argv[3]

def ShiVector (im):
    return np.float32(skimage.feature.corner_shi_tomasi(im)).flatten()
if mode == 'train_dumb':
    TrainDumbImages ()
elif mode == 'validate_dumb':
    TestDumbImages ()
elif mode == 'train_hog':
    TrainHogImages ()
elif mode == 'validate_hog':
    TestHogImages ()
elif mode == 'train_shi':
    TrainUsingFeatureImages (ShiVector)
elif mode == 'validate_shi':
    TestUsingFeatureImages (ShiVector)
else:
    print "Unknown mode must be train_dumb or validate_dumb or train_hog or validate_hog or " + \
          " train_shi or validate_shi"
