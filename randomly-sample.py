from os import walk
import os
import sys
import math
import random
from shutil import copy2
if len(sys.argv) != 8:
    print >>sys.stderr, "%s food object empty training_target validation_target tag_file gtruth"%(sys.argv[0])
    sys.exit(1)
food_dir = sys.argv[1]
object_dir = sys.argv[2]
empty = sys.argv[3]
training_target = sys.argv[4]
validation_target = sys.argv[5]
tag_file = open(sys.argv[6], 'w+')
gtruth = open(sys.argv[7], 'w+')
sample_size = 0

for root, dirs, files in walk(food_dir):
   files = list(files)
   files = filter(lambda s: s.endswith('.png'), files)
   nfiles = len(files)
   sample_size = int(math.floor(0.5 * float(nfiles)))
   print "Found a total of %d food files sampling %d"%(nfiles, sample_size)
   sample = random.sample(files, sample_size) 
   for s in sample:
       copy2(os.path.join(root, s), training_target)
       print >>tag_file, "%s +"%(s)
   left_over = list(set(files) - set(sample))
   for s in left_over:
       copy2(os.path.join(root, s), validation_target)
       print >>gtruth, "%s +"%(s)
   break

# Now choose as many negative and table examples as the original
for root, dirs, files in walk(object_dir):
    files = list(files)
    nfiles = len(files)
    samp_size = min(sample_size, nfiles)
    print "Found a total of %d other files sampling %d"%(nfiles, samp_size)
    sample = random.sample(files, samp_size)
    for s in sample:
        copy2(os.path.join(root, s), training_target)
        print >>tag_file, "%s -"%(s)
    left_over = list(set(files) - set(sample))
    valid_size = min(sample_size, len(left_over))
    training_set = random.sample(left_over, valid_size)
    for s in training_set:
        copy2(os.path.join(root, s), validation_target)
        print >>gtruth, "%s -"%(s)
    break


for root, dirs, files in walk(empty):
    files = list(files)
    nfiles = len(files)
    samp_size = min(sample_size, nfiles)
    print "Found a total of %d empty files sampling %d"%(nfiles, samp_size)
    sample = random.sample(files, samp_size)
    for s in sample:
        copy2(os.path.join(root, s), training_target)
        print >>tag_file, "%s -"%(s)
    left_over = list(set(files) - set(sample))
    valid_size = min(sample_size, len(left_over))
    training_set = random.sample(left_over, valid_size)
    for s in training_set:
        copy2(os.path.join(root, s), validation_target)
        print >>gtruth, "%s -"%(s)
    break
