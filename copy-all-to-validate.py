from os import walk
import os
import sys
import math
import random
from shutil import copy2
if len(sys.argv) != 6:
    print >>sys.stderr, "%s food object empty validation_target gtruth"%(sys.argv[0])
    sys.exit(1)
food_dir = sys.argv[1]
object_dir = sys.argv[2]
empty = sys.argv[3]
validation_target = sys.argv[4]
gtruth = open(sys.argv[5], 'w+')

for root, dirs, files in walk(food_dir):
   files = list(files)
   files = filter(lambda s: s.endswith('.png'), files)
   nfiles = len(files)
   for s in files:
       copy2(os.path.join(root, s), validation_target)
       print >>gtruth, "%s +"%(s)
   break

# Now choose as many negative and table examples as the original
for root, dirs, files in walk(object_dir):
    files = list(files)
    nfiles = len(files)
    for s in files:
        copy2(os.path.join(root, s), validation_target)
        print >>gtruth, "%s -"%(s)
    break


for root, dirs, files in walk(empty):
    files = list(files)
    nfiles = len(files)
    for s in files:
        copy2(os.path.join(root, s), validation_target)
        print >>gtruth, "%s -"%(s)
    break
