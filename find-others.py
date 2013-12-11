from os import walk
import os
import sys
from shutil import move
if len(sys.argv) != 4:
    print >>sys.stderr, "%s <directory> <tag> <move-to>"%(sys.argv[0])
    sys.exit(1)
d = sys.argv[1]
tag = open(sys.argv[2])
tags = tag.readlines()
tag_list = []
target = sys.argv[3]
for tag in tags:
    if '-' in tag:
        tag = tag.split('-')
    else:
        tag = [tag, tag]
    tag = map(long, tag)
    tag_list.append(tag)
def MatchesTag (s):
    s = long(s)
    for tag in tag_list:
        if tag[0] <= s and s <= tag[1]:
            return True
    return False
for root, dirs, files in walk(d):
    for name in files:
        if name.endswith('.png'):
            s = name[:-4]
            if MatchesTag(s):
                src =  os.path.join(root, name)
                dest = os.path.join(target, name)
                print "%s -> %s"%(src, dest)
                move(src, dest)
    break
