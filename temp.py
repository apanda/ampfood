
    height = len(img)
    width = len(img[0])
    features = []
    for y in xrange(0, stride, height - window_size):
        for x in xrange(0, stride, width - window_size):
            wnd = img[y:y+window_size, x:x+window_size]
        # Do edge detection.
