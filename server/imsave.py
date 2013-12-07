# IPython log file

get_ipython().magic(u'logstart imsave.py')
import pygame
import pygame.camera
pygame.init()
pygame.camera.init()
from PIL import Image
cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], (960, 720), 'RGB')
surf = pygame.Surface(cam.get_size())
cam.start()
cam.get_image(surf)
from StringIO import StringIO
c = StringIO()
data = pygame.image.tostring(surf, 'RGBA')
img = Image.fromstring('RGBA', cam.get_size(), data)
img.save(c, 'TIFF')
get_ipython().magic(u'logstop')
