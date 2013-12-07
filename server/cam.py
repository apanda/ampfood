import pygame
import pygame.camera
import threading
import time
from PIL import Image
from StringIO import StringIO
from flask import Flask, send_file, abort
imbuf = None
imbuf_lock = threading.Lock()
def StartCameraLoop ():
    pygame.init()
    pygame.camera.init()
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], (960, 720), 'RGB')
    cam.start()
    cam.get_size()
    surf = pygame.Surface((960, 720))
    def gen_image ():
        cam.get_image(surf)
        global imbuf
        imbuf_new = StringIO()
        data = pygame.image.tostring(surf, 'RGBA')
        img = Image.fromstring('RGBA', cam.get_size(), data)
        img.save(imbuf_new, 'TIFF')
        with imbuf_lock:
            imbuf = imbuf_new.getvalue()
    def capture ():
        while True:
            try:
                gen_image()
            except:
                print "Failed to update image"
            time.sleep(2.0)
    gen_image()
    cap_thread = threading.Thread(target=capture)
    cap_thread.daemon = True
    cap_thread.start()

app = Flask(__name__)
@app.route('/image')
def GetImage ():
    global imbuf_lock
    global imbuf
    with imbuf_lock:
        #print "Found image"
        imbuf_io = StringIO(imbuf)
        return send_file(imbuf_io, mimetype='image/tiff', cache_timeout=10.0)

#@app.route('/boot')
def Boot ():
    StartCameraLoop()
    return "Booted"

Boot()
app.run(host='0.0.0.0')
