from flask import Flask, make_response, send_file, render_template
app = Flask(__name__)
import curl
from StringIO import StringIO
from PIL import Image
import threading
import time
from svm_funcs import *  
imbuf = None
imfname = None
imlabel = None
imbuf_lock = threading.Lock()
svm = None
svm_lock = threading.Lock()
SVM_FILE = "SVM_MODEL"
def ReadThread (delay):
   def ReadFunc ():
       c = curl.Curl('http://128.32.132.63:5000/image')
       im = StringIO(c.get())
       im.seek(0)
       img = Image.open(im)
       imbuf_new = StringIO()
       curtime = '%d.png'%(time.time())
       img.save(imbuf_new, 'PNG')
       img.save(curtime, 'PNG')
       print "Saved %s"%(curtime)
       global imbuf
       global imbuf_lock
       global imfname
       with imbuf_lock:
           imbuf = imbuf_new.getvalue()
           imfname = curtime
           with svm_lock:
               imlabel = RunDumbDetector (imbuf, svm) 
   def ReadLoop ():
       while True:
           try:
               ReadFunc()
           except:
               pass
           time.sleep(delay) 
   with svm_lock:
       svm = LoadSVM(SVM_FILE) 
   try:
      ReadFunc()
   except:
      time.sleep(delay)
      ReadFunc()
   t = threading.Thread(target = ReadLoop)
   t.daemon = True
   t.start()

app = Flask(__name__)

@app.route('/image/latest.png')
def latest ():
    global imbuf
    global imbuf_lock
    with imbuf_lock:
        imbuf_io = StringIO(imbuf)
        return send_file(imbuf_io, mimetype='image/png', cache_timeout=0.0)
@app.route('/image/<name>')
def image (name):
    return send_file(name, mimetype='image/png')

@app.route('/latest.html')
def latest_page ():
    global imbuf_lock
    global imfname
    fname = None
    label = None
    with imbuf_lock:
        fname = imfname
        label = imlabel
    print fname
    return render_template('latest.html', fname=fname, label="Food" if label != 0.0 else "No Food")

def Boot ():
    ReadThread(120.0)
    return "Booted"

Boot()
app.run(host='0.0.0.0')
