from flask import Flask, make_response, send_file, render_template, redirect, url_for
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
label_file = open('server_labels.txt', 'a+')
imbuf_lock = threading.Lock()
svm = None
svm_lock = threading.Lock()
SVM_FILE = "svm_model"
update_file = open('indicate_wrong.txt', 'a+')
update_lock = threading.Lock()

def ReloadSVM ():
   global svm_lock
   global svm
   with svm_lock:
       svm = LoadSVM(SVM_FILE) 

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
       global imlabel
       global label_file
       global svm_lock
       global svm
       with imbuf_lock:
           imbuf = imbuf_new.getvalue()
           imfname = curtime
           with svm_lock:
               imlabel = RunDumbDetector (imbuf, svm) 
               print imlabel
               print >>label_file, "%s %f"%(imfname, imlabel)
               label_file.flush()
   def ReadLoop ():
       while True:
           try:
               ReadFunc()
           except:
               pass
           time.sleep(delay) 
   ReloadSVM()
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
    global imlabel
    fname = None
    label = None
    with imbuf_lock:
        fname = imfname
        label = imlabel
    print fname
    print "Label reported as %f"%(label)
    return render_template('latest.html', fname=fname, label="Food" if label != 0.0 else "No Food")

@app.route('/correct/<name>')
def correct_img (name):
    global update_lock
    global update_file
    with update_lock:
        print "Updating"
        print >>update_file, name
        update_file.flush()
    return redirect(url_for('latest_page'))

@app.route('/reload')
def reload_svm ():
    ReloadSVM ()
    return "Reloaded"

@app.route('/boot')
def Boot ():
    ReadThread(120.0)
    return "Booted"

Boot()
app.run(host='0.0.0.0')
