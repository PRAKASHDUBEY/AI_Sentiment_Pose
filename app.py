from flask import Flask, redirect, url_for, request, render_template, flash, Response
from werkzeug.utils import secure_filename
import SentimentModule as sm
import PoseModule as pm
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__, template_folder='./')

UPLOAD_FOLDER = 'static/uploads'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def home():
    return render_template('Input.html')


@app.route('/processingImage', methods=['POST'])
def image_frame():
    model = request.form['ImageModel']
    if model == "Sentiment":
        detector = sm.sentimentDetector()
    elif model == "Pose":
        detector = pm.poseDetector()
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cap = cv2.imread('./static/uploads/' + filename)
        img = detector.findC(cap)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.savefig('./static/uploads/' + filename)

        flash('Image successfully uploaded and displayed below')
        return render_template('Input.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)



@app.route('/processingVideo', methods=['POST'])
def video_frame():
    model = request.form['VideoModel']
    if model == "Sentiment":
        detector = sm.sentimentDetector()
    elif model == "Pose":
        detector = pm.poseDetector()
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        files = './static/uploads/' + filename
        return Response(video(files,detector), mimetype='multipart/x-mixed-replace; boundary=frame')

def video(files, detector):
    video = cv2.VideoCapture(files)
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            frame = detector.findC(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    model = request.form['LiveModel']
    if model == "Sentiment":
        detector = sm.sentimentDetector()
    elif model == "Pose":
        detector = pm.poseDetector()
    if request.method == 'POST':
        if request.form.get('action1') == 'LIVE':
            return Response(gen_frames(detector), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(detector):
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = detector.findC(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')










if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
