from flask import Flask, render_template, Response, make_response
import cv2
import base64

app = Flask(__name__)

# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
camera = cv2.VideoCapture(0)

#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames(just_frame = False):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if just_frame: return frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/s')
def video_feed_single():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='image/jpeg')

@app.route('/r')
def capture_api():
#   with open("house-thumbs-up.gif", "rb") as f:
    success, frame = camera.read()  # read the camera frame
    ret, buffer = cv2.imencode('.jpg', frame)
    image_binary = buffer.tobytes()
    # image_binary = gen_frames(just_frame=True)

    # response = make_response(base64.b64encode(image_binary))
    response = make_response(image_binary)

    response.headers.set('Content-Type', 'image/jpg')
    response.headers.set('Content-Disposition', 'attachment', filename='image.jpg')
    return response


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)