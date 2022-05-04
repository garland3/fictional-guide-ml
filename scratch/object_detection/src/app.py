from lib.prototype_app import  setup_predictor,make_prediction
from flask import Flask, render_template, Response
import requests
import imageio
import io
from PIL import Image

predictor,cfg = setup_predictor()
app = Flask(__name__)

def read_image_from_host():
    r = requests.get('http://host.docker.internal:5000/r')
    img_arr = imageio.imread(io.BytesIO(r.content))
    return img_arr
    
def img_to_buffer(img):
    im = Image.fromarray(img)
    # im.save(r"/workspaces/torch_tutorials_to_shared/scratch/object_detection/outputs/image_name.jpg")
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()

def prediction_wrapper():
    img = read_image_from_host()
    img_pred = make_prediction(img,predictor,cfg)
    img_buffer = img_to_buffer(img_pred)
    return img_buffer

@app.route('/s')
def single_frame():
    return Response(prediction_wrapper(), mimetype='image/jpeg')

@app.route('/v')    
def stream():
    def s():
        while True:
            r = b'--frame\r\n'+ \
                b'Content-Type: image/jpeg\r\n\r\n' +  prediction_wrapper() + b'\r\n'
            yield r
    # with open("buff.txt",'wb') as f:
    #     f.write(r)
    return Response(s(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')    
def index():
    return render_template('index.html')

# @app.route('/s')
# def video_feed_single():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='image/jpeg')

# @app.route('/r')
# def capture_api():
# #   with open("house-thumbs-up.gif", "rb") as f:
#     success, frame = camera.read()  # read the camera frame
#     ret, buffer = cv2.imencode('.jpg', frame)
#     image_binary = buffer.tobytes()
#     # image_binary = gen_frames(just_frame=True)

#     # response = make_response(base64.b64encode(image_binary))
#     response = make_response(image_binary)

#     response.headers.set('Content-Type', 'image/jpg')
#     response.headers.set('Content-Disposition', 'attachment', filename='image.jpg')
#     return response


# def index():
#     """Video streaming home page."""
#     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)