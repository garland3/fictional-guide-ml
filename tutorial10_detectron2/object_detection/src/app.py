from flask import Flask, Response, render_template
import requests
import imageio
import io
from lib.run_detection import get_image_from_host, setup_config, make_predictor, run_visualizer, predict_img
from PIL import Image

app = Flask(__name__)
cfg = setup_config()
pred = make_predictor(cfg)

def img_to_buffer(img):
    img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def pred_wrapper():
    img = imageio.imread(io.BytesIO(get_image_from_host()))
    out_predictions = predict_img(pred, img)
    img_viz = run_visualizer(out_predictions,img, cfg)
    buf = img_to_buffer(img_viz)
    return buf

@app.route("/s")
def stream():
    def s():
        while True: 
            buf = pred_wrapper()
            r = b"--frame\r\n"+ \
                b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            yield r
    return Response(s(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "Hello World Test"

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port = 8000)