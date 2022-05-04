from flask import Flask, render_template, Response, make_response
import cv2
import base64
# import imageio

app = Flask(__name__)
# start the camera on initialization of the webapp

camera = cv2.VideoCapture(0)
def get_frame():    
    ret, frame = camera.read()
    return frame

def frame_to_buffer(img):
    ret, encoded_img = cv2.imencode(".jpg", img)
    buffer_data = encoded_img.tobytes()
    return buffer_data
    # return (b"--frame\r\n"
    #         b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def single_image():
    frame = get_frame()
    buffer_data = frame_to_buffer(frame)
    response = make_response(buffer_data)
    response.headers.set("Content-Type","image/jpg")
    response.headers.set("Content-Disposition", "attachment", filename="camera.jpg")
    return response

if __name__ == "__main__":
    app.run(debug=True)


