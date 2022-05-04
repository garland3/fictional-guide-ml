from flask import Flask, render_template, Response, make_response
import cv2
import base64

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()


