# https://stackoverflow.com/questions/20820602/image-send-via-tcp
#!/usr/bin/python
import socket
import cv2
import numpy

TCP_IP = 'localhost'
TCP_PORT = 5002

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = cv2.VideoCapture(0)
ret, frame = capture.read()

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
result, imgencode = cv2.imencode('.jpg', frame, encode_param)
data = numpy.array(imgencode)
stringData = data.tostring()

sock.send( str(len(stringData)).ljust(16));
sock.send( stringData );
sock.close()

decimg=cv2.imdecode(data,1)
cv2.imshow('CLIENT',decimg)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# # https://stackoverflow.com/questions/24366839/send-image-over-tcp-save

# from socket import *
# # from PIL import Image
# # import StringIO
# # import ImageGrab


# port =9999

# # print 'starting SERVER ... '
# sock = socket(AF_INET, SOCK_STREAM)
# sock.bind(('', port)) # port to listen on

# # print 'starting to listen...'
# sock.listen(SOMAXCONN)
# client, addr = sock.accept() # when accepting a connection, we get a tuple - two different                     variables with info about the new connection
# # print 'done listening!'
# # print 'client address is: ' + str(addr)


# buf=StringIO.StringIO()#create buffer    

# img=ImageGrab.grab()#take screenshot

# img.save(buf,format='PNG')#save screenshot to buffer

# client.sendall(buf.getvalue())

# sock.close()
# del sock