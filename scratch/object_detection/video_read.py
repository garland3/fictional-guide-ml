import requests
# r = requests.get('http://127.0.0.1:5000/')
# r = requests.get('http://host.docker.internal:5000/video_feed')
r = requests.get('http://host.docker.internal:5000/r')


print(r.status_code,  r.headers['content-type'])

img_data = r.content
with open('imgs/image_name.jpg', 'wb') as handler:
    handler.write(img_data)

# localhost:5001