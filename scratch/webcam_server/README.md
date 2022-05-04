# Thanks!
* this code modified from https://github.com/miguelgrinberg/flask-video-streaming/tree/v1

# goal
* setup a webserver on the host so that I can stream images to my docker container. 
* helpful for me since I"m running windows as my host, but I want to run linux to process some live images. 

# setup
```
conda create --name webcam python=3.9
conda env export --no-builds > my_env.yml
conda activate webcam
```

OR
```
conda install flask
conda install -c conda-forge opencv 

```
