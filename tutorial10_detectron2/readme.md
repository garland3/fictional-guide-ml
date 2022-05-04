# Objective
* run Facebook's detection ML on my webcam
    * Object detection
    * Bounding boxes on the objects. 
* assume that it can't be run in windows 11
    * stream my webcam with a flask webserver
    * read the stream in a docker container
    * run the object detection
    * output to a web interface the results
    * expose the port to the host so I can view the results in windows. 