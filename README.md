# Tensorflow-Basics
My practice sessions with Tensorflow in the Jupyter Notebook
TensorFlow is awesome and so are you, get on with it. This repository contains the basic implementation of tensorflow. 

I did sentiment analysis using tensorflow, demonstrated the processing of the text file that is required here:  https://git.io/f450x

After this, I have loaded the data and fed it to the neural network, it contains a snapshot of how Tensorboard will look like, code: https://git.io/f45ET

For real time object detection, I have used cv2 for capturing and synthesizing with the code
The tricky part in this is to get protobuf right.
The thing which worked for me as a mac user is the command

export PATH:$PATH ...TO PROTOBUF FIL which can be downloaded from github according to your system and then compiling the files seperately, my system was not supporting "*.proto"

So I compiled the files seperately. I have made a .py file that can be run through the command window or terminal in mac.

Go to the Real Time Object Detection folder, models/research/object_detection through command window or terminal
Once you are there, python Real \ Time \ Object \ Detection \ using \ cv2 \ and \ Tensorflow.py, hit enter, and you can see the code working if you get the protobuf right.

Model that I have used is "ssd_mobilenet_v1_coco_2017_11_17" by google.



Go through the code if you are interested.
