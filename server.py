# Importing all the libraries for the server.
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
import tensorflow as tf
import numpy as np 
import os, glob, cv2 
import sys,argparse 

app = Flask(__name__)

# Setting the directory for the uploaded images.
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = '.\static\img'
configure_uploads(app, photos)

# The function that gets the requests from my front end and prepares it for prediction.
@app.route('/classify/', methods=['POST'])
def hello():
    json = {}
    print(request.files)
    if 'image' in request.files:
        filename = photos.save(request.files['image'])
        
        path = app.config['UPLOADED_PHOTOS_DEST'] + '\\' + filename
        print(path)
    
    return jsonify(str(score(path))), 201


# Function that scores the images based on the trained model.
def score(image_path):
    image_path=image_path 
    filename = image_path
    image_size=128
    num_channels=3 
    images = []
    
    # Reading and resizing the images.
    image = cv2.cv2.imread(filename)
    image = cv2.cv2.resize(image, (image_size, image_size),0,0, cv2.cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    x_batch = images.reshape(1, image_size,image_size,num_channels)

    sess = tf.Session()

    # Importing the trained model.
    saver = tf.train.import_meta_graph('image-model.meta')

    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Predicting the uploaded image using the graph of the model.
    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")

    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 5))


    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)

    # Returning the results based on the probabilities.
    for i in result:
        result = ""
        if np.argmax(i) == 0:
            result = "This image is category (1), the classifier is "
        elif np.argmax(i) == 1:
            result = "This image is category (2), the classifier is"
        elif np.argmax(i) == 2:
            result = "This image is category (3), the classifier is"
        elif np.argmax(i) == 3:
            result = "This image is category  (4), the classifier is"
        elif np.argmax(i) == 4:
            result = "This image is category (5), lmao, the classifier is"
     
            
    response = result + " " + str(np.max(i) * 100) + "% sure"
       
    return response
  


app.run()
