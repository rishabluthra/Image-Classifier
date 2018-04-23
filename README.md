# Image Classifier
![A screenshot of the user interface](https://github.com/SuparSquid/Image-Classifier/blob/master/screenshot.PNG?raw=true)<br />
This is a simple image classifier built using Tensorflow that is attached to a Flask server and a Vue user interface that can upload images and receive results from the trained model.


#### To use the classifier
1. Put the sorted training images into their labeled folders in the training_images folder
2. Edit the training categories in line 19 of train.py
3. Run train.py
4. To predict, start server.py and open index.html

#### The classifier contains
- 2 Convolutional layers
- 1 Flattening layer
- 1 Dropout layer
- 2 Dense layers
