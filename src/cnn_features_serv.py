#!/usr/bin/env python

import rospy as ros
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import Model
import numpy as np
import matplotlib.pyplot as plt


def compute_features(req) :

    model = VGG16(weights='imagenet', include_top=False)

    pooling = MaxPooling2D(pool_size=(4,4), dtype='float32')(model.layers[-1].output)
    model = Model(inputs=model.input,outputs=pooling)
    # model2 =  Model(inputs=model.input,outputs=model.layers[9].output)
    # print(model2.summary())

    x = image.img_to_array(req.supervoxel)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x).ravel()

def main() :
    ros.init_node('cnn_features_server')
    service = ros.Service('cnn_features',cnn_features,compute_features)
    ros.spin()

if __name__ == '__main__':
    main()