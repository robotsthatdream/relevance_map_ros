#!/usr/bin/env python

import rospy as ros
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import Model
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from relevance_map.srv import *

def compute_features(req) :
    print("generate feature ...")
    model = VGG16(weights='imagenet', include_top=False)

    pooling = MaxPooling2D(pool_size=(8,8), dtype='float32')(model.layers[-2].output)
    model = Model(inputs=model.input,outputs=pooling)
    # model2 =  Model(inputs=model.input,outputs=model.layers[9].output)
    # print(model2.summary())

    converter = CvBridge()

    try:
        img = converter.imgmsg_to_cv2(req.supervoxel,"rgb8")
    except CvBridgeError as e :
        print(e)

    print(np.shape(img))
    x = np.array(img,dtype='float32')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = model.predict(x).ravel()
    print(len(feature))
    feature = feature/np.amax(feature)
    print("Sending feature.")

    return feature

def main() :
    ros.init_node('cnn_features_server')
    service = ros.Service('cnn_features',cnn_features,compute_features)
    print("Service ready !")
    ros.spin()

if __name__ == '__main__':
    main()