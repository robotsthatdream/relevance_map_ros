#!/usr/bin/env python

import rospy as ros
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import Model
from keras import backend
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from relevance_map.srv import *

def compute_features(req) :
    print("generate feature ...")
    base_model = VGG16(weights='imagenet', include_top=False)

    pooling = MaxPooling2D(pool_size=(8,8), dtype='float32')(base_model.layers[-2].output)
    model = Model(inputs=base_model.input,outputs=pooling)
    # model2 =  Model(inputs=model.input,outputs=model.layers[9].output)
    # print(model2.summary())

    converter = CvBridge()

    try:
        img = converter.imgmsg_to_cv2(req.supervoxels[0],"rgb8")
    except CvBridgeError as e :
        print(e)
    images = np.zeros((len(req.supervoxels),
        np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]))
    images[0] = img
    for i in range(1,len(req.supervoxels)) :
        try:
            img = converter.imgmsg_to_cv2(req.supervoxels[i],"rgb8")
        except CvBridgeError as e :
            print(e)
        images[i] = img

    images = np.array(images,dtype='float32')

    print(np.shape(images))
    # x = np.expand_dims(x, axis=0)
    images = preprocess_input(images)

    features = model.predict(images)

    dim = np.shape(features)[-1]
    print("dim",dim)

    for i in range(0,np.shape(features)[0]) :
        features[i] = features[i]/np.amax(features[i])


    print("Sending feature.")

    backend.clear_session()

    resp = cnn_featuresResponse() 
    resp.features = features.ravel()
    resp.dimension = dim

    return resp

def main() :
    ros.init_node('cnn_features_server')
    service = ros.Service('cnn_features',cnn_features,compute_features)
    print("Service ready !")
    ros.spin()

if __name__ == '__main__':
    main()