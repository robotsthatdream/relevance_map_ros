## relevance_map_ros
Library to facilitate implementation of ros node for relevance map building, exploitation and, viewing.
This code have been used for the following articles : 
	- Building an Affordances Map with Interactive Perception LK Le Goff, O Yaakoubi, A Coninx, S Doncieux - arXiv preprint arXiv:1903.04413, 2019
	- Bootstrapping Robotic Ecological Perception from a Limited Set of Hypotheses Through Interactive Perception LK Le Goff, G Mukhtar, A Coninx, S Doncieux - arXiv preprint arXiv:1901.10968, 2019

---

## How to Install :

# Install first :
- image_processing library https://github.com/robotsthatdream/image_processing
- cmm library  https://github.com/LeniLeGoff/CMM_Lib

# Installation :

After cloning this repository into your catkin workspace,
follow the command lines :
<pre>
cd < catkin workspace >
catkin_make -DCATKIN_WHITE_LIST=relevance_map install
</pre>

# Get Started :

You can find in the examples folder two examples :
	- examples_node : a node to compute a relevance map from an already trained classifier.
		https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/example_node.cpp
		To launch this node use the example_node.launch : https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/example_node.launch.
	- train_classifier : a node to train a classifier to produce a relevance map.
		https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/train_classifier.cpp
		To launch this node use the example_node.launch : https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/train_classifier.launch.

Both nodes need a topic of rgb images and a topic of depth images.

The relevance map could be computed based on different visual features extracted on a supervoxel extraction. 
Three are proposed in this implementation : 
	- centralFPFH : FPFH is a common descriptor that characterizes shape based on a pointcloud of normals (Rusu et al. [2009]). In this implementation, FPFH is extracted on the central point of the pointcloud including the targeted supervoxel and its neighbors. The radius of neighborhood to compute FPFH is set to the size of a supervoxel, thus the central point FPFH takes into account the whole considered pointcloud. The central point is the centroid of the targeted supervoxel. The dimensionality is 33.
	- colorLabHist : Color histogram of a supervoxel extracted on CIELab color encoding. One histogram of 5 bins is computed separately on each dimension (L ∗ a ∗ b) and then they are concatenated. This feature space has 15 dimensions.
	- centralFPFHLabHist : The concatenation of colorLabHist and centralFPFH which has 48 dimensions.

To change the feature you must modify the parameters modality and dimension in the launch file of the node.
These features are implemented in the image_processing library available here : https://github.com/robotsthatdream/Lib_image_processing/blob/master/include/image_processing/features.hpp

In folder examples/gmm_archives, you can find archives of already trained classifiers.
This classifiers have been trained with the feature centralFPFHLabHist, so, they must be used with this feature to compute a relevance map.