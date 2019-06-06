## relevance_map_ros
Library to facilitate implementation of ros node for relevance map building, exploitation and, viewing.

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

You can find in examples folder, example_node.cpp, a basic node which compute a relevance map from a video flow.
https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/example_node.cpp

To launch this node use the example_node.launch : https://github.com/robotsthatdream/relevance_map_ros/blob/master/examples/example_node.launch.

