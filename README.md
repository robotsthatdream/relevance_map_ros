# relevance_map_ros
Library to facilate implementation of ros node for relevance map building, exploitation, viewing.

If you want just to generate plots from the data, is not needed to install the library.

---

# How to Install :

## Install first :
- image_processing library https://github.com/robotsthatdream/image_processing
- iagmm library  https://github.com/LeniLeGoff/IAGMM_Lib

## Installation :

After cloning this repository into your catkin worksapce,
follow the command lines :
<pre>
cd < catkin workspace >
catkin_make -DCATKIN_WHITE_LIST=relevance_map install
</pre>

## How To Generate Graphics from Data :
This graphics are the results presented in the publication : 
Discovering Objects from a Limited Set of Hypotheses Through Interactive Perception
Léni K. Le Goff, Ghanim Mukhtar, Alexandre Coninx, and Stephane Doncieux

For instance, to generate graphs for experiment in simulation env0_obj0 :
<pre>
cd python
python graph_pra.py ../data/simulation/env0_obj0/ classifier_eval.yml 200 3 10
python graph_pos_neg.py ../data/simulation/env0_obj0/ classifier_eval.yml 200
python graph_nbr_comp.py ../data/simulation/env0_obj0/ classifier_eval.yml 200 20
</pre>

graph_pra.py generate a performance graph which show precision, recall and accuracy.
graph_pos_neg.py generate a graph which shows the cumulative number of positives and negatives samples for each iteration of the experiment.
graph_nbr_comp.py generate a graph which show the number of components used in both Gaussian mixture models.
