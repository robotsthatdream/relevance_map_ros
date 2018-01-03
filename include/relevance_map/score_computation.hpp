#ifndef SCORE_COMPUTATION_HPP
#define SCORE_COMPUTATION_HPP

#include <tbb/tbb.h>
#include <image_processing/SurfaceOfInterest.h>

namespace relevance_map {

template <class Node>
class score_computation{
    public:
        score_computation(Node* node) : _node(node){
            _weights = _node->get_soi().get_weights()[_node->get_modality()];
            for(const auto& w : _weights)
                _lbls.push_back(w.first);
        }
        score_computation(_score_computation& sc, tbb::split) :
            _node(sc._node), _weights(sc._weights), _lbls(sc._lbls),
            _tp(0), _tn(0), _fp(0), _fn(0), _total_neg(0), _total_pos(0){}

        void operator ()(const tbb::blocked_range<size_t>& r){
            double tp = _tp, fp = _fp, fn = _fn, tn = _tn, total_neg = _total_neg, total_pos = _total_pos;
            double w;
            uint32_t lbl;
            for(size_t i = r.begin(); i < r.end(); ++i){
                w =_weights[_lbls[i]];
                lbl = _lbls[i];
                bool is_in_back = utilities::is_in_cloud(_node->get_soi().getSupervoxels().at(lbl)->centroid_,_node->get_background());
                if(!is_in_back && w >= _node->get_threshold())
                    tp += w;
                else if(is_in_back && w >= _node->get_threshold())
                    fp += w;
                else if(!is_in_back && w < _node->get_threshold())
                    fn += (1-w);
                else if(is_in_back && w < _node->get_threshold())
                    tn += (1-w);
                if(is_in_back)
                    total_neg += 1;
                else total_pos += 1;
            }
            _tp = tp;
            _tn = tn;
            _fp = fp;
            _fn = fn;
            _total_neg = total_neg;
            _total_pos = total_pos;
        }
        void join(const _score_computation& sc){
            _tp += sc._tp;
            _tn += sc._tn;
            _fp += sc._fp;
            _fn += sc._fn;
            _total_neg += sc._total_neg;
            _total_pos += sc._total_pos;
        }


        double _tp = 0;
        double _fp = 0;
        double _fn = 0;
        double _tn = 0;
        double _total_neg = 0;
        double _total_pos = 0;

    void compute_scores_results(){
        std::chrono::system_clock::time_point timer;
        timer  = std::chrono::system_clock::now();
        //* Compute scores of performance
        double precision, recall, accuracy;
        _compute_precision_recall(precision,recall,accuracy);

        int nb_samples, nb_neg, nb_pos, nb_pos_comp, nb_neg_comp;
        if(_soi_method == "nnmap"){
            nb_samples = _node->_nnmap_class[_modality].get_samples().size();
            nb_neg = _node->_nnmap_class[_modality].get_samples().get_data(0).size();
            nb_pos = _node->_nnmap_class[_modality].get_samples().get_data(1).size();
            nb_pos_comp = 0;
            nb_neg_comp = 0;
        }
        else if(_soi_method == "gmm"){
            nb_samples = _node->_gmm_class[_modality].get_samples().size();
            nb_neg = _node->_gmm_class[_modality].get_samples().get_data(0).size();
            nb_pos = _node->_gmm_class[_modality].get_samples().get_data(1).size();
            nb_pos_comp = _node->_gmm_class[_modality].model()[1].size();
            nb_neg_comp = _node->_gmm_class[_modality].model()[0].size();
        }
        else if(_soi_method == "mcs"){
            nb_samples = _node->_mcs.get_nb_samples();
            nb_neg = _node->_mcs.get_samples().get_data(0).size();
            nb_pos = _node->_mcs.get_samples().get_data(1).size();
            nb_pos_comp = 0;
            nb_neg_comp = 0;
        }

        ROS_INFO_STREAM("--------------------------------------------------------");
        ROS_INFO_STREAM("scores for iteration " << _counter_iter << "\n"
                        << " precision : " << precision << "\n"
                        << " recall : " << recall << "\n"
                        << " accuracy : " << accuracy << "\n"
                        << " nb components : pos " << nb_pos_comp << "; neg " << nb_neg_comp << "\n"
                        << " nb samples : pos " << nb_pos << "; neg " << nb_neg << "\n"
                        << " false samples : pos " << _nb_false_pos << "; neg " << _nb_false_neg);
        ROS_INFO_STREAM("--------------------------------------------------------");

        std::stringstream str;
        str << "iteration_" << _counter_iter;
        _results = std::make_pair(str.str(),
                         std::array<double,10>{{nb_samples,
                                               precision,
                                               recall,
                                               accuracy,
                                               nb_pos,
                                               nb_neg,
                                               nb_pos_comp,
                                               nb_neg_comp,
                                               _nb_false_pos,
                                               _nb_false_neg}});

        std::string output_file;
        cafer_core::ros_nh->getParam("experiment/soi/output_file",output_file);
        if(!utilities::write_results(output_file,_results))
            ROS_ERROR_STREAM("unable to open " << output_file);

        ROS_INFO_STREAM("Computing scores pra finish, time spent : "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now() - timer).count());

    }

    void compute_precision_recall(double& precision, double& recall, double& accuracy){
        ROS_INFO_STREAM("compute precision, recall and accuracy");
        precision = 0; recall = 0; accuracy = 0;

        _score_computation sc(this);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_soi.get_weights()[_modality].size()),sc);


        precision = sc._tp / (sc._tp + sc._fp);
        recall = sc._tp / (sc._tp + sc._fn);
        accuracy = (sc._tp/sc._total_pos + sc._tn/sc._total_neg) / 2.;
    }

    private:
        Node* _node;
        ip::SurfaceOfInterest::saliency_map_t _weights;
        std::vector<uint32_t> _lbls;

};

}

#endif //SCORE_COMPUTATION_HPP
