#ifndef SCORE_COMPUTATION_HPP
#define SCORE_COMPUTATION_HPP

#include <tbb/tbb.h>
#include <image_processing/SurfaceOfInterest.h>
#include <relevance_map/utilities.hpp>

namespace relevance_map {

template <class Node>
class score_computation{
    public:
        score_computation(Node* node, std::string output_file) :
            _node(node), _output_file(output_file){
            _weights = _node->get_soi().get_weights()[_node->get_modality()];
            for(const auto& w : _weights)
                _lbls.push_back(w.first);
        }
        score_computation(score_computation& sc, tbb::split) :
            _node(sc._node), _weights(sc._weights), _lbls(sc._lbls),
            _tp(0), _tn(0), _fp(0), _fn(0), _total_neg(0), _total_pos(0){}

        void operator ()(const tbb::blocked_range<size_t>& r){
            double tp = _tp, fp = _fp, fn = _fn, tn = _tn, total_neg = _total_neg, total_pos = _total_pos;
            double w;
            uint32_t lbl;
            for(size_t i = r.begin(); i < r.end(); ++i){
                w =_weights[_lbls[i]];
                lbl = _lbls[i];
                bool is_in_back = is_in_cloud(_node->get_soi().getSupervoxels().at(lbl)->centroid_,_node->get_background());
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
        void join(const score_computation& sc){
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

    void compute_scores_results(int iter, int nb_false_pos = 0, int nb_false_neg = 0){
        std::chrono::system_clock::time_point timer;
        timer  = std::chrono::system_clock::now();
        //* Compute scores of performance
        double precision, recall, accuracy;
        compute_precision_recall(precision,recall,accuracy);

        int nb_samples, nb_neg, nb_pos, nb_pos_comp, nb_neg_comp,rand_nb_pos, rand_nb_neg;
        if(_node->get_method() == "nnmap"){
            nb_samples = _node->_nnmap_class[_node->get_modality()].get_samples().size();
            nb_neg = _node->_nnmap_class[_node->get_modality()].get_samples().get_data(0).size();
            nb_pos = _node->_nnmap_class[_node->get_modality()].get_samples().get_data(1).size();
            nb_pos_comp = 0;
            nb_neg_comp = 0;
        }
        else if(_node->get_method() == "gmm"){
            nb_samples = _node->_gmm_class[_node->get_modality()].get_samples().size();
            nb_neg = _node->_gmm_class[_node->get_modality()].get_samples().get_data(0).size();
            nb_pos = _node->_gmm_class[_node->get_modality()].get_samples().get_data(1).size();
            nb_pos_comp = _node->_gmm_class[_node->get_modality()].model()[1].size();
            nb_neg_comp = _node->_gmm_class[_node->get_modality()].model()[0].size();
        }
        else if(_node->get_method() == "mcs"){
            nb_samples = _node->_mcs.get_nb_samples();
            nb_neg = _node->_mcs.get_samples().get_data(0).size();
            nb_pos = _node->_mcs.get_samples().get_data(1).size();
            nb_pos_comp = 0;
            nb_neg_comp = 0;
        }
        rand_nb_pos = nb_samples*_total_pos/(_total_pos+_total_neg);
        rand_nb_neg = nb_samples - rand_nb_pos;

        ROS_INFO_STREAM("--------------------------------------------------------");
        ROS_INFO_STREAM("scores for iteration " << iter << "\n"
                        << " precision : " << precision << "\n"
                        << " recall : " << recall << "\n"
                        << " accuracy : " << accuracy << "\n"
                        << " nb components : pos " << nb_pos_comp << "; neg " << nb_neg_comp << "\n"
                        << " nb samples : pos " << nb_pos << "; neg " << nb_neg << "\n"
                        << " false samples : pos " << nb_false_pos << "; neg " << nb_false_neg << "\n"
                        << " rand nb : pos " << rand_nb_pos << " ; neg " << rand_nb_neg);
        ROS_INFO_STREAM("--------------------------------------------------------");

        std::stringstream str;
        str << "iteration_" << iter;
        _result = std::make_pair(str.str(),
                         result_array_t{{nb_samples,
                                               precision,
                                               recall,
                                               accuracy,
                                               nb_pos,
                                               nb_neg,
                                               nb_pos_comp,
                                               nb_neg_comp,
                                               nb_false_pos,
                                               nb_false_neg,
                                               rand_nb_pos,
                                               rand_nb_neg}});

        if(!write_result(_output_file))
            ROS_ERROR_STREAM("unable to open " << _output_file);

        ROS_INFO_STREAM("Computing scores pra finish, time spent : "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now() - timer).count());

    }

    void compute_precision_recall(double& precision, double& recall, double& accuracy){
        ROS_INFO_STREAM("compute precision, recall and accuracy");
        precision = 0; recall = 0; accuracy = 0;

        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_node->get_soi().get_weights()[_node->get_modality()].size()),*this);


        precision = _tp / (_tp + _fp);
        recall = _tp / (_tp + _fn);
        accuracy = (_tp/_total_pos + _tn/_total_neg) / 2.;
    }

    int write_result(std::string file_name){
        ROS_INFO_STREAM("start to write output file");
        std::ofstream ofs(file_name,std::ofstream::out | std::ofstream::app);
        if(!ofs.is_open())
            return 0;

        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << _result.first << YAML::Value
                << YAML::BeginMap
                << YAML::Key << "nbr_samples" << YAML::Value << ((int)_result.second[0])
                << YAML::Key << "precision" << YAML::Value << _result.second[1]
                << YAML::Key << "recall" << YAML::Value << _result.second[2]
                << YAML::Key << "accuracy" << YAML::Value << _result.second[3]
                << YAML::Key << "pos_samples" << YAML::Value << _result.second[4]
                << YAML::Key << "neg_samples" << YAML::Value << _result.second[5]
                << YAML::Key << "pos_components" << YAML::Value << _result.second[6]
                << YAML::Key << "neg_components" << YAML::Value << _result.second[7]
                << YAML::Key << "false_positives" << YAML::Value << _result.second[8]
                << YAML::Key << "false_negatives" << YAML::Value << _result.second[9]
                << YAML::EndMap;


        emitter << YAML::EndMap;

        ofs << emitter.c_str();
        ofs << "\n";
        ofs.close();
        return 1;
    }

    private:
        Node* _node;
        ip::SurfaceOfInterest::saliency_map_t _weights;
        std::vector<uint32_t> _lbls;
        std::pair<std::string,result_array_t> _result;
        std::string _output_file;
        std::string _soi_method;
        std::string _modality;
};

}

#endif //SCORE_COMPUTATION_HPP
