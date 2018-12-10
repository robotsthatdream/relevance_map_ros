#ifndef SCORE_COMPUTATION_HPP
#define SCORE_COMPUTATION_HPP


#include <tbb/tbb.h>
#include <image_processing/SurfaceOfInterest.h>
#include <relevance_map/utilities.hpp>
#include <chrono>
#include <tuple>

namespace relevance_map {

template <class Node>
class score_computation{
    public:

        typedef std::tuple<int,std::vector<std::vector<double>>> result_array_t;


        std::vector<std::string> statistics_name;
        score_computation(Node* node, std::string output_file) :
            _node(node), _output_file(output_file){
            init();
        }
        score_computation(Node* node, std::map<uint32_t,int> true_labels, std::string output_file) :
            _node(node), _output_file(output_file), _true_labels(true_labels){
            init();
        }
        score_computation(score_computation& sc, tbb::split) :
            _node(sc._node), _weights(sc._weights), _lbls(sc._lbls), _lbl(sc._lbl),
            _tp(0), _tn(0), _fp(0), _fn(0), _total_neg(0), _total_pos(0),
            _true_labels(sc._true_labels){}


        void init(){
            _lbls.clear();
            _weights.clear();
            _weights = _node->get_soi().get_weights()[_node->get_modality()];
            for(const auto& w : _weights)
                _lbls.push_back(w.first);
        }

        void operator ()(const tbb::blocked_range<size_t>& r){
            double tp = _tp, fp = _fp, fn = _fn, tn = _tn, total_neg = _total_neg, total_pos = _total_pos;
            double w;
            uint32_t lbl;
            std::map<uint32_t,int> true_labels = _true_labels;
            for(size_t i = r.begin(); i < r.end(); ++i){
                w =_weights[_lbls[i]][_lbl];
                lbl = _lbls[i];

                if(_lbl == true_labels.at(lbl) && w >= _node->get_threshold())
                    tp += w;
                else if(_lbl != true_labels.at(lbl) && w >= _node->get_threshold())
                    fp += w;
                else if(_lbl == true_labels.at(lbl) && w < _node->get_threshold())
                    fn += (1-w);
                else if(_lbl != true_labels.at(lbl)  && w < _node->get_threshold())
                    tn += (1-w);
                if(_lbl != true_labels.at(lbl))
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
        double _lbl = 0;

        void compute_scores_results(int iter, int nb_false_pos = 0, int nb_false_neg = 0){
            std::chrono::system_clock::time_point timer;
            timer  = std::chrono::system_clock::now();
            statistics_name = {"nb_samples","precision","recall","accuracy",
                               "nbr_samples","nbr_comp"};
//                               "pos_components","neg_components",
//                               "false_positives","false_negatives",
//                               "rand_nb_pos","rand_nb_pos"};
            //* Compute scores of performance
            std::vector<double> precision, recall, accuracy;
            compute_precision_recall(precision,recall,accuracy);

            int nb_samples ,rand_nb_pos, rand_nb_neg;
            std::vector<double> nbr_spl,nbr_comp;
            int nbr_class = 0;

            if(_node->get_method() == "nnmap"){
                nb_samples = _node->_nnmap_class[_node->get_modality()].get_samples().size();
                nbr_class = _node->_nnmap_class[_node->get_modality()].get_nbr_class();
                nbr_spl.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_spl[i] = _node->_nnmap_class[_node->get_modality()].get_samples().get_data(i).size();
                nbr_comp.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_comp[i] = 0;
            }
            else if(_node->get_method() == "gmm" || _node->get_method() == "composition"){
                nb_samples = _node->_gmm_class[_node->get_modality()].get_samples().size();
                nbr_class = _node->_gmm_class[_node->get_modality()].get_nbr_class();
                nbr_spl.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_spl[i] = _node->_gmm_class[_node->get_modality()].get_samples().get_data(i).size();
                nbr_comp.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_comp[i] = _node->_gmm_class[_node->get_modality()].model()[i].size();
            }
            else if(_node->get_method() == "mcs"){
                nb_samples = _node->_mcs.get_nb_samples();
                nbr_class = _node->_mcs.get_nbr_class();
                nbr_spl.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_spl[i] = _node->_mcs.get_samples().get_data(i).size();
                nbr_comp.resize(nbr_class);
                for(int i = 0; i < nbr_class; i++)
                    nbr_comp[i] = 0;
            }else ROS_ERROR_STREAM(_node->get_method() << " unknown relevance map method");
            rand_nb_pos = nb_samples*_total_pos/(_total_pos+_total_neg);
            rand_nb_neg = nb_samples - rand_nb_pos;





            std::stringstream str;
            str << "iteration_" << iter;

            std::vector<std::vector<double>> scores =  {precision,recall,accuracy,nbr_spl,nbr_comp};
            set_results(str.str(),std::make_tuple(nb_samples,scores));
            ROS_INFO_STREAM("--------------------------------------------------------");
            ROS_INFO_STREAM("scores for iteration " << iter << "\n"
                            << results_to_string());
            ROS_INFO_STREAM("--------------------------------------------------------");

            /*
                                                  {(double)nb_pos,
                                                   (double)nb_neg,
                                                   (double)nb_pos_comp,
                                                   (double)nb_neg_comp,
                                                   (double)nb_false_pos,
                                                   (double)nb_false_neg,
                                                   (double)rand_nb_pos,
                                                   (double)rand_nb_neg}));*/

            if(!write_result(_output_file))
                ROS_ERROR_STREAM("unable to open " << _output_file);

            ROS_INFO_STREAM("Computing scores pra finish, time spent : "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now() - timer).count());

        }

        void compute_precision_recall(std::vector<double>& precision, std::vector<double>& recall, std::vector<double>& accuracy){
            ROS_INFO_STREAM("compute precision, recall and accuracy");

            int nbr_class = _node->get_nbr_class();
            precision = std::vector<double>(nbr_class,0);
            recall = std::vector<double>(nbr_class,0);
            accuracy = std::vector<double>(nbr_class,0);

            for(int i = 0; i < nbr_class; i++){
                _lbl = i;
                _tp = 0; _tn = 0; _fn = 0; _fp = 0; _total_neg = 0; _total_pos = 0;

//                int w_size =  _node->get_soi().get_weights()[_node->get_modality()].size();

//                double w;
//                uint32_t lbl;
//                for(size_t i = 0; i < w_size; ++i){
//                    w =_weights[_lbls[i]][_lbl];
//                    lbl = _lbls[i];

//                    if(_lbl == _true_labels.at(lbl) && w >= _node->get_threshold())
//                        _tp += w;
//                    else if(_lbl != _true_labels.at(lbl) && w >= _node->get_threshold())
//                        _fp += w;
//                    else if(_lbl == _true_labels.at(lbl) && w < _node->get_threshold())
//                        _fn += (1-w);
//                    else if(_lbl != _true_labels.at(lbl)  && w < _node->get_threshold())
//                        _tn += (1-w);
//                    if(_lbl != _true_labels.at(lbl))
//                        _total_neg += 1;
//                    else _total_pos += 1;
//                }

                tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_node->get_soi().get_weights()[_node->get_modality()].size()),*this);

                precision[i] = _tp / (_tp + _fp);
                recall[i] = _tp / (_tp + _fn);
                accuracy[i] = (_tp/_total_pos + _tn/_total_neg) / 2.;
            }
        }

        int write_result(std::string file_name){
            ROS_INFO_STREAM("start to write output file");

            if(statistics_name.empty()){
                ROS_ERROR_STREAM("statistics_name vector is empty.");
                return 0;
            }
            if(std::get<1>(_result.second).empty()){
                ROS_ERROR_STREAM("result vector is empty.");
                return 0;
            }
            int nbr_class = _node->get_nbr_class();

            std::ofstream ofs(file_name,std::ofstream::out | std::ofstream::app);
            if(!ofs.is_open())
                return 0;

            YAML::Emitter emitter;
            emitter << YAML::BeginMap;
            emitter << YAML::Key << _result.first << YAML::Value
                    << YAML::BeginMap
                    << YAML::Key << statistics_name[0] << YAML::Value << std::get<0>(_result.second);
            for(int k = 0; k < std::get<1>(_result.second).size(); k++){
                emitter << YAML::Key << statistics_name[k+1] << YAML::Value << YAML::BeginSeq;
                for(int i = 0; i < nbr_class; i++){
                    emitter << std::get<1>(_result.second)[k][i];
                }
                emitter << YAML::EndSeq;
            }
            emitter << YAML::EndMap;
            emitter << YAML::EndMap;

            ofs << emitter.c_str();
            ofs << "\n";
            ofs.close();
            return 1;
        }

        void set_results(const std::string& label,const result_array_t& value){
            _result = std::make_pair(label,value);
        }

        void set_true_labels(std::map<uint32_t,int> true_labels){_true_labels = true_labels;}

        std::string results_to_string(){

            std::stringstream sstr;

            if(statistics_name.empty()){
                ROS_ERROR_STREAM("statistics_name vector is empty.");
                return "";
            }

            if(std::get<1>(_result.second).empty()){
                ROS_ERROR_STREAM("result vector is empty.");
                return "";
            }

            sstr << statistics_name[0] << " : " << std::get<0>(_result.second) << std::endl;
            for(int k = 0; k < statistics_name.size()-1; k++){
                sstr << statistics_name[k+1] << " : ";
                for(int i = 0; i < std::get<1>(_result.second)[k].size(); i++)
                {
                    sstr << std::get<1>(_result.second)[k][i] << " ; ";
                }
                sstr << std::endl;
            }
            return sstr.str();
        }

private:
        Node* _node;
        ip::SurfaceOfInterest::relevance_map_t _weights;
        std::vector<uint32_t> _lbls;
        std::pair<std::string,result_array_t> _result;
        std::string _output_file;
        std::string _soi_method;
        std::string _modality;
        std::map<uint32_t,int> _true_labels;
};

}

#endif //SCORE_COMPUTATION_HPP
