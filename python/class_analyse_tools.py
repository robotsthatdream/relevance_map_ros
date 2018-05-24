#!/usr/bin/env python

import yaml
import matplotlib.pyplot as plt
import os
import sys
import scipy.stats as stats
import math
import numpy as np


def load_dataset(path) :

    data = list()
    labels = list()

    with open(path) as FILE:
        yaml_file = yaml.load(FILE)
        for sample in yaml_file["frame_0"]["features"] :
            is_nan = False
            vect = yaml_file["frame_0"]["features"][sample]["value"]
            for val in vect :
                if(val == 'nan' or val == '-nan'):
                    is_nan = True
            if(is_nan):
                continue
            labels.append(int(yaml_file["frame_0"]["features"][sample]["label"]))
            data.append(vect)

    return data, labels


def load_pos_neg_data(path) :
    """ Load pos neg directly from data """
    iteration = list()
    nbr_pos = list()
    nbr_neg = list()

    for folder_name in os.listdir(path) :
        if(folder_name.split("_")[0] != "iteration"):
            continue
        file_name = path + folder_name + "/dataset_" + ".yml"
        if(os.stat(file_name).st_size == 0):
            continue
        with open(file_name) as FILE :
            data = yaml.load(FILE)
            pos_counter = 0.
            neg_counter = 0.
            for feat in data["features"]:
                if(data["features"][feat]["label"] == 1):
                    pos_counter+=1.
                else:
                    neg_counter+=1.
            nbr_pos.append(pos_counter)
            nbr_neg.append(neg_counter)
            iteration.append(float(len(data["features"])))
    return iteration, nbr_pos, nbr_neg


def load_pos_neg(filename):
    """  Load pos neg from classifier_eval yaml file  """
    iteration = list()
    nbr_pos = list()
    nbr_neg = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for dataset in eval_file :
            iteration.append(eval_file[dataset]["nbr_samples"])
            nbr_pos.append(eval_file[dataset]["pos_samples"])
            nbr_neg.append(eval_file[dataset]["neg_samples"])

    return iteration, nbr_pos, nbr_neg

def load_false(filename):
    """  Load pos neg from classifier_eval yaml file  """
    iteration = list()
    nbr_pos = list()
    nbr_neg = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for dataset in eval_file :
            iteration.append(eval_file[dataset]["nbr_samples"])
            nbr_pos.append(eval_file[dataset]["false_positives"])
            nbr_neg.append(eval_file[dataset]["false_negatives"])

    return iteration, nbr_pos, nbr_neg

def load_rand_pos_neg(filename):
    """  Load pos neg from classifier_eval yaml file  """
    iteration = list()
    nbr_pos = list()
    nbr_neg = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for dataset in eval_file :
            iteration.append(eval_file[dataset]["nbr_samples"])
            nbr_pos.append(eval_file[dataset]["rand_nb_pos"])
            nbr_neg.append(eval_file[dataset]["rand_nb_neg"])

    return iteration, nbr_pos, nbr_neg



def load_nbr_comp(filename):
    iteration = list()
    nbr_pos = list()
    nbr_neg = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for dataset in eval_file :
            iteration.append(eval_file[dataset]["nbr_samples"])
            nbr_pos.append(eval_file[dataset]["pos_components"])
            nbr_neg.append(eval_file[dataset]["neg_components"])

    return iteration, nbr_pos, nbr_neg

def load_pra(filename) :

    iteration = list()
    precision = list()
    recall = list()
    accuracy = list()

    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for dataset in eval_file :
            iteration.append(eval_file[dataset]["nbr_samples"])

            if(eval_file[dataset]["precision"] == "-nan" or \
                eval_file[dataset]["precision"] == "nan") :
                precision.append(0)
            else :
                precision.append(eval_file[dataset]["precision"])
            if(eval_file[dataset]["recall"] == "-nan" or \
                eval_file[dataset]["recall"] == "nan"):
                recall.append(0)
            else:    
                recall.append(eval_file[dataset]["recall"])
            if(eval_file[dataset]["accuracy"] == "-nan" or \
                eval_file[dataset]["accuracy"] == "nan"):
                accuracy.append(0)
            else:    
                accuracy.append(eval_file[dataset]["accuracy"])

    return iteration, precision, recall, accuracy

def load_pra_multiclass(filename):
    iteration = list()
    precision = list()
    recall = list()
    accuracy = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for it in eval_file : 
            iteration.append(eval_file[it]["nb_samples"])
            p = list()
            r = list()
            a = list()
            for val in eval_file[it]["precision"] :
                if(val == "-nan" or val == "nan") :
                    p.append(0)
                else:
                    p.append(val)
            precision.append(p)

            for val in eval_file[it]["recall"] :
                if(val == "-nan" or val == "nan") :
                    r.append(0)
                else:
                    r.append(val)
            recall.append(r)

            for val in eval_file[it]["accuracy"] :
                if(val == "-nan" or val == "nan") :
                    a.append(0)
                else:
                    a.append(val)
            accuracy.append(a)
    return iteration, precision, recall, accuracy

def load_pos_neg_multiclass(filename) :
    iteration = list()
    nbr_samples = list()
    print(filename)

    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for it in eval_file :
            iteration.append(eval_file[it]["nb_samples"])
            ns = list()
            for val in eval_file[it]["nbr_samples"] :
                ns.append(val)
            nbr_samples.append(ns)

    return iteration, nbr_samples

def load_nb_comp_multiclass(filename) :
    iteration = list()
    nbr_comp = list()
    print(filename)


    with open(filename) as stream :
        eval_file = yaml.load(stream)
        for it in eval_file :
            iteration.append(eval_file[it]["nb_samples"])
            nc = list()

            for val in eval_file[it]["nbr_comp"] :
                nc.append(val)
            nbr_comp.append(nc)

    return iteration, nbr_comp
    


def load_llhood(file_name) :
    iteration = list()
    loglikelihood = list()

    with open(file_name) as stream:
        line = stream.readline()
        while(line) :
            split_line = line.split(" : ")
            iteration.append(float(split_line[0]))
            loglikelihood.append(float(split_line[1]))
            line = stream.readline()

    return iteration, loglikelihood

def sort_data(x,*args):
    indexes = sorted(range(len(x)), key=lambda k: x[k])
    x.sort()

    y = list()

    for a in args :
        a_sort = list()
        for i in range(0,len(indexes)):
            a_sort.append(a[indexes[i]])
        y.append(a_sort)

    index_to_del = list()
    for i in range(1, len(x)):
        if(x[i] == x[i-1]):
            index_to_del.append(i)

    for i in range(0,len(index_to_del)) :
        del x[index_to_del[i]]
        for a in y:
            del a[index_to_del[i]]
        for k in range(0,len(index_to_del)):
            index_to_del[k] = index_to_del[k] -1

    return x, y

def data_smoothing(data,win) :
    ndata = np.zeros(len(data))
    for i in range(0,len(data)):
        count=0
        for k in range(-win,win+1):
            if(i+k < 0 or i+k >= len(data)):
                continue
            ndata[i] = ndata[i] + data[i+k]
            count+=1
        ndata[i] = ndata[i]/float(count)

    return ndata

def absolute_difference(l1,l2): 
    diff = list()
    for i in range(0,len(l1)) :
        diff.append(abs(l1[i]-l2[i]))
    return diff


def performance(f_naive,f_expert,f_class):
    perf = list()
    for i in range(0,len(f_naive)):
        perf.append((f_class[i] - f_naive[i])/(f_expert[i]-f_naive[i]))
    return perf

def neg_pos_fitting(iteration,nbr_pos,nbr_neg):
    log_iter = list()
    for it in iteration :
        log_iter.append(math.log(it))

    log_nbr_pos = list()
    for n in nbr_pos:
        if(n > 0):
            log_nbr_pos.append(math.log(n))
        else :
            log_nbr_pos.append(0)

    slope, intercept, r_value_neg, p_value_neg, std_err_neg = stats.linregress(log_iter,nbr_neg)

    slope, intercept, r_value_pos, p_value_pos, std_err_pos = stats.linregress(iteration,log_nbr_pos)

    return r_value_pos**2, p_value_pos, std_err_pos, r_value_neg**2, p_value_neg, std_err_neg

def median_perc(data) :
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
    return median, perc_75, perc_25

def average_vector(vects) :
    min_size = len(vects[0])

    for vect in vects :
        if(len(vect) < min_size):
            min_size = len(vect)

    aver_vect = list(0 for i in range(0,min_size))
    min_vect = list(vects[0][i] for i in range(0,min_size))
    max_vect = list(vects[0][i] for i in range(0,min_size))

    for vect in vects :
        for i in range(0,min_size) :
            if(vect[i] < min_vect[i]):
                min_vect[i] = vect[i]
            if(vect[i] > max_vect[i]):
                max_vect[i] = vect[i]
            aver_vect[i] += vect[i]

    for i in range(0,len(aver_vect)):
        aver_vect[i] = float(aver_vect[i])/float(len(vects))

    return aver_vect, min_vect, max_vect