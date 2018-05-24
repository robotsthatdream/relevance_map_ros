#!/usr/bin/env python

import sys
import os
import matplotlib.pyplot as plt
import class_analyse_tools as tools
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

precisions = list()
recalls = list()
accuracys = list()


if len(sys.argv) != 6 :
    print("Usage : \narg1 : archive path")
    print("arg2 : name of file with the scores")
    print("arg3 : number of iteration") 
    print("arg4 : smoothing factor")
    print("arg5 : bootstrap step")
    sys.exit(1)


for arch_exp in os.listdir(sys.argv[1]) :
    archive_folder = sys.argv[1] + arch_exp + "/"

    iteration, precision, recall, accuracy \
     = tools.load_pra(archive_folder + sys.argv[2])
    iteration, tabs = tools.sort_data(iteration,precision,recall,accuracy)

    precision = tabs[0]
    recall = tabs[1]
    accuracy = tabs[2]

    iteration = iteration[:int(sys.argv[3])]
    precision = precision[:int(sys.argv[3])]
    recall = recall[:int(sys.argv[3])]
    accuracy = accuracy[:int(sys.argv[3])]

    precisions.append(precision)
    recalls.append(recall)
    accuracys.append(accuracy)

print(len(iteration))
data_p = np.zeros((len(precisions),len(iteration)))
data_r = np.zeros((len(recalls),len(iteration)))
data_a = np.zeros((len(accuracys),len(iteration)))


for i in range(0,len(precisions)) :
    data_p[i, :] = precisions[i]
    data_r[i, :] = recalls[i]
    data_a[i, :] = accuracys[i]


median_p, perc_75_p, perc_25_p = tools.median_perc(data_p)
median_r, perc_75_r, perc_25_r = tools.median_perc(data_r)
median_a, perc_75_a, perc_25_a = tools.median_perc(data_a)

median_p = tools.data_smoothing(median_p,int(sys.argv[4]))
median_r = tools.data_smoothing(median_r,int(sys.argv[4]))
median_a = tools.data_smoothing(median_a,int(sys.argv[4]))

perc_75_p = tools.data_smoothing(perc_75_p,int(sys.argv[4]))
perc_75_r = tools.data_smoothing(perc_75_r,int(sys.argv[4]))
perc_75_a = tools.data_smoothing(perc_75_a,int(sys.argv[4]))

perc_25_p = tools.data_smoothing(perc_25_p,int(sys.argv[4]))
perc_25_r = tools.data_smoothing(perc_25_r,int(sys.argv[4]))
perc_25_a = tools.data_smoothing(perc_25_a,int(sys.argv[4]))


# aver_prec, min_prec, max_prec = tools.average_vector(precisions)
# aver_rec, min_rec, max_rec = tools.average_vector(recalls)
# aver_acc, min_acc, max_acc = tools.average_vector(accuracys)




iteration = iteration[:len(median_p)]

fig, ax1 = plt.subplots(1,figsize=(5*float(sys.argv[3])/100.,8))
# fig.figure(figsize=(24,14),dpi=80)

ax1.axvline(x=sys.argv[5],linewidth=2,label='bootstrap iterations')
ax1.plot(iteration,median_p,'k-', linewidth=2,label='precision')
ax1.plot(iteration,median_r,'r-', linewidth=2,label='recall')
ax1.plot(iteration,median_a,'g-', linewidth=2,label='accuracy')


# ax1.plot(iteration,perc_25_p,'w',iteration,perc_75_p,'w')
ax1.fill_between(iteration,perc_25_p,median_p,facecolor='black',alpha=0.2)
ax1.fill_between(iteration,perc_75_p,median_p,facecolor='black',alpha=0.2)

# ax1.plot(iteration,perc_25_r,'w',iteration,perc_75_r,'w')
ax1.fill_between(iteration,perc_25_r,median_r,facecolor='red',alpha=0.2)
ax1.fill_between(iteration,perc_75_r,median_r,facecolor='red',alpha=0.2)

# ax1.plot(iteration,perc_25_a,'w',iteration,perc_75_a,'w')
ax1.fill_between(iteration,perc_25_a,median_a,facecolor='green',alpha=0.2)
ax1.fill_between(iteration,perc_75_a,median_a,facecolor='green',alpha=0.2)


ax1.set_ylabel('performance score',fontsize=35)
ax1.set_xlabel('number of iterations',fontsize=35)

ax1.tick_params('x',labelsize=30)
ax1.tick_params('y',labelsize=20)

ax1.set_xlim([0,int(sys.argv[3])])
ax1.set_ylim([0,1.01])
# ratio = 1./(float(sys.argv[3])/100.)
# print("ration " + str(ratio))
# ax1.set_aspect(100.*ratio)

# ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0.,fontsize=25)

exp_name = sys.argv[1].split("/")[-2]

print(exp_name)

plt.tight_layout()

plt.savefig(sys.argv[1] + "../graphs/" + exp_name + "/graph_pra.png")#,bbox_inches='tight')

# plt.show()
