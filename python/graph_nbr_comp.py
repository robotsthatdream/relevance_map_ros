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

iteration = list()
nbr_pos_vect = list()
nbr_neg_vect = list()

if len(sys.argv) != 5 :
    print("Usage : \narg1 : archive path")
    print("arg2 : name of file with the scores")
    print("arg3 : number of iteration") 
    print("arg4 : y axes limit") 
    sys.exit(1)




for arch_exp in os.listdir(sys.argv[1]) :
    archive_folder = sys.argv[1] + arch_exp + "/"

    iteration, nbr_pos, nbr_neg \
     = tools.load_nbr_comp(archive_folder + sys.argv[2])
    iteration, tabs = tools.sort_data(iteration, nbr_pos, nbr_neg)

    nbr_pos = tabs[0]
    nbr_neg = tabs[1]
    
    iteration = iteration[:int(sys.argv[3])]
    nbr_pos = nbr_pos[:int(sys.argv[3])]
    nbr_neg = nbr_neg[:int(sys.argv[3])]

    nbr_pos_vect.append(nbr_pos)
    nbr_neg_vect.append(nbr_neg)



data_pos = np.zeros((len(nbr_pos_vect),len(iteration)))
data_neg = np.zeros((len(nbr_neg_vect),len(iteration)))

for i in range(0,len(nbr_neg_vect)) :
    data_pos[i, :] = nbr_pos_vect[i]
    data_neg[i, :] = nbr_neg_vect[i]

median_pos, perc_75_pos, perc_25_pos = tools.median_perc(data_pos)
median_neg, perc_75_neg, perc_25_neg = tools.median_perc(data_neg)
# aver_pos, min_pos, max_pos = tools.average_vector(nbr_pos_vect)
# aver_neg, min_neg, max_neg = tools.average_vector(nbr_neg_vect)

iteration = iteration[:len(median_pos)]



fig, ax1 = plt.subplots(1,figsize=(8,8))


ax1.plot(iteration,median_pos,'g-',label='number of "moveable" samples',linewidth=2)
ax1.plot(iteration,median_neg,'r-',label='number of "non-moveable" samples',linewidth=2)
# ax1.plot(iteration,perc_25_pos,'g-',iteration,perc_75_pos,'g-',linewidth=.5)
# ax1.plot(iteration,perc_25_neg,'r-',iteration,perc_75_neg,'r-',linewidth=.5)
ax1.fill_between(iteration,perc_25_pos,median_pos,facecolor='green',alpha=.5)
ax1.fill_between(iteration,perc_75_pos,median_pos,facecolor='green',alpha=.5)
ax1.fill_between(iteration,perc_25_neg,median_neg,facecolor='red',alpha=.5)
ax1.fill_between(iteration,perc_75_neg,median_neg,facecolor='red',alpha=.5)



# ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0.,fontsize=25)
#


ax1.set_ylabel('number of components',fontsize=25)
ax1.set_xlabel('number of iterations',fontsize=25)
ax1.set_aspect('auto')
ax1.tick_params(labelsize=20)
ax1.set_xlim([0,int(sys.argv[3])])
ax1.set_ylim([0,int(sys.argv[4])])


exp_name = sys.argv[1].split("/")[-2]
print(exp_name)
plt.tight_layout()

# plt.show()

plt.savefig(sys.argv[1] + "../graphs/" + exp_name + "/graph_nbr_comp.png",bbox_inches='tight')

