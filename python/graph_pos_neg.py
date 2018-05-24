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
rand_pos_vect = list()
rand_neg_vect = list()

with_false = False

if len(sys.argv) != 4 and len(sys.argv) != 5 :
    print("Usage : \narg1 : archive path")
    print("arg2 : name of file with the scores")
    print("arg3 : number of iteration")
    print("option : --with_false plot false samples") 
    sys.exit(1)


if len(sys.argv) == 5 :
    if sys.argv[4] == "--with_false":
        with_false = True

    
if(with_false):
    f_pos_vect = list()
    f_neg_vect = list()



for arch_exp in os.listdir(sys.argv[1]) :
    archive_folder = sys.argv[1] + arch_exp + "/"


    iteration, rand_pos, rand_neg \
     = tools.load_rand_pos_neg(archive_folder + sys.argv[2])

    iteration, nbr_pos, nbr_neg \
     = tools.load_pos_neg(archive_folder + sys.argv[2])
    

    if(with_false):
        iteration, f_pos, f_neg \
        = tools.load_false(archive_folder + sys.argv[2])
        iteration, tabs = tools.sort_data(iteration,nbr_pos,nbr_neg, \
            rand_pos,rand_neg,f_pos,f_neg)
        f_pos = tabs[4]
        f_neg = tabs[5]

        f_neg = f_neg[:int(sys.argv[3])]
        f_pos = f_pos[:int(sys.argv[3])]
        
        f_pos2 = [f_pos[0]]
        f_neg2 = [f_neg[0]]
        n_pos = 0
        n_neg = 0
        for i in range(1,len(f_pos)):
            if(f_pos[i] < f_pos[i-1]):
                n_pos += f_pos[i-1]
            if(f_neg[i] < f_neg[i-1]):
                n_neg += f_neg[i-1]
            f_pos2.append(f_pos[i] + n_pos)
            f_neg2.append(f_neg[i] + n_neg)
        f_pos = f_pos2
        f_neg = f_neg2

        f_pos_vect.append(f_pos)
        f_neg_vect.append(f_neg)
    else : 
        iteration, tabs = tools.sort_data(iteration,nbr_pos,nbr_neg, \
            rand_pos,rand_neg)

    nbr_pos = tabs[0]
    nbr_neg = tabs[1]
    rand_pos = tabs[2]
    rand_neg = tabs[3]
    
    iteration = iteration[:int(sys.argv[3])]
    nbr_pos = nbr_pos[:int(sys.argv[3])]
    nbr_neg = nbr_neg[:int(sys.argv[3])]
    rand_pos = rand_pos[:int(sys.argv[3])]
    rand_neg = rand_neg[:int(sys.argv[3])]

    nbr_pos_vect.append(nbr_pos)
    nbr_neg_vect.append(nbr_neg)
    rand_pos_vect.append(rand_pos)
    rand_neg_vect.append(rand_neg)

data_pos = np.zeros((len(nbr_pos_vect),len(iteration)))
data_neg = np.zeros((len(nbr_neg_vect),len(iteration)))
data_rand_pos = np.zeros((len(rand_pos_vect),len(iteration)))
data_rand_neg = np.zeros((len(rand_neg_vect),len(iteration)))

for i in range(0,len(nbr_neg_vect)) :
    data_pos[i, :] = nbr_pos_vect[i]
    data_neg[i, :] = nbr_neg_vect[i]
    data_rand_pos[i, :] = rand_pos_vect[i]
    data_rand_neg[i, :] = rand_neg_vect[i]

median_pos, perc_75_pos, perc_25_pos = tools.median_perc(data_pos)
median_neg, perc_75_neg, perc_25_neg = tools.median_perc(data_neg)
median_rand_pos, perc_75_rand_pos, perc_25_rand_pos = tools.median_perc(data_rand_pos)
median_rand_neg, perc_75_rand_neg, perc_25_rand_neg = tools.median_perc(data_rand_neg)

if(with_false) :
    data_f_pos = np.zeros((len(f_pos_vect),len(iteration)))
    data_f_neg = np.zeros((len(f_pos_vect),len(iteration)))

    for i in range(0,len(f_pos_vect)) :
        data_f_pos[i, :] = f_pos_vect[i]
        data_f_neg[i, :] = f_neg_vect[i]

    median_f_pos, perc_75_f_pos, perc_25_f_pos = tools.median_perc(data_f_pos)
    median_f_neg, perc_75_f_neg, perc_25_f_neg = tools.median_perc(data_f_neg)

    median_f_pos = median_pos - median_f_pos
    median_f_neg = median_neg - median_f_neg

# iteration = iteration[:len(median_pos)]


fig, ax1 = plt.subplots(1,figsize=(8,8))



ax1.plot(iteration,median_rand_pos,':',color='blue',label='number of "moveable" samples random sampling',linewidth=2)
ax1.plot(iteration,median_rand_neg,':',color='brown',label='number of "non-moveable" samples random sampling',linewidth=2)
# ax1.plot(iteration,perc_25_rand_pos,'g-',iteration,perc_75_rand_pos,'g-',linewidth=.5)
# ax1.plot(iteration,perc_25_rand_neg,'r-',iteration,perc_75_rand_neg,'r-',linewidth=.5)
# ax1.fill_between(iteration,perc_25_rand_pos,median_rand_pos,facecolor='green',alpha=.5)
# ax1.fill_between(iteration,perc_75_rand_pos,median_rand_pos,facecolor='green',alpha=.5)
# ax1.fill_between(iteration,perc_25_rand_neg,median_rand_neg,facecolor='red',alpha=.5)
# ax1.fill_between(iteration,perc_75_rand_neg,median_rand_neg,facecolor='red',alpha=.5)

ax1.plot(iteration,median_pos,'g-',label='number of "moveable" samples',linewidth=2)
ax1.plot(iteration,median_neg,'r-',label='number of "non-moveable" samples',linewidth=2)
# ax1.plot(iteration,perc_25_pos,'g-',iteration,perc_75_pos,'g-',linewidth=.5)
# ax1.plot(iteration,perc_25_neg,'r-',iteration,perc_75_neg,'r-',linewidth=.5)
ax1.fill_between(iteration,perc_25_pos,median_pos,facecolor='green',alpha=.5)
ax1.fill_between(iteration,perc_75_pos,median_pos,facecolor='green',alpha=.5)
ax1.fill_between(iteration,perc_25_neg,median_neg,facecolor='red',alpha=.5)
ax1.fill_between(iteration,perc_75_neg,median_neg,facecolor='red',alpha=.5)

if(with_false):
    ax1.plot(iteration,median_f_pos,'g--',label='number of true "moveable" samples',linewidth=2)
    ax1.plot(iteration,median_f_neg,'r--',label='number of true "non-moveable" samples',linewidth=2)
    # ax1.plot(iteration,perc_25_f_pos,'g:',iteration,perc_75_f_pos,'g:',linewidth=.5)
    # ax1.plot(iteration,perc_25_f_neg,'r:',iteration,perc_75_f_neg,'r:',linewidth=.5)
    # ax1.fill_between(iteration,perc_25_f_pos,median_f_pos,facecolor='green',alpha=.5)
    # ax1.fill_between(iteration,perc_75_f_pos,median_f_pos,facecolor='green',alpha=.5)
    # ax1.fill_between(iteration,perc_25_f_neg,median_f_neg,facecolor='red',alpha=.5)
    # ax1.fill_between(iteration,perc_75_f_neg,median_f_neg,facecolor='red',alpha=.5)

# ax1.plot(iteration,diff_neg_pos,'b-',linewidth=2,label='absolute difference')
# ax1.plot(iteration,min_diff_neg_pos,'b-',iteration,max_diff_neg_pos,'b-')
# ax1.fill_between(min_diff_neg_pos,diff_neg_pos,facecolor='blue',alpha=.5)
# ax1.fill_between(max_diff_neg_pos,diff_neg_pos,facecolor='blue',alpha=.5)

# ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0.,fontsize=25)



ax1.set_ylabel('accumulated number of samples',fontsize=30)
ax1.set_xlabel('number of iterations',fontsize=30)
ax1.set_aspect('equal')
ax1.tick_params(labelsize=20)
ax1.set_xlim([0,int(sys.argv[3])])
ax1.set_ylim([0,int(sys.argv[3])])


# plt.figtext(0.2,0,text,bbox=dict(facecolor='white'))
# plt.figtext(0.7,0,text_2,bbox=dict(facecolor='white'))


exp_name = sys.argv[1].split("/")[-2]

print(exp_name)
plt.tight_layout()


# plt.show()

plt.savefig(sys.argv[1] + "../graphs/" + exp_name + "/graph_pos_neg.png",bbox_inches='tight')


