import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle, mne, warnings, copy# done in mac m1
import numpy as np # done in mac m1
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mne.filter import filter_data as bandpass_filter
from mne.preprocessing import ICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover, SinglePointCrossover 
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
# from keras.models import Sequential
# from keras.layers import Dense,InputLayer
# from keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from math import ceil
import torch
from ptflops import get_model_complexity_info
from typing import Optional, Union
# from tensorflow.python.framework.convert_to_constants import (
    # convert_variables_to_constants_v2_as_graph,
# )

# from tensorflow.keras import Sequential, Model
from pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort

import argparse

#import keras_flops.flops_registory
#outfile=open('logs','w')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#import tensorflow as tf

#tf.get_logger().setLevel('ERROR')
#tf.autograph.set_verbosity(2)
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
import gc

def get_four_class(val, ar):
    # convert binary to multiclass
    emotion = np.ones(val.shape[0])
    assert val.shape[0]==ar.shape[0]
    for i in range(0, val.shape[0]):
        if(val[i]==1 and ar[i]==1): # HVHA
            emotion[i] = 0
        elif(val[i]==1 and ar[i]==0): #HVLA
            emotion[i] = 1
        elif(val[i]==0 and ar[i]==1): #LVHA
            emotion[i] = 2
        else: #LVLA
            emotion[i] = 3
    return emotion
    
def get_four_class_performance(y_test_val, y_test_ar, y_pred_val, y_pred_ar):
    y_test_four = get_four_class(y_test_val, y_test_ar)
    y_pred_four = get_four_class(y_pred_val, y_pred_ar)
    
    four_acc = accuracy_score(y_test_four, y_pred_four)*100
    four_prec = precision_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_recall = recall_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_f1 = f1_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    
    #print(subject_no, four_acc, four_prec, four_recall, four_f1)
    return four_acc, four_prec, four_recall, four_f1
    
def sett(pf):
    res=[pf[0]]
    for p in pf:
        add=True
        for r in res:
            matches=True
            for k in range(len(p)):
                if not np.array_equal(r[k],p[k]):
                    matches=False
                    break
            if matches:
                add=False
                break
        if add:
            res.append(p)
    return res
   
subjects=['s'+str(i).rjust(2,'0') for i in range(1,33)] 
valence_path=Path('valence')
arousal_path=Path('arousal')
valence_files=valence_path.glob('*/*.pkl')
arousal_files=arousal_path.glob('*/*.pkl')
valence_files=[str(fil) for fil in valence_files]
arousal_files=[str(fil) for fil in arousal_files]
valence_f={subject : [fil for fil in valence_files if subject in fil] for subject in subjects}
arousal_f={subject : [fil for fil in arousal_files if subject in fil] for subject in subjects}


pf_valence={subject : [[] for fold in range(10)] for subject in subjects}
pf_arousal={subject : [[] for fold in range(10)] for subject in subjects}
for subject in subjects:
    for fold in range(10):
        fil=[fils for fils in valence_f[subject] if str(fold)==fils.split('/')[-1].split('_')[1]]
        if len(fil)==0:
            continue
        pf=pickle.load(open(fil[0]+'1','rb'))
        pf_valence[subject][fold]=pf
    for fold in range(10):
        fil=[fils for fils in arousal_f[subject] if str(fold)==fils.split('/')[-1].split('_')[1]]
        if len(fil)==0:
            continue
        pf=pickle.load(open(fil[0]+'1','rb'))
        pf_arousal[subject][fold]=pf
pf_fc={subject : [[] for fold in range(10)] for subject in subjects}
for subject in subjects:
    for fold in range(10):
        temp=[]
        
        for i in range(len(pf_valence[subject][fold])):
            for j in range(len(pf_arousal[subject][fold])):
                if len(pf_arousal[subject][fold][j])<8 or len(pf_valence[subject][fold][i])<8:
                    print('no_good')
                    continue
                te=list(get_four_class_performance(pf_valence[subject][fold][i][4],pf_arousal[subject][fold][j][4],pf_valence[subject][fold][i][5],pf_arousal[subject][fold][j][5]))
                te.append(pf_valence[subject][fold][i][6]+pf_arousal[subject][fold][j][6])
                te.append(0)
                temp.append(te)
        to_ex=np.array(temp)
        acc=to_ex[:,0]*-1
        flops=to_ex[:,-2]
        acc=acc.reshape(acc.shape[0],1)
        flops=flops.reshape(flops.shape[0],1)
        to_pf=np.concatenate([acc,flops],axis=1)
        pfs=efficient_non_dominated_sort(to_pf)
        pf=to_ex[pfs[0]]
        pf=sett(pf)
        pf_fc[subject][fold]=pf
 
maxes=[]           
for subject in subjects:
    for fold in range(10):
        if len(pf_valence[subject][fold])==0:
            continue
        pfacc=[i[0] for i in pf_valence[subject][fold]]
        pfflop=[i[-2] for i in pf_valence[subject][fold]]
        #plt.scatter(pfacc,pfflop)
        #plt.savefig(f'results/valence_pf_{subject}_{fold}.png')
        #plt.clf()
    for fold in range(10):
        if len(pf_arousal[subject][fold])==0:
            print(subject,fold)
            continue
        pfacc=[i[0] for i in pf_arousal[subject][fold]]
        pfflop=[i[-2] for i in pf_arousal[subject][fold]]
        #plt.scatter(pfacc,pfflop)
        #plt.savefig(f'results/arousal_pf_{subject}_{fold}.png')
        #plt.clf()
    for fold in range(10):
        if len(pf_fc[subject][fold])==0:
            print(subject,fold)
            continue
        pfacc=[i[0] for i in pf_fc[subject][fold]]
        maxes.append(max(pfacc))
        pfflop=[i[-2] for i in pf_fc[subject][fold]]
        #plt.scatter(pfacc,pfflop)
        #plt.savefig(f'results/fc_pf_{subject}_{fold}.png')
        #plt.clf()

print(sum(maxes)/len(maxes))



