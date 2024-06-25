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
from keras.models import Sequential
from keras.layers import Dense,InputLayer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from math import ceil
from typing import Optional, Union
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

from tensorflow.keras import Sequential, Model
from pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort

import argparse

#import keras_flops.flops_registory
#outfile=open('logs','w')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
import gc
class test_combination(Problem):

    def __init__(self,data):

        super().__init__(n_var=4, n_obj=2,xl=[0,0,2,2],xu=[3,2,99,49])
        self.data=data

    def _evaluate(self, x, out, *args, **kwargs):
        #print(x)
        models=[]
        for config in x:
            clf = keras_classifier(config,self.data)
            models.append((clf,config))
        accuracy=compute_fitness2(models,self.data)
        flops=compute_flops2(models,self.data)
        del models
        gc.collect()
        
        accuracy=accuracy[:,0]
        accuracy=-1*accuracy
        accuracy=np.reshape(accuracy,(accuracy.shape[0],1))
        flops=flops[:,0]
        flops=np.reshape(flops,(flops.shape[0],1))
        out["F"] = np.concatenate([accuracy,flops],axis=1)

def evaluate(x, data, *args, **kwargs):
        #print(x)
        models=[]
        for config in x:
            clf = keras_classifier(config,data)
            models.append((clf,config))
        accuracy=compute_fitness2(models,data)
        flops=compute_flops2(models,data)
        del models
        tf.keras.backend.clear_session()
        gc.collect()
        
        accuracy=accuracy[:,0]
        accuracy=-1*accuracy
        accuracy=np.reshape(accuracy,(accuracy.shape[0],1))
        flops=flops[:,0]
        flops=np.reshape(flops,(flops.shape[0],1))
        return np.concatenate([accuracy,flops],axis=1)

def NSGAA_MLP(data, generations, pop_size,prob_mut=1.0,prob_cross=0.5):
    problem=test_combination(data)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        mutation=PolynomialMutation(prob=prob_mut, repair=RoundingRepair()),
        crossover=SinglePointCrossover(prob=prob_cross),
        eliminate_duplicates=False)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', generations),
                   verbose=False)
    results=[]
    #print(res.X)
    for elem in res.X:
        clf=keras_classifier(elem,data)
        results.append([clf.predict(data['testX'],verbose=0),compute_flops2([(clf,elem)],data)])
   
        del clf
    del res
    del algorithm
    del problem
    
    tf.keras.backend.clear_session()
    gc.collect()
    return results

def keras_classifier(config,data):
    activations = ['linear','sigmoid', 'tanh', 'relu']
    optimizers = ['rmsprop', 'sgd', 'adam']
    #batch_size=min(500,data['trainX'].shape[0])
    batch_size=8
    iter_per_epoch=int(data['trainX'].shape[0]/batch_size)
    patience_stop=np.ceil(80/iter_per_epoch)
    patience_rl=ceil(patience_stop/2)
    es=EarlyStopping(monitor='accuracy',patience=patience_stop)
    rlrop=ReduceLROnPlateau(monitor='accuracy',patience=patience_rl)
    callbacks=[rlrop,es]
    #epochs=int(1000/batch_size)
    epochs=100
    learning_rate_init = 0.09
    activation_index = int(config[0])
    solver_index = int(config[1])
    hidden_layer_size_1 = int(config[2])
    hidden_layer_size_2 = int(config[3])
    Y=to_categorical(data['trainY'])
    model = Sequential()
    #model.add(InputLayer(input_shape=()))
    model.add(Dense(hidden_layer_size_1, input_dim=data['trainX'].shape[1], activation=activations[activation_index]))
    model.add(Dense(hidden_layer_size_2, activation=activations[activation_index]))
    model.add(Dense(Y.shape[1], activation='softmax'))
    #model.add(Dense(data['trainY'].shape[1], activation='softmax'))    
    model.compile(optimizer=optimizers[solver_index], loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data['trainX'], Y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    return model



def estimate_flops(model: Union[Model, Sequential], batch_size: Optional[int] = None) -> int:
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.
    """
    if not isinstance(model, (Sequential, Model)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )

    if batch_size is None:
        batch_size = 1

    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output']='none'
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    # print(frozen_func.graph.get_operations())
    # TODO: show each FLOPS
    return flops.total_float_ops

def compute_fitness2(population, data):
    fitness = []
    for clf,config in population:
        #clf = keras_classifier(config,data)
        #clf = keras_classifier(learning_rate_init=0.09, activation=activations[int(config[0])], solver = optimizers[int(config[1])], alpha=1e-5,\
        #                    hidden_layer_sizes=(int(config[2]), int(config[3]),data),
        #clf.fit(data['trainX'], data['trainY'])
        
        fitness.append([accuracy_score(np.argmax(clf.predict(data['testX'],verbose=0),axis=1), data['testY']), list(config)])        
    return np.asarray(fitness , dtype='object')

def compute_flops2(population, data):
    flops = []
    prediction_flops=0
    for clf,config in population:
        #clf = keras_classifier(config,data)
        #clf = MLPClassifier(learning_rate_init=0.09, activation=activations[int(config[0])], solver = optimizers[int(config[1])], alpha=1e-5,\
        #                    hidden_layer_sizes=(int(config[2]), int(config[3])),\
        #                    max_iter=1000, n_iter_no_change=80)
        #training_flops, prediction_flops = estimate_flops(clf, data['trainX'], data['trainY'])
        training_flops = estimate_flops(clf,batch_size=1)
        flops.append([training_flops,prediction_flops])
    return np.array(flops)

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
def decoding_binary_class(four_class):
    if(four_class.ndim ==1):
        b_val, b_ar = [], []
        for j in range(0, four_class.shape[0]):
            if(four_class[j]==0):
                b_val.append(1)
                b_ar.append(1)
            elif(four_class[j]==1):
                b_val.append(1)
                b_ar.append(0)
            elif(four_class[j]==2):
                b_val.append(0)
                b_ar.append(1)
            else:
                b_val.append(0)
                b_ar.append(0)
        return np.array(b_val), np.array(b_ar)
    else:
        binary_val, binary_ar = [], []
        for i in range(0, four_class.shape[0]):
            bval, bar = decoding_binary_class(four_class[i])
            binary_val.append(bval)
            binary_ar.append(bar)
        return np.array(binary_val), np.array(binary_ar)
    
def get_four_class_performance(y_test_val, y_test_ar, y_pred_val, y_pred_ar):
    y_test_four = get_four_class(y_test_val, y_test_ar)
    y_pred_four = get_four_class(y_pred_val, y_pred_ar)
    
    four_acc = accuracy_score(y_test_four, y_pred_four)*100
    four_prec = precision_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_recall = recall_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_f1 = f1_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    
    #print(subject_no, four_acc, four_prec, four_recall, four_f1)
    return four_acc, four_prec, four_recall, four_f1

def kfold(x, y):
    # do the scalling
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x, y)
    test_data, train_data, train_label, test_label = [], [], [], []
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_data.append(X_train)
        test_data.append(X_test)
        train_label.append(y_train)
        test_label.append(y_test)
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def emotion_classification1(data):
    # GA optimized MLP
    res = NSGAA_MLP(data, generations = 20, pop_size = 20, prob_cross=0.95, prob_mut = 0.001)
    gc.collect()
    y_test = data['testY']
    
    results=[]
    for rest in res:
        y_pred=rest[0]
        y_pred=np.argmax(y_pred,axis=1)
        acc = accuracy_score(y_pred, y_test)*100
        prec = precision_score(y_test, y_pred)*100
        recall = recall_score(y_test, y_pred)*100
        f1 = f1_score(y_test, y_pred)*100
        flops_tr=rest[1][0][0]
        flops_te=rest[1][0][1]
        results.append((acc, prec, recall, f1, flops_tr, flops_te))
    #y_test = data['testY']
    #acc = accuracy_score(y_pred, y_test)*100
    #prec = precision_score(y_test, y_pred)*100
    #recall = recall_score(y_test, y_pred)*100
    #f1 = f1_score(y_test, y_pred)*100
    #print(subject_no, acc, prec, recall, f1)
    #print(type(res))
    del res
    gc.collect()
    return results


def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,default=1,required=True)
    parser.add_argument('--subject',type=str,default='s01',required=True)
    opt=parser.parse_args()
    fold=opt.fold
    subject=opt.subject
    if os.path.isfile(f'arousal/{subject}/pf_{fold}_{subject}.pkl'):
        return
    train_label_valence, test_label_valence, train_label_arousal, test_label_arousal = [], [], [], []
    ddd=pickle.load(open(f'preproc/data_preproc_{subject}.pkl','rb'))
    train_data=ddd['train_data']
    four_train_label=ddd['four_train_label']
    test_data=ddd['test_data']
    four_test_label=ddd['four_test_label']
    for foldt in range(0, 10):
        valence_train_labels, arousal_train_labels = decoding_binary_class(four_train_label[foldt])
        valence_test_labels, arousal_test_labels = decoding_binary_class(four_test_label[foldt])
        train_label_valence.append(valence_train_labels)
        test_label_valence.append(valence_test_labels)
        train_label_arousal.append(arousal_train_labels)
        test_label_arousal.append(arousal_test_labels)
    
    tpf=[]
    res={}
    pf=np.zeros(2)
    max_acc=max_prec=max_recall=max_f1=0
    min_flops=np.inf
    print(f'{fold}')
    os.makedirs(f'arousal/{subject}',exist_ok=True)
    if not os.path.isfile(f'arousal/{subject}/pf_{fold}_{subject}.pkl'):
            
        for run in range(10): # intialize the data diectory
            data = dict(trainX=train_data[fold], testX=test_data[fold], trainY = train_label_arousal[fold],\
                        testY=test_label_arousal[fold])
            #acc, prec, recall, f1, temp_test_val, temp_pred_val = emotion_classification(data)
            cpf = emotion_classification1(data)
            
            for child in cpf:
                acc, prec, recall, f1, flops_tr, flops_ts = child
                if(acc>max_acc):
                    max_acc = acc
                if(prec>max_prec):
                    max_prec = prec
                if(recall>max_recall):
                    max_recall = recall
                if(f1>max_f1):
                    max_f1 = f1
                if(flops_tr<min_flops):
                    min_flops = flops_tr
            cpf=np.array(cpf) 
            to_ex=np.concatenate([pf,cpf],axis=0) if pf.any() else cpf
            acc=to_ex[:,0]*-1
            flops=to_ex[:,-2]
            acc=acc.reshape(acc.shape[0],1)
            flops=flops.reshape(flops.shape[0],1)
            to_pf=np.concatenate([acc,flops],axis=1)
            pfs=efficient_non_dominated_sort(to_pf)
            pf=to_ex[pfs[0]]
            
            print('************************')
            #input()

        tpf.append(pf)
        pickle.dump(tpf,open(f'arousal/{subject}/pf_{fold}_{subject}.pkl','wb'))
        res[str(fold)]=[max_acc, max_prec, max_recall, max_f1, min_flops]                       
        print("fold-" + str(fold+1), max_acc, max_prec, max_recall, max_f1, min_flops)
    # os.makedirs(f'fourclass/{subject}',exist_ok=True)
    # if not os.path.isfile(f'fourclass/{subject}/pf_{fold}_{subject}.pkl'):
        
    #     for run in range(10): # intialize the data diectory
    #         data = dict(trainX=train_data[fold], testX=test_data[fold], trainY = train_label_four_class[fold],\
    #                     testY=test_label_four_class[fold])
    #         #acc, prec, recall, f1, temp_test_val, temp_pred_val = emotion_classification(data)
    #         cpf = emotion_classification1(data)
            
    #         for child in cpf:
    #             acc, prec, recall, f1, flops_tr, flops_ts = child
    #             if(acc>max_acc):
    #                 max_acc = acc
    #             if(prec>max_prec):
    #                 max_prec = prec
    #             if(recall>max_recall):
    #                 max_recall = recall
    #             if(f1>max_f1):
    #                 max_f1 = f1
    #             if(flops_tr<min_flops):
    #                 min_flops = flops_tr
    #         cpf=np.array(cpf) 
    #         to_ex=np.concatenate([pf,cpf],axis=0) if pf.any() else cpf
    #         acc=to_ex[:,0]*-1
    #         flops=to_ex[:,-2]
    #         acc=acc.reshape(acc.shape[0],1)
    #         flops=flops.reshape(flops.shape[0],1)
    #         to_pf=np.concatenate([acc,flops],axis=1)
    #         pfs=efficient_non_dominated_sort(to_pf)
    #         pf=to_ex[pfs[0]]
            
    #         print('************************')
    #         #input()

    #     tpf.append(pf)
    #     pickle.dump(tpf,open(f'fourclass/{subject}/pf_{fold}_{subject}.pkl','wb'))
    #     res[str(fold)]=[max_acc, max_prec, max_recall, max_f1, min_flops]                       
    #     print("fold-" + str(fold+1), max_acc, max_prec, max_recall, max_f1, min_flops)

if __name__=='__main__':
    main()
