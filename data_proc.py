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
import argparse

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
def get_data(subject_no):
    # read the data
    dataset_path = ''
    deap_dataset = pickle.load(open(dataset_path + subject_no + '.dat', 'rb'), encoding='latin1')
    # separate data and labels 
    
    data = np.array(deap_dataset['data']) # for current data
    labels = np.array(deap_dataset['labels']) # for current labels
    # remove 3sec pre baseline
    data  = data[0:40,0:32,384:8064]
    # signal processing
    #input()
    data = signal_pro(data)
    # feature extraction
    #input()
    feature = get_feature(data)
    # class label
    #input()
    four_class_labels = get_class_labels(labels, 'four_class')
    #input()
    return kfold(feature, four_class_labels)
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
def get_four_class_performance(y_test_val, y_test_ar, y_pred_val, y_pred_ar):
    y_test_four = get_four_class(y_test_val, y_test_ar)
    y_pred_four = get_four_class(y_pred_val, y_pred_ar)
    
    four_acc = accuracy_score(y_test_four, y_pred_four)*100
    four_prec = precision_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_recall = recall_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    four_f1 = f1_score(y_test_four, y_pred_four, labels=[0,1,2,3], average='weighted')*100
    
    #print(subject_no, four_acc, four_prec, four_recall, four_f1)
    return four_acc, four_prec, four_recall, four_f1
import time
N_C = None
droping_components = 'one'
def SignalPreProcess(eeg_rawdata):
    """
    :param eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
    :return: filtered EEG raw data
    """
    assert eeg_rawdata.shape[0] == 32
    eeg_rawdata = np.array(eeg_rawdata)

    ch_names = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", 
                "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
                "PO4", "O2"]
  
    info = mne.create_info(ch_names = ch_names, ch_types = ['eeg' for _ in range(32)], sfreq = 128, verbose=False)
    
    raw_data = mne.io.RawArray(eeg_rawdata, info, verbose = False)
    
    raw_data.load_data(verbose = False).filter(l_freq = 4, h_freq = 48, method = 'fir', verbose = False)
    #raw_data.plot()

    ica = ICA(n_components = N_C, random_state = 97, verbose = False)
    ica.fit(raw_data)
    # https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html?highlight=find_eog_#mne.preprocessing.find_eog_events
    eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name = 'Fp1', verbose = None)
    a = abs(eog_scores).tolist()
    if(droping_components == 'one'):
        ica.exclude = [a.index(max(a))]
        
    else: # find two maximum scores
        a_2 = a.copy()
        a.sort(reverse = True)
        exclude_index = []
        for i in range(0, 2):
            for j in range(0, len(a_2)):
                if(a[i]==a_2[j]):
                    exclude_index.append(j)
        ica.exclude = exclude_index
    ica.apply(raw_data, verbose = False)
    # common average reference
    raw_data.set_eeg_reference('average', ch_type = 'eeg')#, projection = True)
    filted_eeg_rawdata = np.array(raw_data.get_data())
    return filted_eeg_rawdata

def signal_pro(input_data):
    print(input_data.shape[0])
    for i in range(input_data.shape[0]):
        input_data[i] = SignalPreProcess(input_data[i].copy())
        print(i)
    return input_data

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
def get_class_labels(labels, class_type):
    # encoding
    emotion = np.ones(40)
    if(class_type=='valence'):
        for i in range(0, 40):
            if labels[i][0]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    elif(class_type=='arousal'):
        for i in range(40):
            if labels[i][1]>=5:
                emotion[i] = 0
            else:
                emotion[i] = 1
    else:
        for i in range(40):
            if(labels[i][0]>=5 and labels[i][1] >=5): # HVHA
                emotion[i] = 0
            elif(labels[i][0]>=5 and labels[i][1]<5): #HVLA
                emotion[i] = 1
            elif(labels[i][0]<5 and labels[i][1]>=5): #LVHA
                emotion[i] = 2
            else: #LVLA
                emotion[i] = 3
    return emotion
def get_feature(data):
    channel_no = [0, 2, 16, 19] # only taking these four channels
    feature_vector = [6.2, 7.3, 6.2, 7.3]
    feature_matrix = []
    for ith_video in range(40):
        features = []
        for ith_channel in channel_no:
            # power spectral density
            # please refer: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.psd.html
            psd, freqs = plt.psd(data[ith_video][ith_channel], Fs = 128)
            # get frequency bands mean power
            theta_mean = np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
            alpha_mean = np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
            beta_mean  = np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
            gamma_mean = np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])
            features.append([theta_mean, alpha_mean, beta_mean, gamma_mean])
        # flatten the features i.e. transform it from 2D to 1D
        feature_matrix.append(np.array(list(chain.from_iterable(features))))
    return np.array(feature_matrix)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--subject',type=str,default='s01',required=True)
    ars=parser.parse_args()
    subject_no = ars.subject
    train_data, four_train_label, test_data, four_test_label = get_data(subject_no)
    # train_label has only four class emotion now extract it into binary class
    train_label_valence, test_label_valence, train_label_arousal, test_label_arousal = [], [], [], []
    dat_dat_dat=dict(train_data=train_data,four_train_label=four_train_label,test_data=test_data,four_test_label=four_test_label)
    pickle.dump(dat_dat_dat,open(f'data_preproc_{subject_no}.pkl','wb'))

if __name__=='__main__':
    main()