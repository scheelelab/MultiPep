# -*- coding: utf-8 -*-

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout,Multiply, LSTM, Add, Concatenate, TimeDistributed
from tensorflow.keras.layers import Conv1D, Flatten, Lambda, MaxPooling1D, GRU, SimpleRNN, PReLU
from tensorflow.keras.layers import Reshape 
from tensorflow.keras.models import Model
#from keras.regularizers import l2, l1
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_score, recall_score,f1_score,cohen_kappa_score,accuracy_score
import time
import argparse
import sys
import numpy as np
import pickle
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.stats.multitest as multi
seed_val = 1234

### Controlling randomness
tf.random.set_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)



vocab = ['A',
 'R',
 'N',
 'D',
 'C',
 'Q',
 'E',
 'G',
 'H',
 'I',
 'L',
 'K',
 'M',
 'F',
 'P',
 'S',
 'T',
 'W',
 'Y',
 'V',]


with open("branches32.pkl","rb") as f:
    branches = pickle.load(f)

nms = np.concatenate(branches)


    
lvl1 = [list(branches[0])+list(branches[1]), list(branches[2])+list(branches[3])+list(branches[4])]
lvl2_1 = [list(branches[0]),list(branches[1])]
lvl2_2 = [list(branches[2]),list(branches[3])+list(branches[4])]
lvl3 = [list(branches[3]),list(branches[4])]
lvl4_1 = [list(branches[0])]
lvl4_2 = [list(branches[1])]
lvl4_3 = [list(branches[2])]
lvl4_4 = [list(branches[3])]
lvl4_5 = [list(branches[4])]
levels = [lvl1, lvl2_1, lvl2_2, lvl3, lvl4_1, lvl4_2, lvl4_3, lvl4_4, lvl4_5]



def read_input_file(inp):
    seqs = []
    with open(inp,"r") as f:
        for i in f:
            seqs.append(i.strip())
    
    rem = []
    for i in seqs:
        if len(set(list(i)).difference(vocab)) != 0:
            rem.append(i)
        if len(i) > 200 or len(i) < 2:
            rem.append(i)
    
    return seqs, rem


def encode_seqs_only(sq_dct, max_, voc):

    lnv = len(voc)
    #dims = len(ac_over)
    alsqs = []
    y_list = []
    x_list = []

    for sq in sq_dct:
        diff = max_ - len(sq)
        if diff % 2 == 0:
            tmps = "9"*int(diff/2) + sq + "9"*int(diff/2)
        if diff % 2 != 0:    
            tmps = "9"*int((diff-1)/2) + sq + "9"*int((diff-1)/2 + 1)
        #tmps = sq + "9"*int(diff)
        alsqs.append(sq)
        tmp_x = np.zeros((max_,lnv))
        for ii in range(len(tmp_x)):  
            if tmps[ii] in voc:
                tmp_x[ii][voc.index(tmps[ii])] = 1.
        x_list.append([tmp_x.flatten()])

    return np.concatenate(x_list,axis=0), alsqs


##############################################################################


init3 = RandomUniform(minval=-0.001, maxval=0.001, seed=seed_val)
init4 = Constant(1)
init5 = RandomUniform(minval=0.001, maxval=0.05, seed=seed_val)
init6 = RandomUniform(minval=-1, maxval=1, seed=seed_val)
init7 = Constant(0.001)

def max_sec_ax_keep_dims(inp):
    return K.max(inp,axis=-2, keepdims=True)

def divider(inp):
    #return inp / (K.max(K.abs(inp),axis=-1, keepdims=True) + K.epsilon())
    return inp / (K.sum(K.abs(inp),axis=-1, keepdims=True) + K.epsilon())

def bottum_up(inp):
    pred = K.max(inp, axis=-1, keepdims=True)
    return pred



def small_cnn(x,kernels,LR,init2):
    l1x = []
    l2_reg = 0.0
    drop = 0.5
    for i in  [4,6,10,16,22,30,40]:
        cxc = len(vocab)
        x4 = Conv1D(kernels,kernel_size=cxc*i, strides=cxc, padding="valid",  activation=LR,kernel_initializer=init2, use_bias=False)(x)
        x4 = PReLU(alpha_initializer=Constant(value=0.3))(x4)
        x4 = Lambda(max_sec_ax_keep_dims)(x4)
        l1x.append(x4)
    
    x4 = Concatenate(axis=-2)(l1x)
    z41x = Flatten()(x4)
    z41x = Lambda(divider)(z41x)
    z41 = Dropout(0.2)(z41x)
    
    z42 = Dense(500, activation='linear',kernel_initializer=init6, use_bias=True)(z41)
    z42 = PReLU(alpha_initializer=Constant(value=0.3))(z42)
    #z42 = Lambda(divider)(z42)
    #z42x = Concatenate()([z41x, z42])
    z42 = Dropout(drop)(z42)
    
    z43 = Dense(500, activation='linear',kernel_initializer=init6, use_bias=True)(z42)
    z43 = PReLU(alpha_initializer=Constant(value=0.3))(z43)
    #z43 = Lambda(divider)(z43)
    #z43x = Concatenate()([z42x, z43])
    z43 = Dropout(drop)(z43)
    
    z44 = Dense(500, activation='linear',kernel_initializer=init6, use_bias=True)(z43)
    z44 = PReLU(alpha_initializer=Constant(value=0.3))(z44)
    #z44 = Lambda(divider)(z44)
    #z44x = Concatenate()([z43x, z44])
    z4 = Dropout(drop)(z44)
    
    return z4

def activation3(x):
    #return K.relu((x-0.5)*2)
    #return K.relu(x, threshold=0.5)
    return K.relu(x/0.5-0.5,max_value=1)


inputs = Input(shape=(len(vocab)*200,1))
x = inputs
xx = Flatten()(x)

init2 = "orthogonal"
name1 = "fold1_"
kernels = 40
LR = "linear"
l2_reg = 0.0


z4 = small_cnn(x,kernels,LR,init2)
lvl3_1 = Dense(len(levels[4][0]), activation='sigmoid',kernel_initializer=init5, use_bias=True, name="outputs5")
outputs5 = lvl3_1(z4)
#outputs5 = Lambda(activation3)(outputs5)


z4 = small_cnn(x,kernels,LR,init2)
lvl3_1 = Dense(len(levels[5][0]), activation='sigmoid',kernel_initializer=init5, use_bias=True, name="outputs6")
outputs6 = lvl3_1(z4)
#outputs6 = Lambda(activation3)(outputs6)


z7 = small_cnn(x,kernels,LR,init2)
lvl3_2 = Dense(len(levels[6][0]), activation='sigmoid',kernel_initializer=init5, use_bias=True, name="outputs7")
outputs7 = lvl3_2(z7)
#outputs7 = Lambda(activation3)(outputs7)


z5 = small_cnn(x,kernels,LR,init2)
lvl3_3 = Dense(len(levels[7][0]), activation='sigmoid',kernel_initializer=init5, use_bias=True, name="outputs8")
outputs8 = lvl3_3(z5)
#outputs8 = Lambda(activation3)(outputs8)


z6 = small_cnn(x,kernels,LR,init2)
lvl3_4 = Dense(len(levels[8][0]), activation='sigmoid',kernel_initializer=init5, use_bias=True, name="outputs9")
outputs9 = lvl3_4(z6)
#outputs9 = Lambda(activation3)(outputs9)


outputs51 = Lambda(bottum_up)(outputs8)
outputs52 = Lambda(bottum_up)(outputs9)
outputs4 = Concatenate(axis=-1)([outputs51,outputs52])

outputs31 = Lambda(bottum_up)(outputs7)
outputs32 = Lambda(bottum_up)(outputs4)
outputs3 = Concatenate(axis=-1)([outputs31,outputs32])

outputs21 = Lambda(bottum_up)(outputs5)
outputs22 = Lambda(bottum_up)(outputs6)
outputs2 = Concatenate(axis=-1)([outputs21,outputs22])

outputs11 = Lambda(bottum_up)(outputs2)
outputs12 = Lambda(bottum_up)(outputs3)
outputs1 = Concatenate(axis=-1)([outputs11,outputs12])


outputs = [outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7, outputs8, outputs9]



decoderx = Model(inputs, outputs)
decoderx.summary()



##############################################################################






th = [0.5,0.7,0.9]

n = ['val_0.5_mean_recall', 'val_0.7_mean_recall', 'val_0.9_mean_recall',
'val_0.5_vs_0.7_recall_pval', 'val_0.5_vs_0.9_recall_pval',
'val_0.5_vs_0.7_recall_qval', 'val_0.5_vs_0.9_recall_qval']
n += [i.replace("recall", "fns") for i in n]
n += [i.replace("val_", "test_") for i in n]

rec_dat = {}
for i in n:
    rec_dat[i] = []


print("Predicting...")
alprds = []
for i in range(10):
    print("... using model", i)
    
    with open("final_models/r1/fold_{}_save_model_based_on_MCC_loss_and_bin_best_all_plusMCC5.pkl".format(i), "rb") as f:    
        Ws = pickle.load(f)

    decoderx.set_weights(Ws)  
    
        
    print("reading input file...")
    files = ["seqs_and_targets/model_{}_val_seqs.txt".format(i), "seqs_and_targets/model_{}_test_seqs.txt".format(i)]
    target_files = ["seqs_and_targets/target_data_{}_val_seqs.pkl".format(i), "seqs_and_targets/target_data_{}_test_seqs.pkl".format(i)]
    
    for jj,ii in enumerate(files):
        s, r = read_input_file(ii)
    
        if len(r) > 0:
            print("please ensure that the sequences only contains these amino acids:")
            print(vocab)
            print("\n")
            print("and that they only contain between 2-200 residues")
            print("\n")
            print("these sequences were not compatible with the above craiteria:")
            for h in r:
                print(h)
            sys.exit()
        
        # getting target data
        with open(target_files[jj], "rb") as f:    
            ys = pickle.load(f)
        ys = np.concatenate(ys[4:], axis=-1)
        
        
        print("encoding peptide sequences...")
        X,xnov = encode_seqs_only(s,200,vocab)
        X = X.reshape((-1,4000,1))
          
    
        tmp_ = decoderx.predict(X)
        tmp_ =  np.concatenate(tmp_[4:],axis=-1)

        if jj == 0:
            typ = "val"
        else:
            typ = "test"
            
        for iii in th:
            tmp_prec = []
            # predictions above threshold iii
            tmp_predictions = np.asarray(tmp_ > iii,dtype=np.float64)
            
            # counting false positives per class
            fn_count_total = np.sum((1 - tmp_predictions) * ys, axis=0)
            TP = np.sum(ys * tmp_predictions, axis=0)
            
            print("\t\tmean recall score:", np.mean(TP / (TP+ fn_count_total)))
            print("\t\tmean TPs:", np.mean(TP))
            
            rec_dat["{}_{}_mean_fns".format(typ, iii)].append(np.mean(fn_count_total))
            
            # getting recall score for every class
            for iiii in range(len(nms)):
                pr = recall_score(ys[:,iiii],  tmp_predictions[:,iiii])
                tmp_prec.append(pr)
            
            
            rec_dat["{}_{}_mean_recall".format(typ, iii)].append(np.mean(tmp_prec))
            
            if iii == 0.5:
                pr05 = tmp_prec
                fn05 = fn_count_total
                
            if iii > 0.5:
                rec_dat["{}_{}_vs_{}_recall_pval".format(typ, 0.5, iii)].append(wilcoxon(pr05, tmp_prec)[1])
                rec_dat["{}_{}_vs_{}_fns_pval".format(typ, 0.5, iii)].append(wilcoxon(fn05, fn_count_total)[1])
        
pval_names = []
fns_count_names = []
for k,v in rec_dat.items():
    if "_pval" in k:
        pval_names.append(k)
        
for i in pval_names:
    print(i)
    rec_dat[i.replace("pval", "qval")] = multi.fdrcorrection(rec_dat[i], alpha=0.01, method = 'indep')[1]




df = pd.DataFrame(rec_dat)
df = df[n]
df.index = ["model_{}".format(i) for i in range(10)]
df.to_excel("recall_data.xlsx")

