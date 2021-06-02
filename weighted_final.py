# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:00 2020
"""


seed_val = 1234
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout,Multiply, LSTM, Add, Concatenate, TimeDistributed
from tensorflow.keras.layers import Conv1D, Flatten, Lambda, MaxPooling1D, GRU, SimpleRNN, PReLU
from tensorflow.keras.layers import Reshape 
from tensorflow.keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Constant
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_score, recall_score,f1_score,cohen_kappa_score,accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time


### Controlling randomness
tf.random.set_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)


#### Readaing peptide sequence data
with open("total_classes_seq_bins32.pkl","rb") as f:
    tot_bins = pickle.load(f)
    
with open("total_classes_seq32.pkl","rb") as f:
    encoder = pickle.load(f)

with open("branches32.pkl","rb") as f:
    branches = pickle.load(f)
branches = [list(i) for i in branches]

lvl1 = [list(branches[0]+branches[1]), list(branches[2]+branches[3]+branches[4])]

lvl2_1 = [list(branches[0]),list(branches[1])]
lvl2_2 = [list(branches[2]),list(branches[3]+branches[4])]

lvl3 = [list(branches[3]),list(branches[4])]

lvl4_1 = [list(branches[0])]
lvl4_2 = [list(branches[1])]
lvl4_3 = [list(branches[2])]
lvl4_4 = [list(branches[3])]
lvl4_5 = [list(branches[4])]

levels = [lvl1,    lvl2_1, lvl2_2,    lvl3,    lvl4_1, lvl4_2, lvl4_3, lvl4_4, lvl4_5]


def items_per_key(dct):
    print("\n")
    for k,v in dct.items():
        print(k,len(v))


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


folds = [
 [[0], [1], [2, 3, 4, 5, 6, 7, 8, 9]],
 [[1], [2], [0, 3, 4, 5, 6, 7, 8, 9]],
 [[2], [3], [0, 1, 4, 5, 6, 7, 8, 9]],
 [[3], [4], [0, 1, 2, 5, 6, 7, 8, 9]],
 [[4], [5], [0, 1, 2, 3, 6, 7, 8, 9]],
 [[5], [6], [0, 1, 2, 3, 4, 7, 8, 9]],
 [[6], [7], [0, 1, 2, 3, 4, 5, 8, 9]],
 [[7], [8], [0, 1, 2, 3, 4, 5, 6, 9]],
 [[8], [9], [0, 1, 2, 3, 4, 5, 6, 7]],
 [[9], [0], [1, 2, 3, 4, 5, 6, 7, 8]]
 ]

mx = 200
mi = 2


labels1 = set()
for i in list(encoder.values()):
    labels1.update(i)
labels1 = list(labels1)
labels1.sort()

labels =  set()
for k,v in tot_bins.items():
    #print(v)
    tmp = k.split(" | ")
    tmp.remove("")
    labels.update(tmp)
labels = list(labels)
labels.sort()

print(labels1)
print("------")
print(labels)

assert labels == labels1
del labels1


#############################################################
## Functions

def making_ys(activity,levels):
    lvls = []
    for l in levels:
        lvls.append([])

    for j,l in enumerate(levels):
        if len(l) == 2:
            lab = np.zeros((len(l),))
            for jj,ll in enumerate(l): 
                if activity in ll:          
                    lab[jj] = 1
            lvls[j].append(lab)

        if len(l) == 1:
            lab = np.zeros((len(l[0]),))
            if activity in l[0]: 
                lab[l[0].index(activity)] = 1
            lvls[j].append(lab)
    return lvls
                
    
def encode_seqs(sq_dct, encoder, max_, voc, ac_over, levels):

    lnv = len(voc)
    #dims = len(ac_over)
    alsqs = []
    y_list = []
    x_list = []

    for sq in sq_dct:
        tmp = []
        for ii in encoder[sq]:
            tmp2 = making_ys(ii, levels)
            if len(tmp) == 0:
                for iii in tmp2:
                    tmp.append(iii)
            else:
                for iii in range(len(tmp2)):
                    #print(sq,tmp[iii])
                    tmp[iii][0] += tmp2[iii][0]
        for ii in tmp:
            ii[0][ii[0] > 0] = 1
        y_list.append(tmp)

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

    return np.concatenate(x_list,axis=0), y_list, alsqs


def folder(folder, tot_bins):
    train_to_encode = []
    val_to_encode = []
    test_to_encode = []
    
    for k,v in tot_bins.items():
        for i in folder[-1]:
            train_to_encode += v[i]
        val_to_encode += v[folder[0][0]]
        test_to_encode += v[folder[1][0]]
        
    return train_to_encode, val_to_encode, test_to_encode


def y_sets(y,levels):
    lvls = [[] for i in levels]
    for j,i in enumerate(y):
        for jj,ii in enumerate(i):
            lvls[jj].append(ii)
        
    for j,i in enumerate(lvls):
        lvls[j] = np.concatenate(lvls[j],axis=0)
    return lvls


def make_rn(x):
    rn = np.arange(len(x))
    np.random.shuffle(rn)
    return rn


def res(X):
    X = X.reshape((len(X),len(X[0]),1))
    return X


def sorter(a,b):
    c = list(zip(a,b))
    c.sort()
    a1,b1 = zip(*c)
    b1 = list(b1[::-1])
    a1 = list(a1[::-1])
    return a1, b1


def estimate_acc(pred,y):
    
    acc = []
    for i in range(len(y)):
        a,b = sorter(pred[i],np.arange(len(pred[i])))
        a1,b1 = sorter(y[i],np.arange(len(y[i])))
        ln = len(y[i][y[i] > 0])
        ac = len(set(b1[:ln]).intersection(set(b[:ln])))/len(b1[:ln])
        acc.append(ac)
    return acc


binary_cross = BinaryCrossentropy()#reduction="sum")
binnz = K.binary_crossentropy


def changer(inp):
    inpp = K.relu((inp-0.5)*(-1))
    inpp = inpp/(inpp + K.epsilon())
    inpp = inp*inpp
    inpp = inpp*inpp
    
    inpp1 = K.relu(inp,threshold=0.5)
    return inpp+inpp1



def sorting(xx):
    tmp = list(zip(xx,np.arange(len(xx))))
    tmp.sort()
    x,lst = zip(*tmp)
    return lst


def calc_real_acc(yt,tmp_val):
    return np.sum(yt == np.round(tmp_val))/(yt.shape[0] * yt.shape[1])

def print_function(name,x,y):
    print(name,"||",sep=" ", end=" ")
    for i in range(len(x)):
        print(x[i], np.round(y[i],4),"|",sep=" ", end=" ")
    print("")
    try:
        print("average:", np.average(y)) 
    except:
        print("average:", np.average(y[1:])) 
    print("\n")

def calc_score(yt,tmp_val,funk):
    out = []
    for i in range(len(yt)):
        indx = np.sum(yt[i],axis=-1)
        ind = indx > 0
        out.append(np.average(funk(tmp_val[i][ind], yt[i][ind])))
    return out

def calc_score_wzero(yt,tmp_val,funk):
    out = []
    out0 = []
    for i in range(len(yt)):
        indx = np.sum(yt[i],axis=-1)
        ind = indx > 0
        ind0 = indx == 0
        out.append(funk(yt[i][ind], tmp_val[i][ind]))
        if np.sum(ind0) > 0:
            out0.append(funk(yt[i][ind0], tmp_val[i][ind0]))
        if np.sum(ind0) == 0:
            out0.append([])
    return out, out0


def calc_roc(yt,tmp_val,funk):
    out = []
    out0 = []
    for i in range(len(yt)):
        indx = np.sum(yt[i],axis=-1)
        ind = indx > 0
        out.append(funk(yt[i][ind], tmp_val[i][ind]))
        out0.append(funk(yt[i], tmp_val[i]))
    return out, out0


def calc_score_wzero_round(yt,tmp_val,funk):
    out = []
    out0 = []
    for i in range(len(yt)):
        indx = np.sum(yt[i],axis=-1)
        ind = indx > 0
        ind0 = indx == 0
        out.append(funk(yt[i][ind], np.round(tmp_val[i][ind])))
        if np.sum(ind0) > 0:
            out0.append(funk(yt[i][ind0], np.round(tmp_val[i][ind0])))
        if np.sum(ind0) == 0:
            out0.append([])
    return out, out0

def per_pred(yv,tmp_val,funk, name):
    mccs = []
    for iq in range(len(yv)):
        mccs.append([funk(yv[iq][:,iqq],np.round(tmp_val[iq][:,iqq])) for iqq in range(len(yv[iq][0]))])
    for iq in range(len(mccs)):
        print(level_names[iq])
        for iqq in mccs[iq]:
            print(np.round(iqq,4), sep=" ", end=" ")
        print("")
    all_mcc = []
    for iq in mccs:
        all_mcc += iq
    all_mcc1 = np.prod(all_mcc)
    all_mcc2 = np.average(all_mcc)
    print("\naverage {}:".format(name), all_mcc2, "| prod", all_mcc1)
    print("\naverage {} for leaves:".format(name), np.average(all_mcc[-len(labels):]), "| prod", np.prod(all_mcc[-len(labels):]))
    return all_mcc2

def per_pred2(yv,tmp_val,funk, name):
    mccs = []
    for iq in range(len(yv)):
        mccs.append([funk(yv[iq][:,iqq],np.round(tmp_val[iq][:,iqq])) for iqq in range(len(yv[iq][0]))])
    for iq in range(len(mccs)):
        print(level_names[iq])
        for iqq in mccs[iq]:
            print(np.round(iqq,4), sep=" ", end=" ")
        print("")
    all_mcc = []
    for iq in mccs:
        all_mcc += iq
    all_mcc1 = np.prod(all_mcc)
    all_mcc2 = np.average(all_mcc)
    #print("\naverage {}:".format(name), all_mcc2, "| prod", all_mcc1)
    #print("\naverage {} for leaves:".format(name), np.average(all_mcc[-len(labels):]), "| prod", np.prod(all_mcc[-len(labels):]))
    return all_mcc2



#############################################################
#### Intializing the network
#### Intializing the network
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



def mena_binn_loss(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    return K.sum(K.mean(binnz(y_true, y_pred), axis=0))




lossesx = [mena_binn_loss for i in levels]
#train_losses = [loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8]


opt = Adam(learning_rate=0.0005) 
decoderx = Model(inputs, outputs)
decoderx.compile(optimizer=opt, loss=lossesx)#, #metrics=["binary_accuracy"])#'mean_squared_error')
#decoderx.summary()



#############################################################
### Running the network
accc = 100000.
toxl = 100000.
ambl = 100000.
pepl = 100000.
hypl = 100000.
enzl = 100000.
total_loss = 100000



level_names = ["lvl1", "lvl2_1", "lvl2_2","lvl3", "lvl4_1", "lvl4_2", "lvl4_3", "lvl4_4", "lvl4_5"]
ws = decoderx.get_weights()
start_ws = ws.copy()
order = np.arange(len(levels))




breaker = 0
losseZ = []
losseZ_train = []
for fs in range(len(folds)):
    
    
    accc = 100000.
    toxl = 100000.
    ambl = 100000.
    pepl = 100000.
    hypl = 100000.
    enzl = 100000.
    total_loss = 100000

    
    
    train_to_encode, val_to_encode, test_to_encode = folder(folds[fs], tot_bins)

    xtr, ytr, xtrsq = encode_seqs(train_to_encode,encoder, mx, vocab, labels,levels)
    xv, yv, xvsq  = encode_seqs(val_to_encode,encoder, mx, vocab, labels,levels)
    xt, yt ,xtsq = encode_seqs(test_to_encode,encoder, mx, vocab, labels,levels)
    
    tmp = []
    for i in [ytr,yv,yt]:
        tmp.append(y_sets(i,levels))
    ytr,yv,yt = tmp
    
    
    weights = []
    for i in range(len(levels)):
        tmp = []
        for ii in range(len(ytr[i][0])):
            w = compute_class_weight("balanced",classes=[0,1], y=ytr[i][:,ii])[1]
            tmp.append(w)
            if i < 4:
                print(levels[i][ii],"\t",w)
            else:
                print(levels[i][0][ii],"\t",w)
        weights.append(tmp)
    weights = np.array(weights)
        
    
    def loss_0(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[0],axis=0))

    def loss_1(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[1],axis=0))
    
    def loss_2(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[2],axis=0))
    
    def loss_3(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[3],axis=0))
    
    def loss_4(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[4],axis=0))
    
    def loss_5(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[5],axis=0))
    
    def loss_6(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[6],axis=0))
    
    def loss_7(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[7],axis=0))
    
    def loss_8(y_true, y_pred):
        return K.sum(K.mean(binnz(y_true, y_pred)*weights[8],axis=0))
    
    print("Initializing the network with class weights based on the training data")
    train_losses = [loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8]
    opt = Adam(learning_rate=0.0005) 
    decoderx = Model(inputs, outputs)
    decoderx.compile(optimizer=opt, loss=train_losses)
    

    
    with open("numpy_mullab_data_tree_final_weighted_{}.pkl".format(fs),"wb") as f:
        pickle.dump([xtr, ytr, xtrsq, xv, yv, xvsq, xt, yt ,xtsq], f)
        print("numpy data saved")
    
    
    tmpenc = [xtr,xv,xt]
    tmpsq = [xtrsq,xvsq,xtsq]
    for i in range(3):
        assert len(tmpenc[i]) == len(tmpsq[i])
        for ii in range(len(tmpenc[i])):
            assert len(tmpsq[i][ii]) == np.sum(tmpenc[i][ii])
    del(tmpenc)
    del(tmpsq)
    
    rntr = make_rn(xtr)
    rnv = make_rn(xv)
    rnt = make_rn(xt)
    
    all_ys = [i[rntr] for i in ytr]
    
    xtr = res(xtr)
    xv = res(xv)
    xt = res(xt)
    
    
    print(xtr.shape)
    for i in (ytr):
        print(i.shape)
    print("\n")
    print(xv.shape)
    for i in (yv):
        print(i.shape)     
    print("\n")
    print(xt.shape)
    for i in (yt):
        print(i.shape) 
    
    
    decoderx.set_weights(start_ws)
    
    name1 = "fold_"+str(fs)+"_save_model_based_weighted_loss_"
    
    for i in range(500):
        print("\nEPOCH", i)
        
        decoderx.fit(xtr[rntr], all_ys, verbose=0, #validation_data=(xv, yv),
                        epochs=1,
                        batch_size=128, use_multiprocessing=True, workers=2)#, class_weight=wdct)#,callbacks=my_callbacks)
        
        rntr = make_rn(xtr)
        all_ys = [i[rntr] for i in ytr]
        
        start = time.time()
        tmp_val = decoderx.predict(xv, batch_size=128)#, use_multiprocessing=True, workers=3)
        end = time.time()
        print(end - start)
        
        
        val_loss_bin = [mena_binn_loss(yv[iq],tmp_val[iq]).numpy() for iq in range(len(yv))]
        print_function("val_loss_bin",level_names,val_loss_bin)
        losseZ.append([val_loss_bin])
        
        ww = decoderx.get_weights()
        
        #val_loss_bin = val_loss
        
        print("\n")
        all_mcc2 = per_pred(yv,tmp_val,matthews_corrcoef,"VAL_MCC")
        
        if val_loss_bin[4] < toxl:
            print("\n\tsaving best tox_hem model",val_loss_bin[4])
            decoderx.save_weights(name1+"best_toxhem.h5")
            toxl = val_loss_bin[4]
            breaker = -1
            ws[28:35] = ww[28:35]
            ws[49:56] = ww[49:56]
            ws[76] = ww[76]
            ws[77] = ww[77]
            ws[82] = ww[82]
            ws[91:93] = ww[91:93]
            ws[97] = ww[97]
            ws[106:108] = ww[106:108]
            ws[112] = ww[112]
            ws[119:121] = ww[119:121]
        
        if val_loss_bin[5] < ambl:
            print("\n\tsaving best ambl model",val_loss_bin[5])
            decoderx.save_weights(name1+"best_ambl.h5")
            ambl = val_loss_bin[5]
            breaker = -1
            ws[35:42] = ww[35:42]
            ws[56:63] = ww[56:63]
            ws[78] = ww[78]
            ws[79] = ww[79]
            ws[83] = ww[83]
            ws[93:95] = ww[93:95]
            ws[98] = ww[98]
            ws[108:110] = ww[108:110]
            ws[113] = ww[113]
            ws[121:123] = ww[121:123]
    
    
        if val_loss_bin[6] < pepl:
            print("\n\tsaving best pepl model",val_loss_bin[6])
            decoderx.save_weights(name1+"best_pepl.h5")
            pepl = val_loss_bin[6]
            breaker = -1
            ws[42:49] = ww[42:49]
            ws[63:70] = ww[63:70]
            ws[80] = ww[80]
            ws[81] = ww[81]
            ws[84] = ww[84]
            ws[95:97] = ww[95:97]
            ws[99] = ww[99]
            ws[110:112] = ww[110:112]
            ws[114] = ww[114]
            ws[123:125] = ww[123:125]
            
        if val_loss_bin[7] < hypl:
            print("\n\tsaving best hyp model",val_loss_bin[7])
            decoderx.save_weights(name1+"best_hyp.h5")
            hypl = val_loss_bin[7]
            breaker = -1
            ws[7:14] = ww[7:14]
            ws[21:28] = ww[21:28]
            ws[72] = ww[72]
            ws[73] = ww[73]
            ws[75] = ww[75]
            ws[87:89] = ww[87:89]
            ws[90] = ww[90]
            ws[102:104] = ww[102:104]
            ws[105] = ww[105]
            ws[117:119] = ww[117:119]
            
        if val_loss_bin[8] < enzl:
            print("\n\tsaving best enz model",val_loss_bin[8])
            decoderx.save_weights(name1+"best_enz.h5")
            enzl = val_loss_bin[8]
            breaker = -1
            ws[:7] = ww[:7]
            ws[14:21] = ww[14:21]
            ws[70] = ww[70]
            ws[71] = ww[71]
            ws[74] = ww[74]
            ws[85:87] = ww[85:87]
            ws[89] = ww[89]
            ws[100:102] = ww[100:102]
            ws[104] = ww[104]
            ws[115:117] = ww[115:117]
        
        if np.sum(val_loss_bin) < total_loss:
            total_loss = np.sum(val_loss_bin)
            print("\n\tsaving best overal best model",np.sum(val_loss_bin))
            decoderx.save_weights(name1+"best_overall_best.h5")
        
        with open(name1+"best_all.pkl", "wb") as f:    
            pickle.dump(ws,f)
        #decoderx.set_weights(ws)
    
        with open(name1+"losses.pkl", "wb") as f:    
            pickle.dump(losseZ,f)
            
        with open(name1+"losses_train.pkl", "wb") as f:    
            pickle.dump(losseZ_train,f)
            
        breaker += 1
        if breaker == 20:
            break
        print("\n")


