# BP4D
import os,cv2#,dlib
import numpy as np
from imutils import face_utils
from keras import backend as K
mapping = {0:1,1:2,2:4,3:6,4:7,5:10,6:12,7:14,8:15,9:17,10:23,11:24}#,12:18,13:20,14:23,15:24,16:27}    
# ddetector = dlib.get_frontal_face_detector()
# dpredictor = dlib.shape_predictor('../Expression/dlibFacialLandmarks/facial-landmarks/shape_predictor_68_face_landmarks.dat')
import random 
import tensorflow as tf
from sklearn.feature_extraction import image
import keras
from keras.callbacks import Callback

from config import *


def batch_generator_im(x,batch_size=80,mode = False):
    ''' mode : flip set or not  
    '''
    attsize=12
    ind = ind1
    
    while True:
        sample_idx = 0
        batch = 0;
        print('entering ... ',len(x))
        
        while sample_idx  < len(x):
            x_train1 = np.zeros((batch_size,224,224,3))
            y_train1 = np.zeros((batch_size,18))
            btc = batch_size
            if sample_idx+batch_size>len(x):
                btc = len(x)-sample_idx
            for i in range(btc):
#                 print(x[sample_idx])
                x_im = ('_').join(x[sample_idx].split('_')[:-1])
                item_n = x[sample_idx].split('_')[-1]
                l = np.load(data_path+x_im+'.npz')
                xm = l['x_train1'][int(item_n),...];
                xm = np.concatenate((xm,xm,xm),axis=-1);
                x_train1[i,...]=xm
                
                y_train1[i,...] = l['y_train'][int(item_n),...]
                sample_idx += 1
            y_train1 = np.expand_dims(y_train,axis=-2)
            batch+=1  
            yield(x_train1, { "per_outputs_{}".format(AU_count): y_train, "att_outputs":y_train,"feat_outputs_{}".format(AU_count):y_train})

        

def batch_generator(x):
    ''' mode : flip set or not  
    '''   
    attsize=12
    i=0
    ind = ind1    
    while True:
        sample_idx = 0
        batch = 0
        print('entering ... ',len(x))
        while sample_idx  < len(x):
            
            l = np.load(data_path+x[sample_idx])
            x_train1 = l['x_train1']
            x_train1 = np.concatenate((x_train1,x_train1,x_train1),axis=-1)
            x_train_att = l['x_train_att']
            x_train_att = np.resize(x_train_att,(64,attsize,attsize,18))
            y_train1 = l['y_train']
            x_train1_att = x_train_att[...,ind]/255.
            y_train=y_train1[:,ind]
            y_train1 = np.expand_dims(y_train,axis=-2)
            y_train2 = y_train            
            sample_idx += 1
            batch+=1  
            yield(x_train1, {  "att_outputs":y_train,"per_outputs_{}".format(AU_count): y_train, "att_loss":x_train1_att,"feat_outputs_{}".format(AU_count):y_train})


def batch_generator_val(x):
    AU_count =12
    ind = ind1 
    while True:
        sample_idx = 0
        batch = 0;
        print('entering ... ',len(x))
        
        while sample_idx  < len(x):
            
            l = np.load(data_path+x[sample_idx])
            x_train1 = l['x_train1'];x_train_att = l['x_train_att'];y_train1 = l['y_train']
            x_train1_att = x_train_att[...,ind]/255.
            y_train=y_train1[:,ind]
            y_train1 = np.expand_dims(y_train,axis=-2)
            y_train2 = y_train
            sample_idx += 1
            batch+=1  
            yield([x_train1,x_train1_att], { "per_outputs_{}".format(AU_count): y_train,"att_outputs": y_train,"att_loss": x_train1_att, "feat_outputs_{}".format(AU_count):y_train})