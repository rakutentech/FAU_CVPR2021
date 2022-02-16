from imutils import face_utils
import numpy as np
import cv2,os


#from DISFA_data_prep import batch_generator_disfa,get_slice_data
from dataloaders import batch_generator,batch_generator_im
import pandas

# Keras libraries
from  keras import optimizers
from keras import backend as K
from keras import backend as K1
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import *
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


K.clear_session()
import tensorflow.compat.v1 as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

from model import baseline_model_best
from config import *


BS=80

file = open(imagelistfile, "r")
content_list = file. readlines()

images =content_list

Total_att_acc =[];Total_perc_acc=[];Total_att_f1 =[];Total_perc_f1=[]
index_list = [];index_im_list = []
for _ in range(3):
    index_list.append([])
    index_im_list.append([])
for im in range(len(images)):    
    if images[im].split('_')[0] in fold1:
        index_im_list[0].append(images[im])
    elif images[im].split('_')[0] in fold2:
        index_im_list[1].append(images[im])   
    elif images[im].split('_')[0] in fold3:
        index_im_list[2].append(images[im])

for ind in range(3):  
    x_test = index_im_list[ind]
    new = list(range(3))
    new.remove(ind)
    
    batches_test = [f for f in os.listdir(data_path) if f.startswith(dataset+'_fold'+str(ind))]

    batches_train = [f for f in os.listdir(data_path) if f.startswith(dataset+'_fold'+str(new[0]))]
    
    batches_train += [f for f in os.listdir(data_path) if f.startswith(dataset+'_fold'+str(new[1]))]
        
    y_train = np.zeros((len(batches_train)*64,AU_count ), dtype='float32')
    row=0
    for batch_iter in batches_train:
        t_t=np.load(data_path+batch_iter)['y_train'][:,ind1]
        y_train[row:row+t_t.shape[0],...]=t_t
        row+=t_t.shape[0]
        
    # Generators
   
    tp = np.sum(y_train,axis=0)
    fp = np.sum((1-y_train),axis=0)
    Pni = (tp+1e-6)/(tp+fp+1e-6)
    alpha = 1/(AU_count*Pni)
    Psi = np.power(alpha,0.5)*Pni
    weights = np.sum(Psi)/Psi


    # Compile model
    model=baseline_model_best(AU_count,weights*0.1,ind)
    if ind==0:
        model.save_weights('initial.h5')
    else:
        model.load_weights('initial.h5')
    # initialize the optimizer and compile the model
    reduce_lr = ReduceLROnPlateau(monitor='val_per_outputs_12_macro_f1', 
                                  mode='max',factor=0.2,
                                  patience=3, min_lr=0.0000001)
    mcp_save = ModelCheckpoint('models/Transformer_FAU_fold{}.h5'.format(ind), 
                               save_best_only=True, save_weights_only=True,
                               monitor='val_att_outputs_macro_f1', mode='max')
    if ind>=0:       
        model.fit_generator(batch_generator(batches_train), 
                            validation_data= batch_generator(batches_test), 
                            epochs = 20, steps_per_epoch=len(batches_train) ,
                            validation_steps=len(batches_test),callbacks=[reduce_lr,mcp_save])
        
    batchsize = BS
    Nooftimes = len(batches_test)
    y_pred_acc = [];y_pred_f1 = np.array([]);y_true_ex = []
    from sklearn.metrics import roc_curve
    model.load_weights('models/Transformer_FAU_fold{}.h5'.format(ind))
    for no in range(Nooftimes):

        print('Predicting '+str(no)+' of '+str(Nooftimes))
       
        x = np.load(data_path+batches_test[no])
        x_train1 = x['x_train1']
        x_train1 = np.concatenate((x_train1,x_train1,x_train1),axis=-1)
        x_train_att = x['x_train_att'];y_train1 = x['y_train']
        x_train1_att = x_train_att[...,ind1]/255.
       
        y_predict = model.predict(x_train1)
        y_test = x['y_train']
        y_test=y_test[:,ind1]
        y_pred = y_predict[1]

        y_pred_acc.append(y_pred)
        y_true = y_test>0.5

        y_true_ex.append(y_true)
        
    y_pred_acc = np.vstack(y_pred_acc)
    y_true_ex = np.vstack(y_true_ex)
    att_acc = [];att_f1=[]; perc_acc=[]; perc_f1=[]
    for l in range(AU_count):      
        a = np.multiply(y_true_ex[:,l],1)
        b = np.multiply(y_pred_acc[:,l]>0.5,1)
        TP = sum(np.multiply(np.logical_and(a==1,b==1),1))
        FP = sum(np.multiply(np.logical_and(a==0,b==1),1))
        FN = sum(np.multiply(np.logical_and(a==1,b==0),1))
        TN = sum(np.multiply(np.logical_and(a==0,b==0),1))
        Precision = TP/(TP+FP+0.00001);print('TP',TP,'FP',FP,'FN',FN,'TN',TN)
        Recall = TP/(TP+FN+0.00001)
        att_f1.append(2*Precision*Recall/(Precision+Recall+0.000001))
        att_acc.append((TP+TN)/(TP+TN+FP+FN+0.0000001))


    print(' accuracy score ....', att_acc)
    print(' f1 score ....', att_f1)
    Total_att_acc.append(att_acc)
    Total_att_f1.append(att_f1)
    model.trainable=True
    
    
print('AVERAGE ATTENTION ACCURACY.....',Total_att_acc)


print('AVERAGE ATTENTION F1 SCORE.....',Total_att_f1)

