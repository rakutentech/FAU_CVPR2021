from imutils import face_utils
#import datetime;from vis.utils import utils
import numpy as np
from keras.optimizers import Adam 
from sklearn.utils import class_weight
from keras.layers import Input, Activation, Conv2D,Conv2DTranspose, Flatten, Dense, MaxPooling2D,Multiply, UpSampling2D,BatchNormalization,concatenate,Concatenate,ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import dot, Reshape,RepeatVector, multiply,Lambda,add
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from  keras import optimizers
import pandas,dlib
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from keras.backend.tensorflow_backend import set_session

# Enter you image and label paths below; image_folder should contain the original images 
# and label_folder contains labels with same name as that in image_folder
image_folder = '../Prepared_data/data1/images/'
label_folder = '../Prepared_data/data1/labels/'



import cv2,os
ddetector = dlib.get_frontal_face_detector()
# dlib  library path
dpredictor = dlib.shape_predictor('../dlibFacialLandmarks/facial-landmarks/shape_predictor_68_face_landmarks.dat')


AU_count =12

batch_size=64

aulist_BP4D = [1,2,4,6,7,10,12,14,15,17,23,24]
mapping_BP4D = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',2:'Brow Lowerer',
                3:'Cheek raiser',4:'Lid Tightener',
                5:'Upper Lip Raiser',6:'Lip Corner Puller',
                7:'Dimpler',8:'Lip Corner Depressor',9:'Chin Raiser',
                10:'Lip Tightener',11:'Lip pressor'}
aulist_DISFA = [1,2,4,6,9,12,25,26]
mapping_DISFA = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',
                 2:'Brow Lowerer',3:'Cheek raiser',4:'Nose Wrinkler',
                 5:'Lip Corner Puller',6:'Lips Part',7:'Jaw Drop'}

aulist_EmotioNet = [1,2,4,5,6,9,12,17,20,25,26,43]
mapping_EmotioNet = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',
                     2:'Brow Lowerer',3:'Upper Lid Raiser',4:'Cheek Raiser', 
                     5:'Nose Wrinkler', 6:'Lip Corner Puller',7:'Chin Raiser', 
                     8:'Lip Stretcher',9:'Lips Part',10:'Jaw drop',11:'Eyes Closed'}

aulist = [1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26,43]
mapping = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',
           2:'Brow Lowerer',3:'Upper Lid Raiser',
           4:'Cheek Raiser', 5:'Lid Tightener',
           6:'Nose Wrinkler', 7: 'Upper Lip Raiser', 
           8:'Lip Corner Puller',9:'Dimpler',10:'Lip Corner Depressor', 
           11:'Chin Raiser', 12: 'Lip Stretcher',13:'Lip Tightener',
           14:'Lip pressor',15:'Lips Part',16:'Jaw Drop',17:'Eyes Closed'}

def AU_plot_ellipsoid(gray1,au,x,shapes):
    [x1,y1,x2,y2,w,h] = x
    
    att_map = np.zeros((gray1.shape[0],gray1.shape[1]))
    
    if au==0:
        (l_x1,l_y1) = (shapes[20])
        (r_x2,r_y2) = (shapes[23])
        cv2.ellipse(gray1,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==1:
        (l_x1,l_y1) = (shapes[18])
        (r_x2,r_y2) = (shapes[25])
        cv2.ellipse(gray1,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1,l_y1),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x2,r_y2),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
       
    elif au==2:
        l_x,l_y = (shapes[19])
        r_x,r_y = (shapes[24])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==3:        
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47])
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1-l_y1)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1-l_y1)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2-l_y2)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2-l_y2)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


        
    elif au==4:
        (l_x1,l_y1) = (shapes[41])
        (r_x1,r_y1) = (shapes[46])
        cv2.ellipse(gray1,(l_x1-round(w/10),l_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x1-round(w/10),r_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x1-round(w/10),l_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x1+round(w/10),r_y1+round(h/6)),(round(w/10),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==5:
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47])
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==6:
        (l_x1,l_y1) = (shapes[29])
        (r_x1,r_y1) = (shapes[31])
        (r_x2,r_y2) = (shapes[35])
        
        cv2.ellipse(gray1,(int(r_x1),int(l_y1)),(20,20),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int(r_x2),int(l_y1)),(20,20),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int(r_x1),int(l_y1)),(20,20),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int(r_x2),int(l_y1)),(20,20),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==7:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[63]
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==8:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==9:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
#         print(l_x,l_y,r_x,r_y)
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==10:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        cv2.ellipse(gray1,(l_x,l_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(r_x,r_y),(round(w/8),round(h/10)),0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(l_x,l_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(r_x,r_y),(round(w/8),round(h/10)),
                    0,0,360,(255,255,255),cv2.FILLED)


        
    elif au==11:
        l_x,l_y = (shapes[59])
        r_x,r_y = (shapes[9])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)



    elif au==12:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


    elif au==13:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)


    elif au==14:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==15:
        l_x,l_y = (shapes[48])
        r_x,r_y = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    elif au==16:
        l_x,_ = (shapes[48])
        r_x,_ = (shapes[54])
        _,l_y = (shapes[50])
        _,r_y = (shapes[57])
        cv2.ellipse(gray1,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x+r_x)/2),int((l_y+r_y)/2)),
                    (max(int((r_x-l_x)/2),0),max(int((r_y-l_y)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
    elif au==17:
        (l_x1,_) = (shapes[36])
        (_,l_y1) = (shapes[38])
        (r_x1,_) = (shapes[39])
        (_,r_y1) = (shapes[41])
        (l_x2,_) = (shapes[42])
        (_,l_y2) = (shapes[44])
        (r_x2,_) = (shapes[45])
        (_,r_y2) = (shapes[47]
        cv2.ellipse(gray1,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(gray1,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),2)
        cv2.ellipse(att_map,(int((l_x1+r_x1)/2),int((l_y1+r_y1)/2)),
                    (max(int((r_x1-l_x1)/2),0),max(int((r_y1+10-l_y1+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)
        cv2.ellipse(att_map,(int((l_x2+r_x2)/2),int((l_y2+r_y2)/2)),
                    (max(int((r_x2-l_x2)/2),0),max(int((r_y2+10-l_y2+10)/2),0)),
                    0,0,360,(255,255,255),cv2.FILLED)

    img_h, img_w= np.shape(gray1)
    
    xw1 = max(int(x1 - 10), 0)  #<---left side
    yw1 = max(int(y1 - 10), 0)  #<---head
    xw2 = min(int(x2 + 10), img_w - 1) #<---right side
    yw2 = min(int(y2 + 10), img_h - 1) #<--- bottom

    att_map1 = cv2.resize(att_map[yw1:yw2,xw1:xw2], dsize=(28,28))

    return att_map1




# BP4D 
file = open(imagelistfile, "r")
content_list = file. readlines()

images =content_list

# images =xx
fold1 = ['F001','F002','F008','F009','F010','F016','F018','F023','M001','M004','M007','M008','M012','M014']
fold2 = ['F003','F005','F011','F013','F020','F022','M002','M005','M010','M011','M013','M016','M017','M018']
fold3 = ['F004','F006','F007','F012','F014','F015','F017','F019','F021','M003','M006','M009','M015']

Total_att_acc =[];Total_perc_acc=[];Total_att_f1 =[];Total_perc_f1=[]
index_list = [];index_im_list = []
for _ in range(3):
    index_list.append([])
    index_im_list.append([])
for im in range(len(images)):    
    if images[im].split('_')[0] in fold1:
        index_list[0].append(im);index_im_list[0].append(images[im])
    elif images[im].split('_')[0] in fold2:
        index_list[1].append(im);index_im_list[1].append(images[im])   
    elif images[im].split('_')[0] in fold3:
        index_list[2].append(im);index_im_list[2].append(images[im])

indlist = index_list
y_pred_acc = np.array([]);y_pred_f1 = np.array([]);y_true_ex = np.array([])


   
for foldno in range(3):   
    x = index_im_list[foldno] 
    sample_idx = 0
    batch = 0;
    print('entering ... ',len(x))
    while sample_idx  < len(x):

        x_train = np.zeros((batch_size, 224, 224, 1), dtype='float32')
        y_train = np.zeros((batch_size,AU_count ), dtype='float32')
        x_train_att = np.zeros((batch_size, 28,28,AU_count), dtype='float32')
        row=0
        while row <batch_size:                                   
            if sample_idx>=len(x):
                break
            line=x[sample_idx]
            imname = line.split('.')[0]
            gray = np.load(image_folder+line)
            faces = ddetector(gray)
            if len(faces)>0:
                (x1, y1, w, h) = face_utils.rect_to_bb(faces[0])
                shapes = dpredictor(gray, faces[0])
                shapes = face_utils.shape_to_np(shapes)

                x2 = x1+w
                y2 = y1+h
                img_h, img_w= np.shape(gray)
                xw1 = max(int(x1 -0), 0)  #<---left side
                yw1 = max(int(y1  -0), 0)  #<---head
                xw2 = min(int(x2 + 0), img_w - 1) #<---right side
                yw2 = min(int(y2 + 0), img_h - 1) #<--- bottom
                im_original = gray[yw1:yw2,xw1:xw2].copy()
                y = np.load(label_folder+imname+'.jpg_y_train.npy')
                test1 = y
                test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                test[:3] = test1[:3]
                test[4:6] = test1[3:5]

                test[7:12] = test1[5:10]
                test[13:15] = test1[-2:]
                y = np.array(test,dtype=float)
                full_att_map = np.zeros((1,224,224,AU_count))
                for i in range(AU_count):
                        au = i
                        att_map = AU_plot_ellipsoid(gray,au,[x1,y1,x2,y2,w,h],shapes)
                        att_map = cv2.blur(att_map,(3,3))

                        attmap_resized = cv2.resize(att_map, dsize=(224,224))
                        full_att_map[0,:,:,i] = attmap_resized

                im_rz = np.expand_dims(cv2.resize(im_original, dsize=(224,224)),axis=0) 
                x_train[row,...] = np.expand_dims(im_rz,axis=-1) 
                x_train_att[row,...] = cv2.resize(np.squeeze(full_att_map),dsize=(28,28))/255.
                y_train[row,...] = y  
                index = np.where(y==1)
                row+=1
            sample_idx += 1
        x_train1 = x_train/255.
        x_train1_att = np.expand_dims(x_train_att.max(axis=-1),axis=-1)
        batch+=1        
        print(batch,im_rz.shape,x_train.shape,x_train_att.shape,y_train.shape,
              'DATA/BP4D/BP4D_fold'+str(foldno)+'_'+str(sample_idx))
        np.savez('DATA/BP4D/BP4D_fold'+str(foldno)+'_'+str(sample_idx),x_train1=x_train1,y_train=y_train,x_train_att=x_train_att)



# DISFA
# fold1 = ['SN002','SN010','SN001','SN026','SN027','SN030','SN032','SN009','SN016']
# fold2 = ['SN006','SN011','SN012','SN013','SN018','SN021','SN024','SN028','SN031']
# fold3 = ['SN003','SN004','SN005','SN007','SN008','SN017','SN023','SN025','SN029']

# Total_att_acc =[];Total_perc_acc=[];Total_att_f1 =[];Total_perc_f1=[]
# index_list = [];index_im_list = []
# for _ in range(3):
#     index_list.append([])
#     index_im_list.append([])
# for im in range(len(images)):    
#     if images[im].split('_')[0] in fold1:
#         index_list[0].append(im);index_im_list[0].append(images[im])
#     elif images[im].split('_')[0] in fold2:
#         index_list[1].append(im);index_im_list[1].append(images[im])   
#     elif images[im].split('_')[0] in fold3:
#         index_list[2].append(im);index_im_list[2].append(images[im])



