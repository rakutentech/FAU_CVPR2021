## testing for a single image 
## download trained models from the link
## 
import cv2
im = cv2.imread('DATA/inference/face1.png')
im = np.expand_dims(cv2.resize(im, dsize=(224,224)),axis=0)
print(im.shape)

AU_count =12

mapping_BP4D = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',2:'Brow Lowerer',
                3:'Cheek raiser',4:'Lid Tightener',
                5:'Upper Lip Raiser',6:'Lip Corner Puller',
                7:'Dimpler',8:'Lip Corner Depressor',9:'Chin Raiser',
                10:'Lip Tightener',11:'Lip pressor'}

from tensorflow.keras import backend as K
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam,RMSprop
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras_pos_embd import TrigPosEmbedding,PositionEmbedding
from keras_multi_head import MultiHeadAttention
from transformer import get_encoders,get_decoders
from config import *

base_model = InceptionV3(weights="imagenet", 
                         include_top=False, input_shape= (224,224,3))
base_model = Model(inputs=base_model.input, 
                   outputs = base_model.get_layer('activation_74').output)

def baseline_model_best(AU_count):
    fc_dim = 256
    # create model
    inputs = Input(shape=(224,224,3)) 
    
    #block 1
    g = base_model(inputs)
    
    gh = Conv2D(64, (3,3), padding='same',
                kernel_initializer='glorot_normal')(g)
    gh1 = Conv2D(AU_count, (1,1), padding='same', 
                 kernel_initializer='glorot_normal')(gh)
    gh2 = Conv2D(AU_count, (1,1), padding='same', 
                 activation='sigmoid',name = "att_loss",
                 kernel_initializer='glorot_normal')(gh1)    
    gh1 = Conv2D(AU_count, (1,1), padding='same', 
                 activation='linear',
                 kernel_initializer='glorot_normal')(gh1)
    gap = GlobalAveragePooling2D()(gh1)
    att_output = Activation('sigmoid',name="att_outputs")(gap)
    attention = gh2
    reshape_embed = Reshape([12*12,AU_count])(attention)
    reshape_embed = Permute((2,1))(reshape_embed)
    print(attention.shape)
    for i in range(AU_count):
        
        layer1 = Lambda(lambda x: K.expand_dims(attention[...,i],axis=-1))(attention)
 
        out = Multiply()([layer1,g])
        g = Add()([out,g])
        mt = Conv2D(64, (1,1), padding='same',
                    kernel_initializer='glorot_normal')(g)
        mt = MaxPooling2D(pool_size=7,
                          strides=(1,1),padding = 'same')(mt)
        mt = BatchNormalization()(mt)        
        mt = Activation('relu')(mt)
        perception = Flatten()(mt)
        
        inter = Dense(fc_dim, activation='relu',
                      kernel_initializer='glorot_normal')(perception)
        tin = Lambda(lambda x: K.expand_dims(x,axis=1))(inter)
        
        if i==0:
            feat_outputs = tin
        else:
            feat_outputs= Concatenate(axis = 1,
                                      name = 'feat_outputs_{}'.format(i+1))([feat_outputs,tin])
    
    feat_outputs_P  = PositionEmbedding(
        input_shape=(None,),
        input_dim = AU_count,
        output_dim = fc_dim,
        mask_zero=0,  # The index that presents padding (because `0` will be used in relative positioning).
        mode=PositionEmbedding.MODE_ADD,)(feat_outputs)
    feat_outputs_P = get_encoders(name = '1',encoder_num=3,
                                  input_layer=feat_outputs_P,
                                  head_num=8,hidden_dim=fc_dim,
                                  dropout_rate=0.1,)
    feat_outputs_P = Flatten()(feat_outputs_P)
    inter = Dense(fc_dim, activation='relu',
                  kernel_initializer='glorot_normal')(feat_outputs_P)
    final = Dense(AU_count, activation='sigmoid',
                  name = 'per_outputs_{}'.format(i+1),
                  kernel_initializer='glorot_normal')(inter)
    model = Model(inputs=inputs,
                  outputs=[att_output,final,attention,feat_outputs])#;model.summary()                                                             
    return model


model=baseline_model_best(AU_count)
model.load_weights('models/Transformer_FAU_fold0.h5')
y_predict = model.predict(im)

nd = np.where(y_predict[1]>0.5)
for i in ind[1]:
    print(mapping_BP4D[i])