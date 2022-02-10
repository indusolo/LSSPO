#!/usr/bin/env python
# coding: utf-8

# #### German credit dataset

# In[1]:


# Necessary imports
import pandas as pd
import numpy as np
from numpy import genfromtxt
import csv
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.models import load_model
from scipy.io import arff
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time
import random
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Layer
#from keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.losses import mse, binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy
from keras.layers import Activation, Dense, Flatten 
from tensorflow.keras import backend as K
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(3)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import scipy
from scipy.stats import multivariate_normal


# In[2]:


import scipy.io
mat = scipy.io.loadmat('GER_TO_USE (1).mat')


# In[3]:


# Train and Test data - 70% Train and 30% Test
x_train = mat['train_x'] 
ytrn = mat['train_y'] 
y_train =  ytrn[:,0]*2+ytrn[:,1]*1

x_test = mat['test_x'] 
ytst = mat['test_y'] 
y_test =  ytst[:,0]*2+ytst[:,1]*1


# In[4]:


mmscaler    = MinMaxScaler(feature_range=(-1.5,1.5))
x_train   = mmscaler.fit_transform(x_train)
x_test = mmscaler.transform(x_test)


# ### VAE

# #### VAE code is adopted and modified from the following reference.
# #### https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py 

# In[5]:


start_dimension = x_train.shape[1]
# network parameters
input_shape = (start_dimension, )
intermediate_dim = 18
batch_size = 24 
latent_dim = 6
epochs = 1000


# In[6]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[7]:


# Encoder model
inputs = keras.Input(shape=input_shape, name='encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# In[8]:


# Decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='tanh')(latent_inputs)
outputs = layers.Dense(start_dimension, activation='tanh')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# In[9]:


# VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')


# In[10]:


# VAE Train validation split
X_imtrain, X_imval, y_imtrain, y_imval = train_test_split(x_train, x_train, 
                                                    test_size=0.2, random_state=1)


# In[11]:


start_time = time.time()


# In[12]:


if __name__ == '__main__':
    
    loss = 'mse'
    models = (encoder, decoder)
    
    if loss == 'bce':
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
    else:
        reconstruction_loss = mse(inputs, outputs)
        
    reconstruction_loss *= start_dimension
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    opt = keras.optimizers.Adam(lr =0.0001)
    vae.compile(optimizer=opt,)
    vae.summary()
            
    vae.fit(X_imtrain,
            epochs=epochs,
            #verbose = 10,
            batch_size=batch_size,
            validation_data=(X_imval, None))
    


# In[13]:


def latent_space_data(vae, data):
    encoder, decoder = vae
    z_mean, _, _ = encoder.predict(data)
    return z_mean
LS_Ger_train = latent_space_data(models, x_train)
LS_Ger_test = latent_space_data(models, x_test)


# ### MSPO

# In[14]:


#def SPOfn(latent_space_im,mino,majo,y_d):
   

latent_space_im = LS_Ger_train
mino = 1
majo = 2
y_d = y_train

nTarget = np.sum(y_d == majo)


posy = y_d == mino
negy = y_d != mino
P = latent_space_im[np.where(posy == True)[0],:]
N = latent_space_im[np.where(negy == True)[0],:]

#print(len(P),len(N))

poscnt = P.shape[0]
NumToGen = nTarget - poscnt
Me  = np.mean((P),axis = 0)
PCov = np.cov(P.T)
#print(NumToGen)
[D,V] = np.linalg.eig(PCov)
#d = [D[x,x] for x in range(D.shape[0])]
d = D
n = P.shape[1] #Feature dimension
idx = d.argsort()[::-1]   
d = d[idx]
V = V[:,idx]
#d = d[0:n+1]
#v = V[:,n::-1]

Ind = (d<= 5e-03)

if np.sum(Ind) != 0:
    M = (list(Ind).index(True)+1)
else:
    M = n
    

PN = np.concatenate((P,N),axis=0)
TCov = np.cov(PN.T)
dT    = np.dot(V.T,np.dot(TCov, V))
dT = [dT[x,x] for x in range(dT.shape[0])]

#Modify the Eigen spectrum according to a 1-Parameter Model
dMod  = np.zeros((n,1))
Alpha = d[0]* d[M-1]*(M-1) /(d[0] - d[M-1]) #d[0]* d[M-1]*(M-1) /(d[0] - d[M-1])
Beta  = ((M)*d[M-1] - d[0])/(d[0] - d[M-1])

for i in range(n):
    if i<M-1:

        dMod[i] = d[i]
    else:
        dMod[i] = Alpha/(i+1+Beta)
        if dMod[i] > dT[i]:
            dMod[i] = dT[i]

R = 1.0
d = dMod
        
########################################
   
Rn = M
Un = len(Me) - M
Ptemp = P

MuR = np.zeros((Rn,1)) #mlayer#
SigmaR = np.identity((Rn)) #v_mat #

MuU = np.zeros((Un,1))
SigmaU = np.identity((Un))

SampGen = np.zeros((int(NumToGen*R), len(Me)))
SampSel = np.zeros((int(NumToGen*R), len(Me)))
Prob    = np.zeros((int(NumToGen*R),1))

cnt = 0
DD = np.sqrt(d)
MuR = MuR.reshape(MuR.shape[0],)
MuU = MuU.reshape(MuU.shape[0],)

while cnt < int(R*NumToGen):
    
    aR =  np.random.multivariate_normal(MuR.T, SigmaR, 1)
    tp = multivariate_normal.pdf(aR, MuR, SigmaR) #aR.pdf(1)
    #print(tp)

    if Un > 0:
        aU = np.random.multivariate_normal(MuU, SigmaU, 1)
        a = np.multiply(np.concatenate((aR,aU),axis=1).T,DD)   #The vector in Eigen transformed domain;
    else:
        a = np.multiply(aR.T,DD)
        #print(a)

    x = np.dot(a.T,V.T)+ Me
    #print(x)
    #pdb.set_trace()
    PDist = np.sqrt(np.sum(np.square((x-P)),axis=1))
    NDist = np.sqrt(np.sum(np.square((x-N)),axis=1))

    [tmp,ind]  = [np.min(NDist),np.argmin(NDist)]

    if np.min(PDist) < tmp:
        PPDist = np.sqrt(np.sum(np.square((N[ind,:]-P)),axis=1))
        if tmp >= np.min(PPDist) and tmp <= np.max(PPDist):
            SampGen[cnt,:] = x
            Prob[cnt,0] = tp  
            cnt+=1
            Ptemp = np.concatenate((Ptemp,SampGen),axis =0)

for i in range (int(R*NumToGen)):
    [tmp,ind]  = [np.min(Prob),np.argmin(Prob)]
    Prob[ind] =  np.inf
    SampSel[i,:] = SampGen[ind,:]

Ynew = SampSel #np.concatenate((SampSel,P),axis = 0)
    
#return Ynew


# In[15]:


Datanew = np.concatenate((SampSel,P),axis = 0)
Total = np.concatenate((Datanew,N),axis = 0)


# In[16]:


label = np.zeros((Total.shape[0],))
label[0:Datanew.shape[0]] = 1
label[Datanew.shape[0]:Total.shape[0]] = 2


# #### Metric function

# In[17]:


def metric_score(y_test, y_pred):
    confMat=confusion_matrix(y_test, y_pred) 
    #print(confMat)
    TP = confMat[0,0]
    TN = confMat[1,1]
    FP = confMat[1,0]
    FN = confMat[0,1]
    #print(TP,FP,TN,FN)
    Eta = 1/2*((TP/np.sum(y_test==1))+(TN/np.sum(y_test==2)))
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    Gmean = np.sqrt(TPR*TNR)
    return(TPR,TNR,Gmean,Eta)


# In[18]:


X_train = (Total)
y_train = (label)
X_test = (LS_Ger_test)
y_test = (y_test)

mm_X = MinMaxScaler(feature_range=(0,1.6))
X_trainscaled= mm_X.fit_transform(X_train)#
X_testscaled= mm_X.transform(X_test) 

clf = MLPClassifier(solver='adam',hidden_layer_sizes=(10, 64, 9, ),activation="relu",
                    batch_size= 64,random_state=1,max_iter = 2000,learning_rate_init = 0.00012,
                   learning_rate= 'adaptive',power_t=0.6,warm_start = False,
                   beta_1 =0.95,beta_2 = 0.850)
# 10,32,8,3 5,10,8,3,
# (8,16,10,9,6,) -4 LS 10,16,12,6,3
# 10,25,10,8,6,(10,25,10,8,6,)
clf.fit(X_trainscaled, y_train)

y_pred=clf.predict(X_testscaled)

target_names = ['class 1', 'class 2']
print('TPR, TNR, GM , Eta:')
print(metric_score(y_test, y_pred))


# In[19]:


kf = KFold(n_splits=10,shuffle = True)
print("Cross Validation")
print('TPR, TNR, GM , Eta:')

for train_indices, test_indices in kf.split((X_trainscaled)):
    clf.fit(X_trainscaled[train_indices], y_train[train_indices])
    # print(clf.score(X_testscaled, y_test))

    y_pred=clf.predict(X_testscaled)
    print(metric_score(y_test, y_pred))
    


# In[20]:


print('Time taken :',time.time()-start_time)


# In[ ]:




