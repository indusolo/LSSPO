#!/usr/bin/env python
# coding: utf-8

# #### Kaggle GMSC 

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
from tensorflow.keras.models import load_model
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time
import random
from sklearn.model_selection import train_test_split 
from numpy.random import seed
seed(2)
import tensorflow as tf
tf.random.set_seed(7)
import random
random.seed(1)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Layer
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.losses import mse, binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy
from tensorflow.keras.layers import Activation, Dense, Flatten 
from tensorflow.keras import backend as K
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import scipy
from scipy.stats import multivariate_normal


# In[2]:


Data_Train = pd.read_csv('cs-training.csv')
#Data_Test = pd.read_csv('../input/give-me-some-credit-dataset/cs-test.csv')


# In[3]:


Data_Train.info()


# In[4]:


Train_data = Data_Train.copy()


# In[5]:


trainID = Train_data['Unnamed: 0']
Train_data.drop(['Unnamed: 0'],axis=1, inplace=True)


# In[6]:


y = Train_data['SeriousDlqin2yrs']
Train_data.drop(['SeriousDlqin2yrs'],axis=1,inplace=True)


# In[7]:


Train_data.info()


# In[8]:


print(Train_data.isnull().sum())


# #### Fill the missing values with mean and mode

# In[9]:


Train_data['MonthlyIncome'].fillna(Train_data['MonthlyIncome'].median(),inplace=True)

Train_data['NumberOfDependents'].fillna(Train_data['NumberOfDependents'].mode()[0], inplace=True)


# In[10]:


Train_data.info()


# In[11]:


feat_outliers = ['DebtRatio','RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse',
                'MonthlyIncome','NumberRealEstateLoansOrLines','NumberOfDependents']

for i in range(len(feat_outliers)):
    col = feat_outliers[i]
    print(feat_outliers[i])
    
    Q3 = Train_data[col].quantile(0.75) #np.quantile(Train_data[col], 0.75)
    Q1 = Train_data[col].quantile(0.25) #np.quantile(Train_data[col], 0.25)
    IQR = Q3 - Q1

    lower_range = Q1 - 2.0 * IQR
    upper_range = Q3 + 2.0 * IQR
    print(lower_range,upper_range)
    outlier_free_list = [x for x in Train_data[col] if (
            (x > lower_range) & (x < upper_range))]
    Train_data.loc[~Train_data[col].isin(outlier_free_list)] = Train_data[col].quantile(0.5)

#Train_data["DebtRatio"],IQR


# In[12]:


#Normalize the data between +1 and -1

mmscaler    = MinMaxScaler(feature_range=(-1,1))
Train_data   = mmscaler.fit_transform(Train_data)


# In[13]:


Train_data.shape


# ### VAE

# #### VAE code is adopted and modified from the following reference.
# #### https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py 

# In[14]:


start_dimension = Train_data.shape[1]
# network parameters
input_shape = (start_dimension, )
intermediate_dim = 6
batch_size = 250
latent_dim = 3
epochs = 300


# In[15]:


def sampling(args):
    z_mean, z_log_var = args
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[16]:


# Encoder model
inputs = keras.Input(shape=input_shape, name='encoder_input')
x = layers.Dense(intermediate_dim, activation='tanh')(inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# In[17]:


# Decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='tanh')(latent_inputs)
outputs = layers.Dense(start_dimension, activation='tanh')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# In[18]:


# VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')


# #### Train-Test split for classifier 

# In[19]:


X_data, X_test, y_data, y_test = train_test_split(Train_data, y, 
                                                    test_size=0.3, random_state=22,stratify=y)


# #### Train-Validation split for VAE

# In[20]:


X_imtrain, X_imval, y_imtrain, y_imval = train_test_split(X_data, X_data, 
                                                    test_size=0.2, random_state=1)


# In[21]:


start_time = time.time()


# In[22]:


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
    


# In[23]:


def latent_space_data(vae, data):
    encoder, decoder = vae
    z_mean, _, _ = encoder.predict(data)
    return z_mean
LS_GMSC_train = latent_space_data(models, X_data)
LS_GMSC_test = latent_space_data(models, X_test)


# #### Metric function

# In[24]:


def metric_score(y_test, y_pred):
    confMat=confusion_matrix(y_test, y_pred) 
    confMat
    TP = confMat[1,1]
    TN = confMat[0,0]
    FP = confMat[0,1]
    FN = confMat[1,0]
    TP,FP,TN,FN
    Eta = 1/2*((TP/np.sum(y_test==1))+(TN/np.sum(y_test==0)))
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    Gmean = np.sqrt(TPR*TNR)
    return(TPR,TNR,Gmean,Eta)
    


# ### MSPO

# In[25]:


#def SPOfn(latent_space_im,mino,majo,y_d):
   
#Input - P and N
# Output = Me(mean), 
#V - Eigen matrix, 
#D - Modified eigen spectrurm value, 
#M -  the portion of reliable
latent_space_im = LS_GMSC_train
mino = 1
majo = 0
y_d = y_data

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
#d = d.astype(np.float32)
n = P.shape[1] #Feature dimension
idx = d.argsort()[::-1]   
d = d[idx]
V = V[:,idx]
  

Ind = (d<= 5e-09)

if np.sum(Ind) != 0:
    M = (list(Ind).index(True)+1)
else:
    M = n
    
#print(Ind,M)

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

R = 0.1
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
SampSel = np.zeros((int(NumToGen), len(Me)))
Prob    = np.zeros((int(NumToGen*R),1))

cnt = 0
DD = np.sqrt(d)
MuR = MuR.reshape(MuR.shape[0],)
MuU = MuU.reshape(MuU.shape[0],)
#print(R*NumToGen)

while cnt < int(R*NumToGen):
    
   
    aR =  np.random.multivariate_normal(MuR.T, SigmaR, 1)
    #print(aR)
    #scipy.stats.multivariate_normal(MuR.T, SigmaR, 1)
    tp = multivariate_normal.pdf(aR, MuR, SigmaR) #aR.pdf(1)
    #print(tp)

    if Un > 0:
        aU = np.random.multivariate_normal(MuU, SigmaU, 1)
        #scipy.stats.multivariate_normal(MuU, SigmaU, 1)
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

for i in range(NumToGen):
    [tmp,ind]  = [np.min(Prob),np.argmin(Prob)]
    Prob[ind] =  np.inf
    SampSel[i,:] = SampGen[ind,:]

Ynew = SampSel #np.concatenate((SampSel,P),axis = 0)
#Total = np.concatenate((Ynew,N),axis = 0)

#return Ynew


# In[26]:


print("No of samples generated:",cnt)


# In[27]:


Datanew = np.concatenate((SampSel,P),axis = 0)
Total = np.concatenate((Datanew,N),axis = 0)


# In[28]:


label = np.zeros((Total.shape[0],))
label[0:Datanew.shape[0]] = 1
label[Datanew.shape[0]:Total.shape[0]] = 0


# In[29]:


Total.shape


# In[30]:


X_train = (Total)
y_train = (label)
X_test = (LS_GMSC_test)
y_test = (y_test)


mm_X = MinMaxScaler(feature_range=(-1.0,1.0))
X_trainscaled= mm_X.fit_transform(X_train) #mm_X.fit_transform(X_train)  #sc_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test) #mm_X.transform(X_test)  #sc_X.transform(X_test)

clf = MLPClassifier(solver='adam',hidden_layer_sizes=(8,47,5),activation="relu",
                    random_state=1,max_iter = 5000,batch_size=200,learning_rate_init=0.31,
                    learning_rate ='constant',warm_start = True)
# 10,32,8,3 5,10,8,3,
clf.fit(X_trainscaled, y_train)

y_pred=clf.predict(X_testscaled)
#print(clf.score(X_testscaled, y_test))

print(metric_score(y_test, y_pred))


# In[32]:


print('Time taken :',time.time()-start_time)


# In[ ]:





# In[ ]:





# In[ ]:




