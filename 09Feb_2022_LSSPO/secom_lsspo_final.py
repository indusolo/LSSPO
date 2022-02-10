#!/usr/bin/env python
# coding: utf-8

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
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
import csv
from sklearn.impute import SimpleImputer as Imputer
import tensorflow as tf
tf.random.set_seed(2)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Layer
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.layers import Activation, Dense, Flatten 
from tensorflow.keras import backend as K
from numpy.random import seed
seed(1)
import random
random.seed(3)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score


# In[2]:


#Read the Secom dataset
reader = csv.reader(open("SECOM/SECOM.TXT"), delimiter=" ")
Data = pd.DataFrame(reader)
labels = csv.reader(open("SECOM/SECOM_labels.TXT"), delimiter=" ")
secom_labels = pd.DataFrame(labels)
y = secom_labels[0].astype('int8')
print("Data size",Data.shape)
print('Labels size',y.shape)
print("Minority class count =", np.sum(y==1))
print(Data.head())
Data.info()
#Data = Data.astype('float')
Target = pd.DataFrame()
Target['tar'] = y


# In[3]:


Data.shape


# In[4]:


#Handling missing data
missdata_Columns = []
for i in Data.columns:
    if np.sum(Data[i][:] == 'NaN') >0:
        missdata_Columns.append([i,np.sum(Data[i][:] == 'NaN')])
        #print( i, ':',np.sum(Data[i][:] == 'NaN') )
        
missdata_df = pd.DataFrame(missdata_Columns)
plt.figure(figsize=(15,5))
plt.bar(missdata_df[0],missdata_df[1],color='red')
plt.legend('Missing')
#plt.savefig('SecomDatamissing.png')
#plt.show()


# In[5]:


#Replace NaN in missing values with -1.0
Data = Data.replace(to_replace = 'NaN', value = '-1.0')
Datafloat = Data.copy()
#Convert to float datatype
Datafloat = Datafloat.astype('float64')


# In[6]:


# Impute Missing Values
def impute(df, cols,strat):
    imputer = Imputer(missing_values = -1.0,strategy=strat)
    df_impute = df[cols]
    df[cols] = imputer.fit_transform(df_impute.values.reshape(-1,1))
    return df


# In[7]:


#Imputing strategy mean, nearest and linear interpolation
for i in Datafloat.columns:
    if np.abs(np.sum(Datafloat[i][:] == -1.0)) >= 700:
        
        Datafloat.drop([i],inplace = True,axis =1)
    elif np.abs(np.sum(Datafloat[i][:] == -1.0)) <=30:
    #    print(i)
        600 #and np.abs(np.sum(Data[i][:] == -1.0)) <=780:
        Datafloat = impute(Datafloat,i,'mean')
    elif np.abs(np.sum(Datafloat[i][:] == -1.0))  > 30 and np.abs(np.sum(Datafloat[i][:] == -1))  <=200:
        Datafloat[i].replace(to_replace = -1.0, value = None)
        Datafloat[i].interpolate(method ='nearest', limit_direction ='forward')
    elif np.abs(np.sum(Datafloat[i][:] == -1.0))  > 200 and np.abs(np.sum(Datafloat[i][:] == -1))  <=300:
        Datafloat[i].replace(to_replace = -1.0, value = None)
        Datafloat[i].interpolate(method ='linear', limit_direction ='both')


# In[8]:


#Get the unique value columns
def uniq_cols_fn(data):
    uniq_col_list = []
    for column in data.columns:
        if data[column].nunique() == 1:
            uniq_col_list.append(column)
    return uniq_col_list


# In[9]:


len(uniq_cols_fn(Datafloat))


# In[10]:


Datafloat = Datafloat.drop(axis=1, columns=uniq_cols_fn(Datafloat))
Datafloat.shape


# In[11]:


np.sum(y==-1)


# In[12]:


#Normalize the data between +1 and -1
mmscaler    = MinMaxScaler(feature_range=(-1,1))
Datascaled   = mmscaler.fit_transform(Datafloat)


# In[13]:


TrainDatascaled = Datascaled
y_trn = y


# In[14]:


TrainDatascaled.shape


# In[15]:


def metrics_aa_gm(ypred, ytrue):
    cm = confusion_matrix(ytrue, ypred)
    sum_classes = np.sum(cm, axis=1)
    true_pred = np.diagonal(cm)
    tp_rate = true_pred/sum_classes
    ACSA = np.mean(tp_rate)
    GM = np.sqrt(np.prod(tp_rate))
    return ACSA, GM


# ## VAE

# #### VAE code is adopted and modified from the following reference.
# #### https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py

# In[16]:


start_dimension = Datafloat.shape[1]
input_shape = (Datafloat.shape[1], )
intermediate_dim = 220 
batch_size = 24
latent_dim = 7
epochs = 1000


# ### Sampling

# In[17]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ### Encoder

# In[18]:


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


# ### Decoder

# In[19]:


# Decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(start_dimension, activation='tanh')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# In[20]:


# VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')


# In[21]:



X_trainvae, X_valvae, y_trvae, y_valvae = train_test_split(TrainDatascaled, TrainDatascaled, 
                                                    test_size=0.15, random_state=1)


# In[22]:


start_time = time.time()


# In[23]:


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
    vae.compile(optimizer='adam')
    vae.summary()
    vae.fit(X_trainvae,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valvae, None))


# In[24]:


def latent_space_data(vae, data, labels):
    encoder, decoder = vae
    z_mean, _, _ = encoder.predict(data)
    return z_mean
latent_space_train = latent_space_data(models, TrainDatascaled, y_trn)


# ### Train test split for MLP

# In[25]:


X_mlp = latent_space_train
y_mlp = y_trn
X_trainmlp, X_valmlp, y_trmlp, y_valmlp = train_test_split(X_mlp,y_mlp, 
                                                    test_size=0.3, random_state=1, stratify = y_mlp)


# ### MSPO

# In[26]:


#def SPOfn(latent_space_im,mino,majo,y_d):
   
latent_space_im = X_trainmlp
mino = 1
majo = -1
y_d = y_trmlp

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
[D,V] = np.linalg.eig(PCov)
#d = [D[x,x] for x in range(D.shape[0])]
d = D
#d = d.astype(np.float32)
n = P.shape[1] #Feature dimension
idx = d.argsort()[::-1]   
d = d[idx]
V = V[:,idx]
#d = d[0:n+1]
#v = V[:,n::-1]



Ind = (d<= 5e-04)

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

R = 1
d = dMod
        
########################################

#OUTPUT:
#The oversampled dataset
import scipy
from scipy.stats import multivariate_normal
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
#MuU = MuU.reshape(MuU.shape[0],)

while cnt < int(R*NumToGen):
    
    
    aR =  np.random.multivariate_normal(MuR.T, SigmaR, 1)
    tp = multivariate_normal.pdf(aR, MuR, SigmaR) #aR.pdf(1)
    #print(tp)

    if Un > 0:
        aU = np.random.multivariate_normal(MuU.T, SigmaU, 1)
        #scipy.stats.multivariate_normal(MuU, SigmaU, 1)
        a = np.multiply(np.concatenate((aR,aU),axis=0),DD)   #The vector in Eigen transformed domain;
    else:
        a = np.multiply(aR.T,DD)
       
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


# In[27]:


print("Samples generated:",cnt)


# In[28]:


Datanew = np.concatenate((SampSel,P),axis = 0)
Total = np.concatenate((Datanew,N),axis = 0)
label = np.zeros((Total.shape[0],))
label[0:Datanew.shape[0]] = 1
label[Datanew.shape[0]:Total.shape[0]] = -1


# In[29]:


score_mspo = silhouette_score(Total, label, metric='l2')
print("Silhouette score for MSPO oversampled data:")
print(score_mspo)


# ### MLP on SPO Latent Space

# In[ ]:





# In[30]:


'''
shuffled_indices = np.random.permutation((Total.shape[0])) #return a permutation of the indices

#print(f"shuffled indices: {shuffled_indices}")

Total = Total[shuffled_indices]

label = label[shuffled_indices]
#label = label.astype('int8')
'''


# In[31]:


X_train = (Total)
y_train = (label)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size =250,solver='adam',hidden_layer_sizes=(9,16, 8,6,),
                    activation="tanh",learning_rate_init = 0.15,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'constant',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for Original Latent space")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# #### Evaluation metric

# In[32]:


def metrics_aa_gm(ypred, ytrue):
    cm = confusion_matrix(ytrue, ypred)
    sum_classes = np.sum(cm, axis=1)
    true_pred = np.diagonal(cm)
    tp_rate = true_pred/sum_classes
    ACSA = np.mean(tp_rate)
    GM = np.sqrt(np.prod(tp_rate))
    return ACSA, GM


# In[33]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[34]:


print('Time taken:',time.time()-start_time)


# In[35]:


score_mspo = silhouette_score(Total, label, metric='l2')
print("Silhouette score for MSPO oversampled data:")
print(score_mspo)

