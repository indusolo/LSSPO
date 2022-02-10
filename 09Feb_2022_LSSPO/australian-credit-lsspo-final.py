#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Australian Credit Dataset
# Necessary imports
import pandas as pd
import numpy as np
from numpy import genfromtxt
#from pymfe.mfe import MFE
import csv
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from focal_loss import BinaryFocalLoss
from tensorflow.keras.models import load_model
from scipy.io import arff
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time
import random
from sklearn.model_selection import train_test_split 
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(3)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Layer
#from keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.losses import mse, binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy
from tensorflow.keras.layers import Activation, Dense, Flatten 
from tensorflow.keras import backend as K

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
# In[5]:
import scipy.io
mat = scipy.io.loadmat('AUS_NEW.mat')


# In[2]:


# In[6]:


x_train = mat['train_x'] 
ytrn = mat['train_y'] 
y_train =  ytrn[:,0]*2+ytrn[:,1]*1

x_test = mat['test_x'] 
ytst = mat['test_y'] 
y_test =  ytst[:,0]*2+ytst[:,1]*1



# ### T-SNE

# In[8]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
tsne_org = tsne.fit_transform(x_train)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["r","b"])
plt.figure(figsize=(8,8))
colors = ['b', 'r']

points = tsne_org[y_train ==1]
print(points.shape)
p2 = plt.scatter(points[:, 0], points[:, 1], marker=('o'), color=colors[0])
points = tsne_org[y_train == 2]
p1 = plt.scatter(points[:, 0], points[:, 1], marker=('^'), color=colors[1])
plt.legend((p2,p1),('Minority','Majority'),loc='upper right')

#plt.savefig('Aus_TSNE.png')
#plt.show()


mmscaler    = MinMaxScaler(feature_range=(-1,1))

scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)
x_train   = mmscaler.fit_transform(x_train)
x_test = mmscaler.transform(x_test)


# In[3]:


# ### VAE

# In[10]:


start_dimension = x_train.shape[1]
# network parameters
input_shape = (start_dimension, )
intermediate_dim = 8
batch_size = 24
latent_dim = 5
epochs = 1000


# In[11]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[12]:


# build encoder model
inputs = keras.Input(shape=input_shape, name='encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary 
# with the TensorFlow backend
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# In[13]:


# build decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(start_dimension, activation='tanh')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# In[14]:


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')


# In[15]:


X_imtrain, X_imval, y_imtrain, y_imval = train_test_split(x_train, x_train, 
                                                    test_size=0.2, random_state=1)

start_time = time.time()
# In[16]:


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
    


# In[4]:


# Train data latent space
def latent_data(vae, data):
    encoder, decoder = vae
    z_mean, _, _ = encoder.predict(data)
    return z_mean
LS_Aus_train = latent_data(models, x_train)
LS_Aus_test = latent_data(models, x_test)


# ### T-SNE

# In[19]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
tsne_org = tsne.fit_transform(LS_Aus_train)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["r","b"])
plt.figure(figsize=(8,8))
colors = ['r', 'b']

points = tsne_org[y_train ==1]
print(points.shape)
p2 = plt.scatter(points[:, 0], points[:, 1], marker=('o'), color=colors[0])
points = tsne_org[y_train == 2]
p1 = plt.scatter(points[:, 0], points[:, 1], marker=('^'), color=colors[1])
plt.legend((p2,p1,),('Minority','Majority',), loc='upper right')
#plt.savefig('Aus_TSNE.png')
#plt.show()


# In[20]:




# ### MSPO

# In[24]:

latent_space_im = LS_Aus_train
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
print(NumToGen)
[D,V] = np.linalg.eig(PCov)
#d = [D[x,x] for x in range(D.shape[0])]
d = D
print(D)
#d = d.astype(np.float32)
n = P.shape[1] #Feature dimension
idx = d.argsort()[::-1]   
d = d[idx]
V = V[:,idx]
#d = d[0:n+1]
#v = V[:,n::-1]



Ind = (d<= 5e-09)

if np.sum(Ind) != 0:
    M = (list(Ind).index(True)+1)
else:
    M = n
    
print(Ind,M)

PN = np.concatenate((P,N),axis=0)
TCov = np.cov(PN.T)
dT    = np.dot(V.T,np.dot(TCov, V))
dT = [dT[x,x] for x in range(dT.shape[0])]


#Modify the Eigen spectrum according to a 1-Parameter Model
dMod  = np.zeros((n,1))
Alpha = d[0]* d[M-1]*(M-1) /(d[0] - d[M-1]) #d[0]* d[M-1]*(M-1) /(d[0] - d[M-1])
Beta  = ((M)*d[M-1] - d[0])/(d[0] - d[M-1])
print(Alpha,Beta)

for i in range(n):
    if i<M-1:

        dMod[i] = d[i]
    else:
        dMod[i] = Alpha/(i+1+Beta)
        if dMod[i] > dT[i]:
            dMod[i] = dT[i]

R = 0.7
d = dMod
print(d)
    
########################################

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
SampSel = np.zeros((int(NumToGen*R), len(Me)))
Prob    = np.zeros((int(NumToGen*R),1))

cnt = 0
DD = np.sqrt(d)
MuR = MuR.reshape(MuR.shape[0],)
MuU = MuU.reshape(MuU.shape[0],)
print(R*NumToGen)

while cnt < int(R*NumToGen):
    
    if(cnt%2000 == 0):
        print(cnt)

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

for i in range (int(R*NumToGen)):
    [tmp,ind]  = [np.min(Prob),np.argmin(Prob)]
    Prob[ind] =  np.inf
    SampSel[i,:] = SampGen[ind,:]

Ynew = SampSel #np.concatenate((SampSel,P),axis = 0)
#Total = np.concatenate((Ynew,N),axis = 0)

#return Ynew


# In[25]:


Datanew = np.concatenate((SampSel,P),axis = 0)
Total = np.concatenate((Datanew,N),axis = 0)
label = np.zeros((Total.shape[0],))
label[0:Datanew.shape[0]] = 1
label[Datanew.shape[0]:Total.shape[0]] = 2


# In[26]:


# In[5]:


# ### T-SNE on MSPO

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
tsne_org = tsne.fit_transform(Total)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["r","b"])
plt.figure(figsize=(8,8))
colors = ['r', 'b']

points = tsne_org[label ==1]
print(points.shape)
p2 = plt.scatter(points[:, 0], points[:, 1], marker=('o'), color=colors[0])
points = tsne_org[label == 2]
p1 = plt.scatter(points[:, 0], points[:, 1], marker=('^'), color=colors[1])

plt.legend((p2,p1,),('Minority','Majority',), loc='upper right')
#plt.savefig('Aus_TSNE.png')
#plt.show()


# In[29]:


# #### MLP Classifier

# In[6]:


X_train = (Total)
y_train = (label)
X_test = (LS_Aus_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3.06,3.06))
X_trainscaled= mm_X.fit_transform(X_train) #mm_X.fit_transform(X_train)  #sc_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test) #mm_X.transform(X_test)  #sc_X.transform(X_test)
  
clf = MLPClassifier(solver='adam',batch_size= 200,hidden_layer_sizes=
                    (6,22,8,3,2),
                    activation="relu",random_state=1,max_iter = 5000,
                    learning_rate= 'constant',warm_start = True,
                    learning_rate_init =0.0014,
                   alpha =0.0000999,beta_1=0.9,beta_2=0.999,)

clf.fit(X_trainscaled, y_train)

y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["1",'2'])#,'5','6','7','8','9'])
fig.figure_.suptitle("Confusion Matrix for MSPO Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class 1', 'class 2']#,'class 5','class 6','class 7','class 8','class 9']
print(classification_report(y_pred,y_test, target_names=target_names))

# In[30]:


# #### Evaluation Metric

# In[7]:


def metricFn(y_test, y_pred):
    
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
    return (TPR,TNR,Gmean,Eta)
# print('TPR , TNR , GM ,Eta')
# print(metricFn(y_test, y_pred))


# #### Cross validation

# In[8]:


kf = KFold(n_splits=10,shuffle = True)
print("Cross Validation")
print('TPR , TNR , GM ,Eta')

for train_indices, test_indices in kf.split(X_trainscaled):
    clf.fit(X_trainscaled[train_indices], y_train[train_indices])
    # print(clf.score(X_testscaled, y_test))

    y_pred=clf.predict(X_testscaled)
    print(metricFn(y_test, y_pred))
    


# In[9]:


print("Time taken:",time.time()-start_time)


# In[ ]:




