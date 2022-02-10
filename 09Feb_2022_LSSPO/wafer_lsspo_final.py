#!/usr/bin/env python
# coding: utf-8

# ### Wafer dataset

# In[ ]:


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
from tensorflow.keras.models import load_model
from scipy.io import arff
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import random
random.seed(1)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import scipy
from scipy.stats import multivariate_normal


# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Layer
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.layers import Activation, Dense, Flatten 
from tensorflow.keras import backend as K
import scipy
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score


# In[ ]:



Traindata, Trainmeta = arff.loadarff('Wafer_TRAIN.arff')
Testdata, Testmeta = arff.loadarff('Wafer_TEST.arff')
df_Train = pd.DataFrame(Traindata)
df_Test =  pd.DataFrame(Testdata)
df_Train.info()
df_Train.head()


# In[ ]:


df_Test.shape


# In[ ]:


#Non-numerical datatype in the train and test set
print(df_Train.dtypes[df_Train.dtypes == object])
print(df_Test.dtypes[df_Test.dtypes == object])
#Convert the target attribute to integer
Train_target = pd.DataFrame()
Test_target = pd.DataFrame()
Train_target['target'] = df_Train['target'].astype(np.int8).copy()
Test_target['target'] = df_Test['target'].astype(np.int8).copy()
#np.sum(Train_target == -1)
ax = plt.figure()
ax = sns.countplot(x='target',data=Train_target).set_title('Traindata class distribution')
plt.savefig('Traindataimbalance.png')
ax1 = plt.figure()
ax1 = sns.countplot(x='target',data=Test_target).set_title('Testdata class distribution')
plt.savefig('Testdataimbalance.png')


# In[ ]:


#Training Datast
y_train = np.array(Train_target)
df_Train['target'] =y_train
y_train = np.array(df_Train['target']) #np.array(df_Test_z['target'])
#TrainData = np.array(df_Train.drop('target', axis=1)) #np.array(df_Test_z.drop('target', axis=1))
print("Majority class =",np.sum(y_train==1))
print("Minority class =",np.sum(y_train==-1))


# In[ ]:


#Test data preparation
y_test = np.array(Test_target)
df_Test['target'] =y_test
y_test = np.array(df_Test['target']) #np.array(df_Test_z['target'])
#TestData = np.array(df_Test.drop('target', axis=1)) #np.array(df_Test_z.drop('target', axis=1))
print("Majority class =",np.sum(y_test==1))
print("Minority class =",np.sum(y_test==-1))


# #### Training and Test dataset preparation for the experiment

# In[ ]:


Datanew = pd.concat([df_Train,df_Test])
y = Datanew.target
Datanew = np.array(Datanew.drop('target', axis=1))
Datanew.shape
Trn_imb_set = (50,3000)
imb_index = np.insert(np.cumsum(Trn_imb_set), 0, 0)
classes = np.array([-1,1])
TrainData = np.zeros((np.sum(Trn_imb_set),Datanew.shape[1]))
Tst_imb_set = (712,3402)
Tstimb_index = np.insert(np.cumsum(Tst_imb_set), 0, 0)
TestData = np.zeros((np.sum(Tst_imb_set),Datanew.shape[1]))


# In[ ]:


for i in range(classes.shape[0]):
    yind = np.where(y == classes[i])
    sel = np.random.choice(yind[0], Trn_imb_set[i], replace=False)
    TrainData[imb_index[i]:imb_index[i+1],:] = Datanew[sel]
    nsel = np.setdiff1d(yind,sel)
    TestData[Tstimb_index[i]:Tstimb_index[i+1],:] = Datanew[nsel]#np.delete(Datanew,sel,axis =0)
        
y_train = np.hstack([np.ones((50,))*-1,np.ones((3000,))])
y_train = y_train.astype('int8')
y_test = np.hstack([np.ones((712,))*-1,np.ones((3402,))])
y_test = y_test.astype('int8')


# In[ ]:


TestData.shape


# In[ ]:


#Normalize the data between +1 and -1
mmscaler    = MinMaxScaler(feature_range=(-1,1))
TrainData   = mmscaler.fit_transform(TrainData)


# In[ ]:


TestData   = mmscaler.transform(TestData)


# In[ ]:


y_trn = y_train


# ### VAE

# #### VAE code is adopted and modified from the following reference.
# #### https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py 

# In[ ]:


start_dimension = 152
input_shape = (152, )
intermediate_dim = 100
batch_size = 200
latent_dim = 4 
epochs = 700


# In[ ]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ### Encoder

# In[ ]:


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

# In[ ]:


# Decoder model
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(start_dimension, activation='tanh')(x)

# instantiate decoder model
decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# In[ ]:


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')


# In[ ]:


from sklearn.model_selection import train_test_split 
X_trainvae, X_valvae, y_trvae, y_valvae = train_test_split(TrainData, TrainData, 
                                                    test_size=0.15, random_state=11)


# In[ ]:


start_time = time.time()


# In[ ]:


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


# In[ ]:


def latent_data(vae, data):
    encoder, decoder = vae
    z_mean, _, _ = encoder.predict(data)
    return z_mean
LS_train = latent_data(models, TrainData)
LS_test = latent_data(models, TestData)


# ### MSPO

# In[ ]:


#def SPOfn(latent_space_im,mino,majo,y_d):
   
latent_space_im = LS_train
mino = -1
majo = 1
y_d = y_trn

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
SampSel = np.zeros((int(NumToGen), len(Me)))
Prob    = np.zeros((int(NumToGen*R),1))

cnt = 0
DD = np.sqrt(d)
MuR = MuR.reshape(MuR.shape[0],)
MuU = MuU.reshape(MuU.shape[0],)

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

for i in range (int(R*NumToGen)):
    [tmp,ind]  = [np.min(Prob),np.argmin(Prob)]
    Prob[ind] =  np.inf
    SampSel[i,:] = SampGen[ind,:]

Ynew = SampSel #np.concatenate((SampSel,P),axis = 0)
#Total = np.concatenate((Ynew,N),axis = 0)

#return Ynew


# In[ ]:


Datanew = np.concatenate((SampSel,P),axis = 0)
Total = np.concatenate((Datanew,N),axis = 0)
label = np.zeros((Total.shape[0],))
label[0:Datanew.shape[0]] = -1
label[Datanew.shape[0]:Total.shape[0]] = 1


# ### MLP on SPO latent space

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:



def metrics_aa_gm(ypred, ytrue):
    cm = confusion_matrix(ytrue, ypred)
    sum_classes = np.sum(cm, axis=1)
    true_pred = np.diagonal(cm)
    tp_rate = true_pred/sum_classes
    ACSA = np.mean(tp_rate)
    GM = np.prod(tp_rate)**(1/cm.shape[0])
    return ACSA, GM


# In[ ]:


X_train = Total
y_trnmlp = label
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =16,hidden_layer_sizes=(8,16,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for SPO Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


#print('F1score:',f1_score(y_test, y_pred, average='weighted'))
#print('The geometric mean is {}'.format(geometric_mean_score(y_test,y_pred)))
acsa,gm = metrics_aa_gm(y_pred, y_test)
print("Metrics for VAE-LSSPO technique")
print('ACSA =',acsa,'GM=',gm)


# In[ ]:


# kf = KFold(n_splits=10,shuffle = True)
# print("Cross Validation")

# for train_indices, test_indices in kf.split((X_trainscaled)):
#     clf.fit(X_trainscaled[train_indices], y_trnmlp[train_indices])
#     # print(clf.score(X_testscaled, y_test))

#     y_pred=clf.predict(X_testscaled)
#     print('F1score:',f1_score(y_test, y_pred, average='weighted'),"GM: {}".format(geometric_mean_score(y_test,y_pred)))
#     #print('The geometric mean is {}'.format(geometric_mean_score(y_test,y_pred)))
    


# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[ ]:


score_lsspo = silhouette_score(Total, label, metric='l2')
print("Silhouette score for VAE-LSSPO latent data =",score_lsspo)


# In[ ]:


print('Time taken:',time.time()-start_time)


# #### No oversampling

# In[ ]:





# In[ ]:


score_noOS = silhouette_score(LS_train, y_trn, metric='l2')
print("Silhouette score for VAE latent data =",score_noOS)


# ##### MLP on latent data without oversampling

# In[ ]:


X_train = LS_train
y_trnmlp = y_trn
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =16,hidden_layer_sizes=(8,16,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for SPO Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


acsa,gm = metrics_aa_gm(y_pred, y_test)
print("Metrics for no oversampling technique")
print('ACSA =',acsa,'GM=',gm)


# #### SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


oversample = SMOTE()
X_sm, y_sm = oversample.fit_resample(LS_train, y_trn)


# In[ ]:


X_train = X_sm  
y_trnmlp = y_sm
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =64,hidden_layer_sizes=(8,32,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for SMOTE Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


acsa,gm = metrics_aa_gm(y_pred, y_test)
print("Metrics for SMOTE technique")
print('ACSA =',acsa,'GM=',gm)


# In[ ]:


score_sm = silhouette_score(X_sm, y_sm, metric='l2')
print("Silhouette score for SMOTE oversampled latent data =",score_sm)


# #### ADASYN

# In[ ]:


from imblearn.over_sampling import ADASYN 


# In[ ]:


oversample = ADASYN()
X_adsn, y_adsn = oversample.fit_resample(LS_train, y_trn)


# In[ ]:


score_adsn = silhouette_score(X_adsn, y_adsn, metric='l2')
print("Silhouette score for VAE+ADASYN oversampled data =",score_adsn)


# In[ ]:


X_train = X_adsn   
y_trnmlp = y_adsn
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =16,hidden_layer_sizes=(8,32,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for ADASYN Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print('No oversampling - VAE+ADASYN+MLP')
print( 'ACSA =',acsa,'GM=', gm,)


# #### RANDOM SAMPLING

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_rand, y_rand = ros.fit_resample(LS_train, y_trn)


# In[ ]:


score_rand = silhouette_score(X_rand, y_rand, metric='l2')
print("Silhouette score for VAE+Random oversampled data =",score_rand)


# In[ ]:


X_train = X_rand   
y_trnmlp = y_rand
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =16,hidden_layer_sizes=(8,32,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for RANDOM SAMPLED Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print('No oversampling - VAE+RANDOM+MLP')
print( 'ACSA =',acsa,'GM=', gm,)


# #### BORDERLINE SMOTE

# In[ ]:


from imblearn.over_sampling import BorderlineSMOTE
X_bos, y_bos = BorderlineSMOTE().fit_resample(LS_train, y_trn)


# In[ ]:


score_bos = silhouette_score(X_bos, y_bos, metric='l2')
print("Silhouette score for VAE+Borderline SMOTE oversampled data =",score_bos)


# In[ ]:


X_train = X_bos    
y_trnmlp = y_bos
X_test = (LS_test)
y_test = (y_test)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-3,3))
X_trainscaled= X_train #mm_X.fit_transform(X_train)
X_testscaled= X_test #mm_X.transform(X_test)

clf = MLPClassifier(batch_size =16,hidden_layer_sizes=(8,32,10,7,),activation="tanh",
                    solver = 'sgd',random_state=1,max_iter = 5000,learning_rate_init = 0.0058,
                   learning_rate = 'adaptive')
#clf = MLPClassifier(hidden_layer_sizes=(4,12,6,5),activation="tanh",learning_rate_init = 0.0008,random_state=1,max_iter = 5000)
clf.fit(X_trainscaled, y_trnmlp)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for BORDERLINE SMOTE SAMPLED Latent space")
#plt.savefig('ConfusionMatrixoforiginallatentspace.png')
#plt.show()
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print('No oversampling - VAE+BoS+MLP')
print( 'ACSA =',acsa,'GM=', gm,)


# In[ ]:




