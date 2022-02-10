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


# In[16]:


X_trainmlp, X_valmlp, y_trmlp, y_valmlp = train_test_split(TrainDatascaled, y_trn, 
                                                    test_size=0.3, random_state=1, stratify = y_trn)


# ##### SMOTE

# In[17]:


from imblearn.over_sampling import SMOTE


# In[18]:


oversample = SMOTE()
X_sm, y_sm = oversample.fit_resample(X_trainmlp, y_trmlp)


# In[19]:


X_sm.shape


# In[20]:


X_train = (X_sm)
y_train = (y_sm)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size =200,solver='sgd',hidden_layer_sizes=(512,256,64,16),
                    activation="tanh",learning_rate_init = 0.00352,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'adaptive',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for SMOTE Latent space")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[21]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[22]:


score_sm = silhouette_score(X_sm, y_sm, metric='l2')
print("Silhouette score for SMOTE oversampled data:")
print(score_sm)


# ##### No oversampling

# In[23]:


X_train = (X_trainmlp)
y_train = (y_trmlp)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size = 64,solver='sgd',hidden_layer_sizes=(512,256,64,16,),
                    activation="tanh",learning_rate_init = 0.00015,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'adaptive',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for original data")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[24]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[25]:


score_sm = silhouette_score(TrainDatascaled, y_trn, metric='l2')
print("Silhouette score for original data:")
print(score_sm)


# ##### ADASYN oversampling

# In[26]:


from imblearn.over_sampling import ADASYN


# In[27]:


oversample = ADASYN()
X_adsn, y_adsn = oversample.fit_resample(X_trainmlp, y_trmlp)


# In[28]:


X_train = (X_adsn)
y_train = (y_adsn)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size =200,solver='sgd',hidden_layer_sizes=(512,256,64,16,),
                    activation="tanh",learning_rate_init = 0.00292,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'adaptive',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for ADASYN oversampled data")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[29]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[30]:


score_adsn = silhouette_score(X_adsn, y_adsn, metric='l2')
print("Silhouette score for adasyn oversampled data:")
print(score_adsn)


# ##### Random oversampling

# In[31]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_rand, y_rand = ros.fit_resample(X_trainmlp, y_trmlp)


# In[32]:


X_train = (X_rand)
y_train = (y_rand)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size =24,solver='adam',hidden_layer_sizes=(512,256,64,32,),
                    activation="tanh",learning_rate_init = 0.00192,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'constant',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for RANDOM oversampled data")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[33]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[34]:


score_rand = silhouette_score(X_rand, y_rand, metric='l2')
print("Silhouette score for RANDOM oversampled data:")
print(score_rand)


# ##### BORDERLINE SMOTE

# In[35]:


from imblearn.over_sampling import BorderlineSMOTE
X_bos, y_bos = BorderlineSMOTE().fit_resample(X_trainmlp, y_trmlp)


# In[36]:


X_train = (X_bos)
y_train = (y_bos)
X_test = (X_valmlp)
y_test = (y_valmlp)

#X_train, X_test, y_train, y_test = train_test_split(X,y_im,random_state=1, test_size=0.1)
sc_X = StandardScaler()
mm_X = MinMaxScaler(feature_range=(-2.85,2.85))

X_trainscaled= mm_X.fit_transform(X_train)
X_testscaled= mm_X.transform(X_test)

clf = MLPClassifier(batch_size = 200,solver='sgd',hidden_layer_sizes=(512,256,128,32,),
                    activation="tanh",learning_rate_init = 0.00192,random_state=1,shuffle = True,
                    max_iter = 5000,learning_rate = 'constant',beta_1=0.9,beta_2=0.999,
                   alpha=0.009,warm_start = False)
#learning_rate_init = 0.0000595
clf.fit(X_trainscaled, y_train)
y_pred=clf.predict(X_testscaled)
print(clf.score(X_testscaled, y_test))

fig=plot_confusion_matrix(clf, X_testscaled, y_test,display_labels=["-1",'1'])
fig.figure_.suptitle("Confusion Matrix for BORDERLINE SMOTE oversampled data")
target_names = ['class -1', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[37]:


# Evaluation metrics
acsa, gm = metrics_aa_gm(y_pred, y_test) 
print (acsa, gm,)


# In[38]:


score_BOS = silhouette_score(X_bos, y_bos, metric='l2')
print("Silhouette score for Borderline SMOTE oversampled data:")
print(score_BOS)


# In[ ]:




