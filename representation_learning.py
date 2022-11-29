#!/usr/bin/env python
# coding: utf-8

# In[1]:


from catenets.models import DRNet
#from importlib import reload  
from catenets.models.representation_nets import SNet1, SNet2

from catenets.experiments.simulation_utils import simulate_treatment_setup
#get_ipython().run_line_magic('load_ext', 'autotime')
import pandas as pd
import numpy as np
import pickle
import scipy
from scipy.sparse import csr_matrix
import sklearn
from matplotlib import pyplot as plt


# In[2]:


data_path = "../antiasthma-v2/"
(Y_tr, T_tr, X_tr, W_tr)=pickle.load(open(data_path+'YTXW_train.pkl','rb'))
(Y_va, T_va, X_va, W_va)=pickle.load(open(data_path+'YTXW_val.pkl','rb'))
(Y_te, T_te, X_te, W_te)=pickle.load(open(data_path+'YTXW_test.pkl','rb'))


tot_t = Y_tr.shape[0]
totexs = Y_tr.shape[0]+Y_va.shape[0]+Y_te.shape[0]
tot_tv = Y_tr.shape[0]+Y_va.shape[0]
Y_frames = [Y_tr, Y_va, Y_te]
X_frames = [X_tr, X_va, X_te]
T_frames = [T_tr, T_va, T_te]
W_frames = [W_tr, W_va, W_te]
Y = pd.concat(Y_frames)
Y = np.array(Y)
X = pd.concat(X_frames)
T = pd.concat(T_frames)
W = pd.concat(W_frames)
patids = W.patid.unique()
T = T["antiasthma"]
T.shape,Y.shape,X.shape,W.shape
n_pats = len(W.patid.unique())
n_phecodes = len(W.phecode3.unique())
W_mat = np.zeros((n_pats,n_phecodes))

for row in W.iterrows():
    pid = np.where(patids==int(row[1]["patid"]))
    pcd = int(row[1]["phecode3"])
    W_mat[pid,pcd] = row[1]["log_count"]
    
    
le_dx=pickle.load(open(data_path+'le_dx.pkl','rb'))
a = np.array(T)
Treatment = np.zeros((len(a),6))
ctr = 0
for item in a:
    Treatment[ctr][item] = 1
    ctr = ctr+1
Xs = np.concatenate((W_mat,X),axis=1)
#X = Xs/np.max(Xs,axis=0)

P = np.zeros(totexs)

for i in range(totexs):
    P[i] = 0.5


# estimate CATE using SNet1 =CFRNet

ermse = 0.0
cate_pred_all = []
ctr = 0
for k in range(5,6):
    print("\n For treatment: ",k)
    
    t = SNet1(val_split_prop=0.25,n_iter=1500,batch_size=100)
    w = np.array(Treatment[0:tot_tv,k],dtype=int)
    y = np.array(Y[0:tot_tv],dtype=int)
    X_train = Xs[0:tot_tv]
    #print(X_train)
    X_test = Xs[tot_tv :totexs]
    #cate = cate[0:tot_tv]
    p = P[0:tot_tv]
    t.fit(X_train, y, w) # w is t here 
    cate_pred_train = t.predict(X_test) # without potential outcomes
    t_test = SNet1(val_split_prop=0.1,n_iter=1500,batch_size=100)
    print("Now fitting on test data for ERMSE")
    t_test.fit(X_test, Y[tot_tv :totexs], np.array(Treatment[tot_tv :totexs,k],dtype=int)) # w is t here 
    cate_pred_test = t_test.predict(X_test) 
    cate_pred_all.append(cate_pred_train)
    ermse = ermse+(np.square(cate_pred_train - cate_pred_test).mean())
    ctr = ctr+1
    

    
ermse = ermse/ctr

print("The estimated root mean squared error for the CFRNet is: ",ermse)


from sklearn import preprocessing

def prepare(y, tr, w, x, rx2id, target):
    patid_temp = list(w['patid'].unique())
    temp_le = preprocessing.LabelEncoder()
    temp_le.fit(list(patid_temp))
    w['row_idx'] = temp_le.transform(w['patid'])
    
    w_sparse = csr_matrix((w['log_count'], (w['row_idx'], w['phecode3'])))
    w = w_sparse.toarray()
    
    x_temp = np.concatenate((w, x.values), axis=1)
    
    treatment_train = [0] * len(tr)
    temp_index = tr.index
    idx = 0

    def get_classes(value):
        return [k for k, v in rx2id.items() if v == value]

    for i in temp_index:
        classes = tr.loc[i, 'antiasthma']
        if (classes != target):
            treatment_train[idx] = 'control'
        else:
            treatment_train[idx] = 'treatment'
        idx += 1
        
    treatment = pd.DataFrame(treatment_train)
    treatment.index = temp_index
    treatment.columns = ['treatment']
    
    y = pd.DataFrame(y)
    feature_df = pd.DataFrame(x_temp)
    feature_df.index = y.index
    
    df = pd.concat([y, treatment, tr, feature_df], axis=1)
    df.index = np.arange(0, len(df))
    return df

le_dx=pickle.load(open(data_path+'le_dx.pkl','rb'))
le_patid=pickle.load(open(data_path+'le_patid.pkl','rb'))
selected_patient_feature=['age_onset','obs_win','female']+['race__'+c for c in ['A','B','H','U','W']]
rx2id = pickle.load(open(data_path+'drug_dict.pkl', 'rb'))

target = 5
df_val0 = prepare(Y_va, T_va, W_va, X_va, rx2id, target)
df_test0 = prepare(Y_te, T_te, W_te, X_te, rx2id, target)
df_train0 = prepare(Y_tr, T_tr, W_tr, X_tr, rx2id, target)

x_train0 = df_train0.iloc[:, 5:]
x_test0 = df_test0.iloc[:, 5:]
x_val0 = df_val0.iloc[:, 5:]

y_val = df_val0['adrd']

from xgboost import XGBClassifier, XGBRegressor
xgb = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin1 = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin0 = XGBClassifier(max_depth=6, random_state=1108, n_estimators=100)

x0 = df_train0.loc[df_train0['treatment'] == 'control', 0:248]
y0 = df_train0.loc[df_train0['treatment'] == 'control', 'adrd']
xgb_plugin0.fit(x0, y0)
x1 = df_train0.loc[df_train0['treatment'] == 'treatment', 0:248]
y1 = df_train0.loc[df_train0['treatment'] == 'treatment', 'adrd']
xgb_plugin1.fit(x1, y1)

y_pred0 = xgb_plugin0.predict(x_test0)
y_pred1 = xgb_plugin1.predict(x_test0)
t_plugin = y_pred1 - y_pred0

t = df_train0['treatment']

treatment = [0] * len(t)
for i in range(len(t)):
    if t[i] == 'control':
        treatment[i] = 0
    else:
        treatment[i] = 1

xgb.fit(x_train0,treatment)
cate = cate_pred_train.flatten()
t_test = df_test0['treatment']
#print(t_test)
treatment_test = [0] * len(t_test)
for i in range(len(t_test)):
    if t_test[i] == 'control':
        treatment_test[i] = 0
    else:
        treatment_test[i] = 1 
        
y_test = df_test0['adrd']

t_val = df_val0['treatment']
treatment_val = [0] * len(t_val)
for i in range(len(t_val)):
    if t_val[i] == 'control':
        treatment_val[i] = 0
    else:
        treatment_val[i] = 1 
        
plug_in = (t_plugin-cate)**2
ps = xgb.predict_proba(x_test0)[:, 1]
a = (treatment_test - ps)
ident = np.array([1]*len(ps))
c = (ps*(ident-ps))
b = np.array([2]*len(treatment_test))*treatment_test*(treatment_test-ps) / c
y_pred1.sum()
l_de = (ident - b) * t_plugin**2 + b*y_val*(t_plugin - cate) + (- a*(t_plugin - cate)**2 + cate**2)

print("PEHE for CFR net is: ",np.sum(l_de) + np.sum(plug_in))



#Dragonnet
ermse = 0.0
cate_pred_all = []
ctr = 0
for k in range(5,6):
    print("\n For treatment: ",k)
    
    t = SNet2(val_split_prop=0.25,n_iter=1500,batch_size=500)
    w = np.array(Treatment[0:tot_tv,k],dtype=int)
    y = np.array(Y[0:tot_tv],dtype=int)
    X_train = Xs[0:tot_tv]
    #print(X_train)
    X_test = Xs[tot_tv :totexs]
    #cate = cate[0:tot_tv]
    p = P[0:tot_tv]
    t.fit(X_train, y, w) # w is t here 
    cate_pred_train = t.predict(X_test) # without potential outcomes
    t_test = SNet2(val_split_prop=0.1,n_iter=1500,batch_size=500)
    print("Now fitting on test data for ERMSE")
    t_test.fit(X_test, Y[tot_tv :totexs], np.array(Treatment[tot_tv :totexs,k],dtype=int)) # w is t here 
    cate_pred_test = t_test.predict(X_test) 
    cate_pred_all.append(cate_pred_train)
    ermse = ermse+(np.square(cate_pred_train - cate_pred_test).mean())
    ctr = ctr+1
    

    
ermse = ermse/ctr


print("The estimated root mean squared error for the DragonNet is: ",ermse)


xgb = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin1 = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin0 = XGBClassifier(max_depth=6, random_state=1108, n_estimators=100)
xgb_plugin0.fit(x0, y0)
xgb_plugin1.fit(x1, y1)
y_pred0 = xgb_plugin0.predict(x_test0)
y_pred1 = xgb_plugin1.predict(x_test0)
t_plugin = y_pred1 - y_pred0
xgb.fit(x_train0,treatment)
cate = cate_pred_train.flatten()
plug_in = (t_plugin-cate)**2
ps = xgb.predict_proba(x_test0)[:, 1]
a = (treatment_test - ps)
ident = np.array([1]*len(ps))
c = (ps*(ident-ps))
b = np.array([2]*len(treatment_test))*treatment_test*(treatment_test-ps) / c
l_de = (ident - b) * t_plugin**2 + b*y_val*(t_plugin - cate) + (- a*(t_plugin - cate)**2 + cate**2)
print("PEHE for Dragon net is: ",np.sum(l_de) + np.sum(plug_in))


#DR Net 

ermse = 0.0
cate_pred_all = []
ctr = 0
for k in range(5,6):
    print("\n For treatment: ",k)
    
    t = DRNet(val_split_prop=0.25,n_iter=1500,batch_size=100)
    w = np.array(Treatment[0:tot_tv,k],dtype=int)
    y = np.array(Y[0:tot_tv],dtype=int)
    X_train = Xs[0:tot_tv]
    #print(X_train)
    X_test = Xs[tot_tv :totexs]
    #cate = cate[0:tot_tv]
    p = P[0:tot_tv]
    t.fit(X_train, y, w) # w is t here 
    cate_pred_train = t.predict(X_test) # without potential outcomes
    t_test = SNet1(val_split_prop=0.1,n_iter=1500,batch_size=100)
    print("Now fitting on test data for ERMSE")
    t_test.fit(X_test, Y[tot_tv :totexs], np.array(Treatment[tot_tv :totexs,k],dtype=int)) # w is t here 
    cate_pred_test = t_test.predict(X_test) 
    cate_pred_all.append(cate_pred_train)
    ermse = ermse+(np.square(cate_pred_train - cate_pred_test).mean())
    ctr = ctr+1
    

    
ermse = ermse/ctr
print("The estimated root mean squared error for the DRNet is: ",ermse)

xgb = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin1 = XGBClassifier(max_depth=6, random_state=1105, n_estimators=100)
xgb_plugin0 = XGBClassifier(max_depth=6, random_state=1108, n_estimators=100)
xgb_plugin0.fit(x0, y0)
xgb_plugin1.fit(x1, y1)
y_pred0 = xgb_plugin0.predict(x_test0)
y_pred1 = xgb_plugin1.predict(x_test0)
t_plugin = y_pred1 - y_pred0
xgb.fit(x_train0,treatment)
cate = cate_pred_train.flatten()
plug_in = (t_plugin-cate)**2
ps = xgb.predict_proba(x_test0)[:, 1]
a = (treatment_test - ps)
ident = np.array([1]*len(ps))
c = (ps*(ident-ps))
b = np.array([2]*len(treatment_test))*treatment_test*(treatment_test-ps) / c
l_de = (ident - b) * t_plugin**2 + b*y_val*(t_plugin - cate) + (- a*(t_plugin - cate)**2 + cate**2)

print("PEHE for DR net is: ",np.sum(l_de) + np.sum(plug_in))





