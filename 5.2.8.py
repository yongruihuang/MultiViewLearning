#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import datetime
import pickle
import scipy.sparse as ss
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
from sklearn import linear_model, svm, neural_network, ensemble


import IPython.display as ipd
import copy
import random
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold

import torch.nn as nn
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook as tqdm
import os


# # Read

# In[2]:


logging.info('start read')
df_master_records = pickle.load(open('../data_sortout/df_master_records.pickle', 'rb'))
se_id_install_list = pickle.load(open('../data_sortout/se_id_install_list.pickle', 'rb'))
df_install_behave = pickle.load(open('../data_sortout/df_install_behave_no_date.pickle', 'rb'))
df_behave_time = pickle.load(open('../data_sortout/df_time_cut.pickle', 'rb'))
se_userlog_cross = pickle.load(open('../data_sortout/se_userlog_cross_id.pickle', 'rb'))
logging.info('finish read')


# In[3]:


VAR_SAVE = 'var/baseline_'


# # Feature

# ## user attribute

# In[4]:


def get_master_user_discrete(df_master_records):
    
    df_master_records['qcut_amount_bin'] = pd.qcut(df_master_records['amount_bin'], 5)
    df_master_records['new_client'] = df_master_records['loan_sequence'] == 1
    df_master_records['qcut_age'] = pd.qcut(df_master_records['age'], 5, duplicates='drop')
    
    df_master_records['qcut_min_income'] = pd.qcut(df_master_records['min_income'], 6, duplicates='drop')
    df_master_records['qcut_max_income'] = pd.qcut(df_master_records['max_income'].apply(int), 6, duplicates='drop')
    
#     df_master_records['qcut_loan_sequence'] = pd.qcut(df_master_records['loan_sequence'], 6, duplicates='drop')

    pne_hot_cols = [ 'months', 'gender', 'educationid', 'marriagestatusid', 'income', 
                    'qcut_amount_bin', 'qcut_age', 'qcut_min_income', 'qcut_max_income']
    
    return  pd.get_dummies(df_master_records[pne_hot_cols], columns = pne_hot_cols)


# ## app

# In[5]:


class MutualInfoSelection():
    
    def __init__(self, topN):
        self.topN = topN
    
    def get_select_vector(self, pkgs):
        """
        """
        v = np.zeros(len(self.mp_select_feature_order))
        for pkg in pkgs:
            if pkg in self.mp_select_feature_order:
                v[self.mp_select_feature_order[pkg]] = 1
        return v

    def fit(self, train_se_id_pkg_list, train_y):
        rows = []
        cols = []
        data = []
        for i in range(train_se_id_pkg_list.shape[0]):
            pkg_list = train_se_id_pkg_list.iloc[i]
            cols.extend(pkg_list)
            rows.extend([i] * len(pkg_list))
            data.extend([1] * len(pkg_list))
        train_sparse = ss.coo_matrix((data,(rows,cols)))

        mutual_weight = mutual_info_classif(train_sparse, train_y)
        select_feature = np.argsort(mutual_weight)[-self.topN:]
        self.mp_select_feature_order = {}
        self.mp_order_select_feature = {}

        for i, feature in enumerate(select_feature):
            self.mp_select_feature_order[feature] = i
            self.mp_order_select_feature[i] = feature
    
    def transform(self, se_id_pkg_list):
        se_app_feature = se_id_pkg_list.apply(self.get_select_vector)
        df_app_feature = pd.DataFrame(np.array(list(se_app_feature)))
        df_app_feature.index = se_app_feature.index

        return df_app_feature


# # Train

# ## 数据划分

# In[5]:


split_date = datetime.datetime(2019, 8, 31)
end_date = datetime.datetime(2019, 9, 30)

df_master_records = df_master_records.dropna(axis=0, how='any')
df_train_master = df_master_records.query('loan_date <= @split_date')
df_test_master = df_master_records.query('loan_date > @split_date & loan_date <= @end_date')
all_train_id = list(df_train_master.index)
all_test_id = list(df_test_master.index)
logging.info('all_train_id len :%d, all_test_id: %d' % (len(all_train_id), len(all_test_id)))


# ## 共用函数

# In[6]:


def train(prefix, dataset_dict, model):
    
    train_x = dataset_dict['train_x']
    train_y = dataset_dict['train_y']
    test_x = dataset_dict['test_x']
    test_y = dataset_dict['test_y']
    train_new_client = dataset_dict['train_new_client'].values
    test_new_client = dataset_dict['test_new_client'].values
    
#     sc = StandardScaler()
#     sc.fit(train_x)
#     train_x = sc.transform(train_x)
#     test_x = sc.transform(test_x)
    
    model.fit(train_x, train_y)
    
    predict_train = model.predict_proba(train_x)
    predict_test = model.predict_proba(test_x)
    
    auc_train = roc_auc_score(train_y, predict_train[:, 1])
    auc_test = roc_auc_score(test_y, predict_test[:, 1])
    
    new_auc_train = roc_auc_score(train_y[train_new_client], predict_train[:, 1][train_new_client])
    new_auc_test = roc_auc_score(test_y[test_new_client], predict_test[:, 1][test_new_client])

    old_auc_train = roc_auc_score(train_y[~train_new_client], predict_train[:, 1][~train_new_client])
    old_auc_test = roc_auc_score(test_y[~test_new_client], predict_test[:, 1][~test_new_client])

    train_ret_dict = {
        prefix : auc_train,
        "new_%s" % prefix: new_auc_train,
        "old_%s" % prefix : old_auc_train
    }
    
    test_ret_dict = {
        prefix : auc_test,
        "new_%s" % prefix: new_auc_test,
        "old_%s" % prefix : old_auc_test
    }
    return train_ret_dict, test_ret_dict

def get_dataset_user(df_feature, target_name, save_name, load_from_file = True):
    data_path = VAR_SAVE + save_name + target_name + '.pickle'
    if load_from_file and (os.path.exists(data_path)):
        return pickle.load(open(data_path, 'rb'))
    
    select_id = df_feature.index
    select_train_id = list( set(select_id) & set(all_train_id) )
    select_test_id = list( set(select_id) & set(all_test_id) )
    select_df_train = df_feature.loc[select_train_id]
    select_df_test = df_feature.loc[select_test_id]
    select_train_y = df_master_records.loc[select_train_id][target_name]
    select_test_y = df_master_records.loc[select_test_id][target_name]
    
    ret = {
        'train_x' : np.array(select_df_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(select_df_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }, select_df_train, select_df_test
    
    pickle.dump(ret, open(data_path, 'wb'), protocol = 4)
    return ret

def get_dataset_app(se_id_pkg, target_name, save_name, load_from_file = True):
    
    data_path = VAR_SAVE + save_name + target_name + '.pickle'
    if(load_from_file and os.path.exists(data_path)):
        return pickle.load(open(data_path, 'rb'))

    select_id = se_id_pkg.index
    select_train_id = list( set(select_id) & set(all_train_id) )
    select_test_id = list( set(select_id) & set(all_test_id) )
    
    app_selection = MutualInfoSelection(3000)
    select_train_y = df_master_records.loc[select_train_id][target_name]
    
    app_selection.fit(se_id_pkg.loc[select_train_id], select_train_y.values)
    logging.info('finish fit app_selection :%s' % target_name)
    select_df_train = app_selection.transform(se_id_pkg.loc[select_train_id])
    select_df_test = app_selection.transform(se_id_pkg.loc[select_test_id])
    
    select_test_y = df_master_records.loc[select_test_id][target_name]
    
    ret = {
        'train_x' : np.array(select_df_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(select_df_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }, select_df_train, select_df_test
    
    pickle.dump(ret, open(data_path, 'wb'), protocol = 4)
    return ret

def get_dataset_userlog(se_id_pkg, target_name, save_name, load_from_file = True):
    data_path = VAR_SAVE + save_name + target_name + '.pickle'
    if(load_from_file and os.path.exists(data_path)):
        return pickle.load(open(data_path, 'rb'))

    select_id = se_id_pkg.index
    select_train_id = list( set(select_id) & set(all_train_id) )
    select_test_id = list( set(select_id) & set(all_test_id) )
    
    se_train = se_id_pkg.loc[select_train_id]
    se_test = se_id_pkg.loc[select_test_id]
    
    userlog_action = 137
    train_np = np.zeros((se_train.shape[0], userlog_action))
    test_np = np.zeros((se_test.shape[0], userlog_action))
    
    for i in tqdm(range(se_train.shape[0])):
        for item in se_train.iloc[i]:
            train_np[i][item-1] += 1
    
    
    for i in tqdm(range(se_test.shape[0])):
        for item in se_test.iloc[i]:
            test_np[i][item-1] += 1
    
    select_df_train = pd.DataFrame(train_np, index=se_train.index)
    select_df_test = pd.DataFrame(test_np, index=se_test.index)
    select_train_y = df_master_records.loc[select_train_id][target_name]
    select_test_y = df_master_records.loc[select_test_id][target_name]

    ret = {
        'train_x' : np.array(select_df_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(select_df_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }, select_df_train, select_df_test
    
    pickle.dump(ret, open(data_path, 'wb'), protocol = 4)
    return ret

def get_dataset_merge(df_feature, target_name):
    
    select_id = df_feature.index
    select_train_id = list( set(select_id) & set(all_train_id) )
    select_test_id = list( set(select_id) & set(all_test_id) )
    select_df_train = df_feature.loc[select_train_id]
    select_df_test = df_feature.loc[select_test_id]
    select_train_y = df_master_records.loc[select_train_id][target_name]
    select_test_y = df_master_records.loc[select_test_id][target_name]
    
    ret = {
        'train_x' : np.array(select_df_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(select_df_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }
    
    return ret


# ## user attribute

# In[8]:


df_user_one_hot = get_master_user_discrete(df_master_records)
user_train_df_list, user_test_df_list = [], []

for i, model in enumerate([
    linear_model.LogisticRegression(), 
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    print(model)
    
    train_ret_dict, test_ret_dict = {}, {}
    

    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        user_dataset_dict, user_train_df, user_test_df = get_dataset_user(df_user_one_hot, 'target_%s' % target_name, 'user_info_')
        if i == 0:
            user_train_df_list.append(user_train_df)
            user_test_df_list.append(user_test_df)
        
        logging.info('start train user attribute:%s' % target_name)
        use_rets = train(target_name, user_dataset_dict, model)
        train_ret_dict.update(use_rets[0])
        test_ret_dict.update(use_rets[1])
    
    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# ## App list

# In[9]:


applist_train_df_list, applist_test_df_list = [], []

for i, model in enumerate([
    linear_model.LogisticRegression(), 
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0), 
]):
    print(model)
    train_ret_dict, test_ret_dict = {}, {}

    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        logging.info('start train app list:%s' % target_name)
        app_list_dataset_dict, applist_train_df, applist_test_df =         get_dataset_app(se_id_install_list, 'target_%s' % target_name, 'app_list_')

        if i == 0:
            applist_train_df_list.append(applist_train_df)
            applist_test_df_list.append(applist_test_df)
            
        use_rets = train(target_name, app_list_dataset_dict, model)
        train_ret_dict.update(use_rets[0])
        test_ret_dict.update(use_rets[1])
    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# ## App install behave

# In[10]:


appbehave_df_train, appbehave_df_test = [], []

for i, model in enumerate([
    linear_model.LogisticRegression(),
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    print(model)
    train_ret_dict, test_ret_dict = {}, {}
    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        logging.info('start train app behave:%s' % target_name)
        app_behave_dataset_dict, behave_df_train, behave_df_test =             get_dataset_app(df_install_behave['pkg_id'], 'target_%s' % target_name, 'app_behave_')

        if i == 0:
            appbehave_df_train.append(behave_df_train)
            appbehave_df_test.append(behave_df_test)
            
        use_rets = train(target_name, app_behave_dataset_dict, model)
        train_ret_dict.update(use_rets[0])
        test_ret_dict.update(use_rets[1])
    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# ## userlog

# In[11]:


userlog_df_train_list, userlog_df_test_list, userlog_dataset_dict_list = [], [], []
for i, model in enumerate([
    linear_model.LogisticRegression(), 
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    print(model)
    
    train_ret_dict, test_ret_dict = {}, {}
    for j,target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        logging.info('start train user log:%s' % target_name)
        userlog_dataset_dict, userlog_df_train, userlog_df_test =         get_dataset_userlog(se_userlog_cross, 'target_%s' % target_name, 'user_log_')

        if  i == 0:
            userlog_df_train_list.append(userlog_df_train)
            userlog_df_test_list.append(userlog_df_test)
    
        use_rets = train(target_name, userlog_dataset_dict, model)
        train_ret_dict.update(use_rets[0])
        test_ret_dict.update(use_rets[1])
    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# ## merge (drop missing)

# In[ ]:


def merge_df_list(user_df_list, applist_df_list, appbehave_df, userlog_list, drop = True):
    df_merge_list = []
    logging.info('start merge')
    for (df_user_attribute, df_applist, df_appbehave, df_userlog) in zip(user_df_list, applist_df_list, appbehave_df, userlog_list):
        df_merge = pd.concat([df_user_attribute, df_applist, df_appbehave, df_userlog], axis=1)
        if drop:
            df_merge = df_merge.dropna()
        else:
            df_merge = df_merge.fillna(0)
        df_merge_list.append(df_merge)
    logging.info('finish merge')
    return df_merge_list


# In[ ]:


df_merge_train_list = merge_df_list(user_train_df_list, applist_train_df_list, appbehave_df_train, userlog_df_train_list)
df_merge_test_list = merge_df_list(user_test_df_list, applist_test_df_list, appbehave_df_test, userlog_df_test_list)


# In[14]:


for i, model in enumerate([
    linear_model.LogisticRegression(), 
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    print(model)
    train_ret_dict, test_ret_dict = {}, {}
    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        df_merge = pd.concat([df_merge_train_list[j], df_merge_test_list[j]])
        merge_dataset = get_dataset_merge(df_merge, 'target_%s' % target_name)
        logging.info('start train merge attribute:%s' % target_name)
        merge_rets = train(target_name, merge_dataset, model)
        train_ret_dict.update(merge_rets[0])
        test_ret_dict.update(merge_rets[1])

    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# ## merge fill zeros

# In[ ]:


# df_merge_train_list = merge_df_list(user_train_df_list, applist_train_df_list, appbehave_df_train, userlog_df_train_list, False)
# df_merge_test_list = merge_df_list(user_test_df_list, applist_test_df_list, appbehave_df_test, userlog_df_test_list,  False)


# In[ ]:


# merge_dataset_list = []
# for i, model in enumerate([
#     linear_model.LogisticRegression(), 
#     svm.SVC(probability=True, max_iter = 100),
#     neural_network.MLPClassifier(random_state=0, max_iter=300),
#     ensemble.GradientBoostingClassifier(random_state=0),
# ]):
#     print(model)
#     train_ret_dict, test_ret_dict = {}, {}
#     for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
#         df_merge = pd.concat([df_merge_train_list[j], df_merge_test_list[j]])
#         merge_dataset = get_dataset_merge(df_merge, 'target_%s' % target_name)
#         merge_dataset_list.append(merge_dataset)

#         logging.info('start train merge fill zeros attribute:%s' % target_name)
#         merge_rets = train(target_name, user_dataset_dict, model)
#         train_ret_dict.update(merge_rets[0])
#         test_ret_dict.update(merge_rets[1])

#     df_ret = pd.DataFrame([
#         train_ret_dict, test_ret_dict
#     ], index = ['train', 'test'])
#     ipd.display(df_ret)


# In[7]:


def get_df_merge_dataset(target_name):
    logging.info('start loading %s' % target_name)
    _, df_train_user_info, df_test_user_info = pickle.load(open('var/baseline_user_info_%s.pickle' % target_name, 'rb'))
    _, df_train_app_list, df_test_app_list = pickle.load(open('var/baseline_app_list_%s.pickle' % target_name, 'rb'))
    _, df_train_app_behave, df_test_app_behave = pickle.load(open('var/baseline_app_behave_%s.pickle' % target_name, 'rb')) 
    _, df_train_user_log, df_test_user_log = pickle.load(open('var/baseline_user_log_%s.pickle' % target_name, 'rb')) 
    df_merge_train = pd.concat([df_train_user_info, df_train_app_list, df_train_app_behave, df_train_user_log], axis=1)
    df_merge_train = df_merge_train.fillna(0)
    df_merge_test = pd.concat([df_test_user_info, df_test_app_list, df_test_app_behave, df_test_user_log], axis=1)
    df_merge_test = df_merge_test.fillna(0)
 
    select_train_id = df_merge_train.index
    select_test_id = df_merge_test.index 
    select_train_y = df_master_records.loc[select_train_id][target_name]
    select_test_y = df_master_records.loc[select_test_id][target_name]
    
    ret = {
        'train_x' : np.array(df_merge_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(df_merge_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }

    logging.info('finish loading %s' % target_name)

    return ret


# In[ ]:


for i, model_orgin in enumerate([
    linear_model.LogisticRegression(), 
    svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    model = copy.deepcopy(model_orgin)
    print(model)
    train_ret_dict, test_ret_dict = {}, {}
    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        merge_dataset_dict = get_df_merge_dataset('target_%s' % target_name)
        logging.info('start train merge fill zeros attribute:%s' % target_name)
        merge_rets = train(target_name, merge_dataset_dict, model)
        train_ret_dict.update(merge_rets[0])
        test_ret_dict.update(merge_rets[1])

    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# In[8]:


for i, model_orgin in enumerate([
    ensemble.GradientBoostingClassifier(random_state=0),
]):
    model = copy.deepcopy(model_orgin)
    print(model)
    train_ret_dict, test_ret_dict = {}, {}
    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        merge_dataset_dict = get_df_merge_dataset('target_%s' % target_name)
        logging.info('start train merge fill zeros attribute:%s' % target_name)
        merge_rets = train(target_name, merge_dataset_dict, model)
        train_ret_dict.update(merge_rets[0])
        test_ret_dict.update(merge_rets[1])

    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# In[15]:


import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
from transformers import *

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class DNN(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        hidden = 128
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)

class DNNTrain():
    def __init__(self):
        self.model = None
        
    def fit(self, X, y):
        self.model = DNN(X.shape[1]).cuda()
        self.model.train()

        torch_dataset = Data.TensorDataset(torch.tensor(X.astype('float32')), torch.tensor(y.astype('int')))
        data_loader = Data.DataLoader(
            dataset=torch_dataset,      
            batch_size=1024,      
            shuffle=True,
            num_workers = 0,
        )

        optimizer = AdamW(self.model.parameters(), lr = 0.01, weight_decay = 0.01)
 
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(12):
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                pre = self.model(batch_x)    
                loss = loss_func(pre, batch_y.long())
                #backward
                S = time.time()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 2)
                optimizer.step()
                
    def predict_proba(self, X):
        self.model.eval()
        torch_dataset = Data.TensorDataset(torch.tensor(X.astype('float32')))
        data_loader = Data.DataLoader(
            dataset=torch_dataset,      
            batch_size=1024,      
            shuffle=True,
            num_workers = 0,
        )
        pre_list = []
        for i, (batch_x,) in enumerate(tqdm(data_loader)):
            pre = self.model(batch_x.cuda())    
            pre_list.append(pre.cpu().detach().numpy())
        return np.concatenate(pre_list,  axis = 0)


# In[8]:


train_ret_dict, test_ret_dict = {}, {}
for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
    merge_dataset_dict = get_df_merge_dataset('target_%s' % target_name)
    logging.info('start train merge fill zeros attribute:%s' % target_name)
    merge_rets = train(target_name, merge_dataset_dict, neural_network.MLPClassifier(random_state=0, max_iter=100))
    print(merge_rets)
    train_ret_dict.update(merge_rets[0])
    test_ret_dict.update(merge_rets[1])

df_ret = pd.DataFrame([
    train_ret_dict, test_ret_dict
], index = ['train', 'test'])
ipd.display(df_ret)


# ## App concat

# In[8]:


def get_app_merge_dataset(target_name):
    logging.info('start loading %s' % target_name)
    _, df_train_app_list, df_test_app_list = pickle.load(open('var/baseline_app_list_%s.pickle' % target_name, 'rb'))
    _, df_train_app_behave, df_test_app_behave = pickle.load(open('var/baseline_app_behave_%s.pickle' % target_name, 'rb')) 
    df_merge_train = pd.concat([df_train_app_list, df_train_app_behave], axis=1)
    df_merge_train = df_merge_train.fillna(0)
    df_merge_test = pd.concat([df_test_app_list, df_test_app_behave], axis=1)
    df_merge_test = df_merge_test.fillna(0)
 
    select_train_id = df_merge_train.index
    select_test_id = df_merge_test.index 
    select_train_y = df_master_records.loc[select_train_id][target_name]
    select_test_y = df_master_records.loc[select_test_id][target_name]
    
    ret = {
        'train_x' : np.array(df_merge_train),
        'train_y' : np.array(select_train_y),
        'test_x' : np.array(df_merge_test),
        'test_y' : np.array(select_test_y),
        'train_new_client' : df_master_records.loc[select_train_id]['loan_sequence'] == 1,
        'test_new_client' : df_master_records.loc[select_test_id]['loan_sequence'] == 1
    }

    logging.info('finish loading %s' % target_name)

    return ret


# In[9]:


for i, model_orgin in enumerate([
#     linear_model.LogisticRegression(), 
#     svm.SVC(probability=True, max_iter = 100),
    neural_network.MLPClassifier(random_state=0, max_iter=300),
#     ensemble.GradientBoostingClassifier(random_state=0),
]):
    model = copy.deepcopy(model_orgin)
    print(model)
    
    train_ret_dict, test_ret_dict = {}, {}
    for j, target_name in enumerate(['1m30+', '2m30+', '3m30+', '4m30+']):
        merge_dataset_dict = get_app_merge_dataset('target_%s' % target_name)
        logging.info('start train merge fill zeros attribute:%s' % target_name)
        merge_rets = train(target_name, merge_dataset_dict, model)
        train_ret_dict.update(merge_rets[0])
        test_ret_dict.update(merge_rets[1])

    df_ret = pd.DataFrame([
        train_ret_dict, test_ret_dict
    ], index = ['train', 'test'])
    ipd.display(df_ret)


# In[ ]:




