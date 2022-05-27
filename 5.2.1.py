#!/usr/bin/env python
# coding: utf-8

# # import 

# In[1]:


import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
# import seaborn as sns

import IPython.display as ipd
import copy
import random
from pandarallel import pandarallel
# Initialization
pandarallel.initialize(progress_bar=True)
# df.parallel_apply(func)
import time
from gensim.models.word2vec import Word2Vec 
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold

from transformers import *
import torch.nn as nn
import math
from tqdm import tqdm_notebook as tqdm
from transformers.modeling_bert import BertConfig, BertEncoder, BertAttention,BertSelfAttention,BertLayer,BertPooler,BertLayerNorm

from gensim.models.word2vec import Word2Vec 
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
from scipy.special import softmax

from category_encoders import *


# # Read

# In[2]:


logging.info('start read')
df_master_records = pickle.load(open('../data_sortout/df_master_records.pickle', 'rb'))
se_id_install_list = pickle.load(open('../data_sortout/se_id_install_list.pickle', 'rb'))
df_install_behave = pickle.load(open('../data_sortout/df_install_behave_no_date.pickle', 'rb'))
df_behave_time = pickle.load(open('../data_sortout/df_time_cut.pickle', 'rb'))
# df_userlog = pickle.load(open('../data_sortout/df_userlog_sequence_less.pickle', 'rb'))
se_userlog_cross = pickle.load(open('../data_sortout/se_userlog_cross_id.pickle', 'rb'))
df_userlog_time_seq = pickle.load(open('../data_sortout/df_userlog_time_seq.pickle', 'rb'))

# df_app_list_te_sequence = pickle.load(open('../data_sortout/df_app_list_target_encode_sequence.pickle', 'rb'))
# df_app_behave_te_sequence = pickle.load(open('../data_sortout/df_app_behave_target_encode_sequence.pickle', 'rb'))

df_app_list_te_qcut = pickle.load(open('../data_sortout/df_app_list_target_qcut.pickle', 'rb'))
df_app_behave_te_qcut = pickle.load(open('../data_sortout/df_app_behave_target_qcut.pickle', 'rb'))


logging.info('finish read')


# In[3]:


min(df_master_records['loan_date']), max(df_master_records['loan_date']), 


# In[4]:


df_master_records.shape, se_id_install_list.shape, df_install_behave.shape, se_userlog_cross.shape


# In[5]:


print(len(set(se_id_install_list.index) & set(df_install_behave.index) & set(se_userlog_cross.index)), len(set(se_id_install_list.index) & set(df_install_behave.index) )      ,len(set(se_id_install_list.index) & set(se_userlog_cross.index))      ,len(set(df_install_behave.index) & set(se_userlog_cross.index))    )  


# # 数据划分

# In[6]:


split_date = datetime.datetime(2019, 8, 31)
end_date = datetime.datetime(2019, 9, 30)

df_master_records = df_master_records.dropna(axis=0, how='any')
df_train_master = df_master_records.query('loan_date <= @split_date')
df_test_master = df_master_records.query('loan_date > @split_date & loan_date <= @end_date')
all_train_id = list(df_train_master.index)
all_test_id = list(df_test_master.index)
logging.info('all_train_id len :%d, all_test_id: %d' % (len(all_train_id), len(all_test_id)))
df_target = df_master_records[['target_1m30+', 'target_2m30+', 'target_3m30+', 'target_4m30+']]


# In[7]:


max_app_list_id = max(se_id_install_list.apply(max))
max_app_behave_id = max(df_install_behave['pkg_id'].apply(max))
max_uselog_id = max(se_userlog_cross.apply(max))
start_app_list_id = max_app_list_id + 1
start_app_behave_id = max_app_behave_id + 1 
start_uselog_id = max_uselog_id + 1


# # Feature

# ## user info

# In[8]:


def get_master_user_discrete(df_master_records):
    
    df_master_records['qcut_amount_bin'] = pd.qcut(df_master_records['amount_bin'], 5)
    df_master_records['new_client'] = df_master_records['loan_sequence'] == 1
    df_master_records['qcut_age'] = pd.qcut(df_master_records['age'], 5, duplicates='drop')
    
    df_master_records['qcut_min_income'] = pd.qcut(df_master_records['min_income'], 6, duplicates='drop')
    df_master_records['qcut_max_income'] = pd.qcut(df_master_records['max_income'].apply(int), 6, duplicates='drop')

#     df_master_records['qcut_loan_sequence'] = pd.qcut(df_master_records['loan_sequence'], 6, duplicates='drop')
#     pne_hot_cols = ['months', 'gender', 'educationid', 'marriagestatusid', 'income', 
#                     'qcut_amount_bin', 'new_client', 'qcut_loan_sequence', 'qcut_age', 'qcut_min_income', 'qcut_max_income']
    pne_hot_cols = ['months', 'gender', 'educationid', 'marriagestatusid', 'income', 
                'qcut_amount_bin', 'qcut_age', 'qcut_min_income', 'qcut_max_income']

    return  pd.get_dummies(df_master_records[pne_hot_cols], columns = pne_hot_cols)
df_user_one_hot = get_master_user_discrete(df_master_records)


# In[9]:


df_user_one_hot.shape


# ## target encode

# ### tools

# In[10]:


def target_encode_cross_validation(df_train, df_test, se_target_col, random_seed = 0):
    
    n_flod = 5
    folds = KFold(n_splits=n_flod, shuffle=True, random_state=random_seed)
        
    df_vals = []
    df_test_ret = df_test[['pkg_id']].copy()
    df_test_ret[['pkg_id']] = 0
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(df_train, se_target_col)):
        
        target_encoder = TargetEncoder(cols=['pkg_id'])
        train_df = df_train.iloc[trn_idx]
        val_df = df_train.iloc[val_idx]
        target_encoder.fit(train_df['pkg_id'], se_target_col.iloc[trn_idx])
        
        df_vals.append(target_encoder.transform(val_df['pkg_id']))  
        
        df_test_ret[['pkg_id']] += target_encoder.transform(df_test['pkg_id'])[['pkg_id']] / n_flod
        
        logging.info("finish fold : %d" % n_fold)    
    
    df_val_ret = pd.concat(df_vals)
    df_val_ret = df_val_ret.sort_index()
    
    df_val_ret.columns = se_target_col.name + '_' + df_val_ret.columns
    df_test_ret.columns = se_target_col.name + '_' + df_test_ret.columns
    return df_val_ret, df_test_ret

def target_sort_out(se_data):
    master_ids = []
    pkg_ids = []
    for master_id in se_data.index:
        pkg_ids.extend(list(se_data.loc[master_id]))
        master_ids.extend([master_id] * len(se_data.loc[master_id]))
    
    return pd.DataFrame({
        'master_id' : master_ids,
        'pkg_id' : pkg_ids,
    })


# ### app list

# In[9]:


df_app_list_target_id = target_sort_out(se_id_install_list)
df_app_list_target_id = pd.merge(df_app_list_target_id, df_target, how = 'left', left_on = 'master_id', right_index = True)
df_app_list_target_id = df_app_list_target_id.dropna()


# In[ ]:


df_train_app_list = df_app_list_target_id[df_app_list_target_id.master_id.isin(set(all_train_id))]
df_test_app_list = df_app_list_target_id[df_app_list_target_id.master_id.isin(set(all_test_id))]


# In[ ]:


df_app_list_train_target_encode_list, df_app_list_test_target_encode_list = [], []
for target in ['target_1m30+', 'target_2m30+', 'target_3m30+', 'target_4m30+']:
    df_app_list_train_target_encode, df_app_list_test_target_encode = target_encode_cross_validation(df_train_app_list, df_test_app_list, df_train_app_list[target])
    df_app_list_train_target_encode_list.append(df_app_list_train_target_encode)
    df_app_list_test_target_encode_list.append(df_app_list_test_target_encode)


# In[13]:


df_app_list_train_target_encode = pd.concat(df_app_list_train_target_encode_list, axis = 1)
df_app_list_test_target_encode = pd.concat(df_app_list_test_target_encode_list, axis = 1)
df_app_list_target_encode = df_app_list_train_target_encode.append(df_app_list_test_target_encode).sort_index()


# In[15]:


df_app_list_target_encode_with_id = pd.concat([df_app_list_target_id, df_app_list_target_encode], axis = 1)


# In[18]:


se_target_qcut_list = []
for target_name in ['target_1m30+_pkg_id', 'target_2m30+_pkg_id', 'target_3m30+_pkg_id', 'target_4m30+_pkg_id']:
    se_target_qcut = pd.qcut(df_app_list_target_encode[target_name], 16)
    se_target_qcut_list.append(se_target_qcut)


# In[19]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[20]:


df_target_qcut = pd.concat(se_target_qcut_list, axis = 1)
df_target_qcut['target_1m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_1m30+_pkg_id']) + 1
df_target_qcut['target_2m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_2m30+_pkg_id']) + 1
df_target_qcut['target_3m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_3m30+_pkg_id']) + 1
df_target_qcut['target_4m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_4m30+_pkg_id']) + 1


# In[21]:


df_app_list_target_qcut_with_id = pd.concat([df_app_list_target_id[['master_id', 'pkg_id']], df_target_qcut[['target_1m30_qcut', 'target_2m30_qcut', 'target_3m30_qcut', 'target_4m30_qcut',]]], axis = 1)


# In[22]:


df_app_list_target_encode_sequence = df_app_list_target_qcut_with_id.groupby('master_id').apply(lambda x : pd.Series({
    'target_1m30_qcut' : x['target_1m30_qcut'].values,
    'target_2m30_qcut' : x['target_2m30_qcut'].values,
    'target_3m30_qcut' : x['target_3m30_qcut'].values,
    'target_4m30_qcut' : x['target_4m30_qcut'].values,
    'pkg_id' : x['pkg_id'].values,
}))
# pickle.dump(df_app_list_target_encode_sequence, open('../data_sortout/df_app_list_target_qcut.pickle', 'wb'))


# In[34]:


check_id = 15161
df_app_list_target_encode_sequence['pkg_id'].loc[check_id] == se_id_install_list.loc[check_id]


# In[14]:


df_app_list_target_encode_with_id = pd.concat([df_app_list_target_id, df_app_list_target_encode], axis = 1)
df_app_list_target_encode_sequence = df_app_list_target_encode_with_id.groupby('master_id').apply(lambda x : pd.Series({
    'target_1m30+_pkg_id' : x['target_1m30+_pkg_id'].values,
    'target_2m30+_pkg_id' : x['target_2m30+_pkg_id'].values,
    'target_3m30+_pkg_id' : x['target_3m30+_pkg_id'].values,
    'target_4m30+_pkg_id' : x['target_4m30+_pkg_id'].values,
}))
pickle.dump(df_app_list_target_encode_sequence, open('../data_sortout/df_app_list_target_encode_sequence.pickle', 'wb'))


# ### app behave

# In[9]:


df_app_behave_target_id = target_sort_out(df_install_behave['pkg_id'])
df_app_behave_target_id = pd.merge(df_app_behave_target_id, df_target, how = 'left', left_on = 'master_id', right_index = True)
df_app_behave_target_id = df_app_behave_target_id.dropna()


# In[10]:


df_train_app_behave = df_app_behave_target_id[df_app_behave_target_id.master_id.isin(set(all_train_id))]
df_test_app_behave = df_app_behave_target_id[df_app_behave_target_id.master_id.isin(set(all_test_id))]


# In[11]:


df_app_behave_train_target_encode_list, df_app_behave_test_target_encode_list = [], []
for target in ['target_1m30+', 'target_2m30+', 'target_3m30+', 'target_4m30+']:
    df_app_behave_train_target_encode, df_app_behave_test_target_encode = target_encode_cross_validation(df_train_app_behave, df_test_app_behave, df_train_app_behave[target])
    df_app_behave_train_target_encode_list.append(df_app_behave_train_target_encode)
    df_app_behave_test_target_encode_list.append(df_app_behave_test_target_encode)


# In[12]:


df_app_behave_train_target_encode = pd.concat(df_app_behave_train_target_encode_list, axis = 1)
df_app_behave_test_target_encode = pd.concat(df_app_behave_test_target_encode_list, axis = 1)
df_app_behave_target_encode = df_app_behave_train_target_encode.append(df_app_behave_test_target_encode).sort_index()


# In[14]:


se_target_qcut_list = []
for target_name in ['target_1m30+_pkg_id', 'target_2m30+_pkg_id', 'target_3m30+_pkg_id', 'target_4m30+_pkg_id']:
    se_target_qcut = pd.qcut(df_app_behave_target_encode[target_name], 16)
    se_target_qcut_list.append(se_target_qcut)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df_target_qcut = pd.concat(se_target_qcut_list, axis = 1)
df_target_qcut['target_1m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_1m30+_pkg_id']) + 1
df_target_qcut['target_2m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_2m30+_pkg_id']) + 1
df_target_qcut['target_3m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_3m30+_pkg_id']) + 1
df_target_qcut['target_4m30_qcut'] = preprocessing.LabelEncoder().fit_transform(df_target_qcut['target_4m30+_pkg_id']) + 1


# In[15]:


df_app_behave_target_qcut_with_id = pd.concat([df_app_behave_target_id[['master_id', 'pkg_id']], df_target_qcut[['target_1m30_qcut', 'target_2m30_qcut', 'target_3m30_qcut', 'target_4m30_qcut',]]], axis = 1)
df_app_behave_target_encode_sequence = df_app_behave_target_qcut_with_id.groupby('master_id').apply(lambda x : pd.Series({
    'target_1m30_qcut' : x['target_1m30_qcut'].values,
    'target_2m30_qcut' : x['target_2m30_qcut'].values,
    'target_3m30_qcut' : x['target_3m30_qcut'].values,
    'target_4m30_qcut' : x['target_4m30_qcut'].values,
}))
pickle.dump(df_app_behave_target_encode_sequence, open('../data_sortout/df_app_behave_target_qcut.pickle', 'wb'))


# In[20]:


df_app_behave_target_encode_with_id = pd.concat([df_app_behave_target_id, df_app_behave_target_encode], axis = 1)


# In[21]:


df_app_behave_target_encode_sequence = df_app_behave_target_encode_with_id.groupby('master_id').apply(lambda x : pd.Series({
    'target_1m30+_pkg_id' : x['target_1m30+_pkg_id'].values,
    'target_2m30+_pkg_id' : x['target_2m30+_pkg_id'].values,
    'target_3m30+_pkg_id' : x['target_3m30+_pkg_id'].values,
    'target_4m30+_pkg_id' : x['target_4m30+_pkg_id'].values,
}))


# In[22]:


# df_app_behave_target_encode_sequence


# In[23]:


pickle.dump(df_app_behave_target_encode_sequence, open('../data_sortout/df_app_behave_target_encode_sequence.pickle', 'wb'))


# # word2vec

# ## app list

# In[10]:


sentences = []
for data_list in tqdm(se_id_install_list):
    sentence = [str(start_app_list_id),]
    for data in data_list:
        sentence.append(str(data))
    sentences.append(sentence)
window = 200
mincount = 1
n_dim = 100
logging.info('start word2vec')
wv_model= Word2Vec(sentences, min_count=mincount, size = n_dim, window = window, sg = 0, workers = 12)
logging.info('finish word2vec')
pickle.dump(wv_model, open('../data_sortout/wv_model_app_list.pickle', 'wb'))


# In[ ]:


# se_app_list_embedding = se_id_install_list.apply(lambda x : wv_model.wv[x].sum(axis = 0)) 
# def apply_wv_sum(app_list):
#     ret = np.zeros(100)
#     for app in app_list:
#         if(app in wv_model.wv):
#             ret += wv_model.wv[app]
#     return ret

# se_app_list_embedding = se_id_install_list.apply(apply_wv_sum) 
# se_app_list_embedding = pickle.load(open('/data/ccnth/ppd/se_app_list_embedding.pickle', 'rb'))


# ## app behave

# In[11]:


sentences = []
for data_list in tqdm(df_install_behave['pkg_id']):
    sentence = [str(start_app_behave_id),]
    for data in data_list:
        sentence.append(str(data))
    sentences.append(sentence)
window = 5
mincount = 1
n_dim = 100
logging.info('start word2vec install behave')
wv_model= Word2Vec(sentences, min_count=mincount, size = n_dim, window = window, sg = 0, workers = 12)
logging.info('finish word2vec install behave')
pickle.dump(wv_model, open('../data_sortout/wv_model_app_behave.pickle', 'wb'))


# In[ ]:


# logging.info('se_app_behave_embedding')
# def get_behave_embed(data_list):
#     data_list_str = [str(data) for data in data_list]
    
#     return wv_model_app_behave.wv[data_list_str].sum(axis = 0)
# se_app_behave_embedding = df_install_behave.pkg_id.apply(get_behave_embed) 
# logging.info('se_app_behave_embedding')
# wv_behave_sum = np.array(list(se_app_behave_embedding.values))
# x_mean = np.mean(wv_behave_sum, axis = 0)
# x_std = np.std(wv_behave_sum, axis = 0)
# wv_behave_sum_norm = (wv_behave_sum - x_mean) / x_std
# se_app_behave_embedding_norm = pd.Series(list(wv_behave_sum_norm))
# se_app_behave_embedding_norm.index = se_app_behave_embedding.index


# ## userlog

# In[ ]:


sentences = []
for data_list in tqdm(se_userlog_cross):
    sentence = [str(start_uselog_id),]
    for data in data_list:
        sentence.append(str(data))
    sentences.append(sentence)
    
window = 5
mincount = 1
n_dim = 100
logging.info('start word2vec install behave')
wv_model= Word2Vec(sentences, min_count=mincount, size = n_dim, window = window, sg = 0, workers = 12)
logging.info('finish word2vec install behave')
pickle.dump(wv_model, open('../data_sortout/wv_model_userlog_cross.pickle', 'wb'))


# ## load

# In[10]:


wv_model_app_list = pickle.load(open('../data_sortout/wv_model_app_list.pickle', 'rb'))
wv_model_app_behave = pickle.load(open('../data_sortout/wv_model_app_behave.pickle', 'rb'))
wv_model_userlog = pickle.load(open('../data_sortout/wv_model_userlog_cross.pickle', 'rb'))


# # train

# ## 调参

# In[11]:


from collections import namedtuple

ARG = namedtuple('ARG', [
    'batch_size',
    'epoch',
    'lr',
    'weight_decay',
    'debug',
    'n_embedding',
    'app_install_list_max_length',
    'app_behave_max_length',
    'userlog_max_length',
    'n_eval',
    'dropout_rate',
    'n_worker',
    'use_cuda',
    'n_gpu',
    'device',
    'card_list'
])
 
args = ARG(
    batch_size = 256,
    epoch = 12,
    lr = 0.001,
    weight_decay = 0.0,
    dropout_rate = 0.,
    debug = False,
    n_embedding = 100,
    app_install_list_max_length = 256,
    app_behave_max_length = 256,
    userlog_max_length = 256,
    n_eval = len(all_test_id)+1,
    n_worker = 0,
    use_cuda = True,
    n_gpu = 1,
    card_list = [0, 1],
    device=torch.device("cuda:1"),
#     device=torch.device("cpu")

)


# ## dataset

# In[12]:


install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))


# In[13]:


class AppDataset(Data.Dataset):
    def __init__(self, master_ids):
        self.master_ids = list(master_ids)
        
    def __len__(self):
        return len(self.master_ids)
    
    def __getitem__(self,idx):
        return self.master_ids[idx]

x_dict = {
    
    'user_info' : np.zeros((args.batch_size, df_user_one_hot.shape[1])),

    'app_list' : np.zeros((args.batch_size, args.app_install_list_max_length + 1)).astype('int'),
    'app_list_te_qcut' : np.zeros((args.batch_size, args.app_install_list_max_length + 1, 4)).astype('int'),
    'app_list_len' :  np.zeros((args.batch_size,)).astype('int'),
    
    'app_behave' : np.zeros((args.batch_size, args.app_behave_max_length + 1)).astype('int'),
    'app_behave_time_cut' : np.zeros((args.batch_size, args.app_behave_max_length + 1)).astype('int'),
    'app_behave_time_qcut' : np.zeros((args.batch_size, args.app_behave_max_length + 1)).astype('int'),
    
    'app_behave_action' : np.zeros((args.batch_size, args.app_behave_max_length + 1)).astype('int'),
    'app_behave_te_qcut' : np.zeros((args.batch_size, args.app_behave_max_length + 1, 4)).astype('int'),

    'app_behave_len' :  np.zeros((args.batch_size,)).astype('int'),
    
    'userlog' : np.zeros((args.batch_size, args.userlog_max_length + 1)).astype('int'),
    'userlog_len' :  np.zeros((args.batch_size,)).astype('int'),
    'userlog_day_qcut' : np.zeros((args.batch_size, args.userlog_max_length + 1)).astype('int'),
    'userlog_day_cut' : np.zeros((args.batch_size, args.userlog_max_length + 1)).astype('int'),
    'userlog_second_qcut' : np.zeros((args.batch_size, args.userlog_max_length + 1)).astype('int'),
    'userlog_second_cut' : np.zeros((args.batch_size, args.userlog_max_length + 1)).astype('int'),
    
    'view_mask' : np.zeros((args.batch_size, 4)).astype('int'),
}

def set_first_token():
    x_dict['app_list'][:, 0] = start_app_list_id
    x_dict['app_behave'][:, 0] = start_app_behave_id
    x_dict['userlog'][:, 0] = start_uselog_id

    x_dict['app_list'][:, 0] = 0
    x_dict['app_behave'][:, 0] = 0
    x_dict['userlog'][:, 0] = 0

    x_dict['userlog_day_qcut'][:, 0] = 8
    x_dict['userlog_day_cut'][:, 0] = 8
    x_dict['userlog_second_qcut'][:, 0] = 32
    x_dict['userlog_second_cut'][:, 0] = 32

set_first_token()

def collate_fn(master_ids):
    master_ids = np.array(master_ids)

#     sub_master_id = se_id_install_list.loc[master_ids]
#     df_sub_behave = df_install_behave.loc[master_ids]
#     df_sub_time = df_behave_time.loc[master_ids]
    for i, master_id in enumerate(master_ids):
        if master_id in user_info_set:
            x_dict['user_info'][i] = df_user_one_hot.loc[master_id].values
            x_dict['view_mask'][i][0] = 1
        else:
            x_dict['user_info'][i] = 0
            x_dict['view_mask'][i][0] = 0

        if master_id in install_list_set:
            app_list = se_id_install_list.at[master_id][:args.app_install_list_max_length]
            x_dict['app_list_len'][i] = len(app_list) + 1
            x_dict['app_list'][i][1 : x_dict['app_list_len'][i]] = app_list
            x_dict['app_list'][i][x_dict['app_list_len'][i] :] = 0
            
#             target_encode_data = np.array(list(df_app_list_te_qcut.loc[master_id].values))[:, :args.app_install_list_max_length].T
#             x_dict['app_list_te_qcut'][i][1 : x_dict['app_list_len'][i]] = target_encode_data
            
            x_dict['view_mask'][i][1] = 1
        else:
            x_dict['app_list_len'][i] = 1
            x_dict['app_list'][i][1:] = 0
            x_dict['app_list_te_qcut'][i] = 0
            x_dict['view_mask'][i][1] = 0

        if master_id in install_behave_set:
            app_behave = df_install_behave['pkg_id'].at[master_id][-args.app_behave_max_length:]
            len_app = len(app_behave) + 1
            x_dict['app_behave_len'][i] = len_app
            x_dict['app_behave'][i][1: len_app] = app_behave
            x_dict['app_behave'][i][len_app :] = 0
            
#             target_encode_data = np.array(list(df_app_behave_te_qcut.loc[master_id].values))[:, -args.app_behave_max_length:].T
#             x_dict['app_behave_te_qcut'][i][1 : len_app] = target_encode_data
            
            x_dict['app_behave_time_cut'][i][1:len_app] = df_behave_time['cut_id'].at[master_id][-args.app_behave_max_length:]
            x_dict['app_behave_time_qcut'][i][1:len_app] = df_behave_time['qcut_id'].at[master_id][-args.app_behave_max_length:]
            x_dict['app_behave_action'][i][1:len_app] = df_install_behave['action'].at[master_id][-args.app_behave_max_length:]
            x_dict['view_mask'][i][2] = 1
        else:
            x_dict['app_behave_len'][i] = 1
            x_dict['app_behave'][i][1:] = 0
            x_dict['app_behave_te_qcut'][i] = 0

            x_dict['app_behave_time_cut'][i][1:] = 0
            x_dict['app_behave_time_qcut'][i][1:] = 0
            x_dict['app_behave_action'][i][1:] = 0
            x_dict['view_mask'][i][2] = 0
        
        
        if master_id in user_log_set:
            userlog_list = se_userlog_cross.at[master_id][:args.userlog_max_length]
            len_userlog = len(userlog_list) + 1
            x_dict['userlog_len'][i] = len_userlog
            x_dict['userlog'][i][1 : len_userlog] = userlog_list
            x_dict['userlog'][i][len_userlog :] = 0
            x_dict['userlog_day_qcut'][i][1 : len_userlog] = df_userlog_time_seq['qcut_day_id'].at[master_id][:args.userlog_max_length]
            x_dict['userlog_day_qcut'][i][len_userlog :] = 0
            x_dict['userlog_day_cut'][i][1 : len_userlog] = df_userlog_time_seq['cut_day_id'].at[master_id][:args.userlog_max_length]
            x_dict['userlog_day_cut'][i][len_userlog :] = 0
            x_dict['userlog_second_qcut'][i][1 : len_userlog] = df_userlog_time_seq['qcut_second_id'].at[master_id][:args.userlog_max_length]
            x_dict['userlog_second_qcut'][i][len_userlog :] = 0
            x_dict['userlog_second_cut'][i][1 : len_userlog] = df_userlog_time_seq['cut_second_id'].at[master_id][:args.userlog_max_length]
            x_dict['userlog_second_cut'][i][len_userlog :] = 0

            x_dict['view_mask'][i][3] = 1
        else:
            x_dict['userlog_len'][i] = 1
            x_dict['userlog'][i] = 0
            x_dict['userlog_day_qcut'][i] = 0
            x_dict['userlog_day_cut'][i] = 0
            x_dict['userlog_second_qcut'][i] = 0
            x_dict['userlog_second_cut'][i] = 0

            x_dict['view_mask'][i][3] = 0
    
    
    len_id = master_ids.shape[0]
    x_dict['app_list'][len_id:] = 0
    x_dict['app_behave'][len_id:] = 0
    return {
        
        'user_info' : torch.tensor(x_dict['user_info'][:len_id]).float(),
        
        'app_list' : torch.tensor(x_dict['app_list'][:len_id]).long(),
        'app_list_te_qcut' : torch.tensor(x_dict['app_list_te_qcut'][:len_id]).long(),
        'app_list_len' : torch.tensor(x_dict['app_list_len'][:len_id]).long(),
        
        'app_behave' : torch.tensor(x_dict['app_behave'][:len_id]).long(),
        'app_behave_te_qcut' : torch.tensor(x_dict['app_behave_te_qcut'][:len_id]).long(),
        'app_behave_len' : torch.tensor(x_dict['app_behave_len'][:len_id]).long(),
        
        'app_behave_time_cut' : torch.tensor(x_dict['app_behave_time_cut'][:len_id]).long(),
        'app_behave_time_qcut' : torch.tensor(x_dict['app_behave_time_qcut'][:len_id]).long(),
        'app_behave_action' : torch.tensor(x_dict['app_behave_action'][:len_id]).long(),
        'userlog' : torch.tensor(x_dict['userlog'][:len_id]).long(),
        'userlog_len' : torch.tensor(x_dict['userlog_len'][:len_id]).long(),
        'userlog_day_qcut' : torch.tensor(x_dict['userlog_day_qcut'][:len_id]).long(),
        'userlog_day_cut' : torch.tensor(x_dict['userlog_day_cut'][:len_id]).long(),
        'userlog_second_qcut' : torch.tensor(x_dict['userlog_second_qcut'][:len_id]).long(),
        'userlog_second_cut' : torch.tensor(x_dict['userlog_second_cut'][:len_id]).long(),

        'view_mask' : torch.tensor(x_dict['view_mask'][:len_id]).long(),
        'labels1' : torch.tensor(df_target.loc[master_ids]['target_1m30+'].values).long(),
        'labels2' : torch.tensor(df_target.loc[master_ids]['target_2m30+'].values).long(),
        'labels3' : torch.tensor(df_target.loc[master_ids]['target_3m30+'].values).long(),
        'labels4' : torch.tensor(df_target.loc[master_ids]['target_4m30+'].values).long(),
    }


# ## model

# ### embed weight

# In[14]:


def get_app_list_wv_weight():
    weight = np.zeros((start_app_list_id+1, 100))
    for i in tqdm(range(1, start_app_list_id+1)):
        weight[i] = wv_model_app_list.wv[str(i)]
    weight_tensor = torch.tensor(weight).float()
    return weight_tensor

def get_app_behave_wv_weight():
    weight = np.zeros((start_app_behave_id+1, 100))
    for i in tqdm(range(1, start_app_behave_id+1)):
        weight[i] = wv_model_app_behave.wv[str(i)]
    weight_tensor = torch.tensor(weight).float()
    return weight_tensor

def get_userlog_wv_weight():
    weight = np.zeros((start_uselog_id+1, 100))
    for i in tqdm(range(1, start_uselog_id+1)):
        weight[i] = wv_model_userlog.wv[str(i)]
    weight_tensor = torch.tensor(weight).float()
    return weight_tensor

app_list_weight = get_app_list_wv_weight()
app_behave_weight = get_app_behave_wv_weight()
userlog_weight = get_userlog_wv_weight()


# ### sub model

# In[15]:


def masked_softmax(X, valid_len):
    if valid_len is None:
        return F.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask = (torch.arange(0,X.shape[-1]).repeat(X.shape[0],1).to(args.device) < valid_len).repeat(1, X.shape[1]).view(shape)
        
        X = X.masked_fill_(~mask, -float('inf'))
        return F.softmax(X,dim=-1).view(shape)

def make_mask(X, valid_len):
    if valid_len is None:
        return F.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])

        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).byte()
        return mask.unsqueeze(2) 

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `query`: (`batch_size`, #queries, `d`)
    # `key`: (`batch_size`, #kv_pairs, `d`)
    # `value`: (`batch_size`, #kv_pairs, `dim_v`)
    # `valid_len`: either (`batch_size`, ) or (`batch_size`, xx)
    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return torch.bmm(attention_weights, value)
    
class MLPAttention(nn.Module):
    def __init__(self, key_size, query_size, units, dropout=0., **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, units, bias=False)
        self.W_q = nn.Linear(query_size, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len):
        query, key = self.W_k(query), self.W_q(key)
        # Expand query to (`batch_size`, #queries, 1, units), and key to
        # (`batch_size`, 1, #kv_pairs, units). Then plus them with broadcast
        features = query.unsqueeze(2) + key.unsqueeze(1)
        scores = self.v(features).squeeze(-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return torch.bmm(attention_weights, value)

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,X):
        mean=X.mean(-1,keepdim=True)
        std=X.std(-1,keepdim=True)
        return self.gamma*(X-mean)/(std+self.eps)+self.beta
    
class MLPAttentionPool(nn.Module):
    def __init__(self,key_size,units):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(key_size,units,bias=False),
                                  nn.Tanh(),
                                  nn.Linear(units,1,bias=False))
        
    def masked_softmax_1d(self, X, valid_len):
        if valid_len is None:
            return F.softmax(X,dim=-1), _
        else:
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])

            mask=(torch.arange(0,X.shape[-1]).repeat(X.shape[0],1).to(X.device)<valid_len).byte()
            X = X.masked_fill_(~mask, -float('inf'))
            return F.softmax(X,dim=-1).view(shape), mask

    def forward(self, key, valid_len):
        scores = self.proj(key).squeeze(-1)
        attention_weights, mask = self.masked_softmax_1d(scores,valid_len)
        seq_out = attention_weights.unsqueeze(-1) * key
        return seq_out.sum(1)

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 128
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)

class UserNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        n_dim = df_user_one_hot.shape[1]
        self.dense_hidden = Dense(n_dim, config.hidden)

        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def forward(self, input_dict):
        
        x = input_dict['user_info'].to(args.device)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        hidden = self.dense_hidden(x)
        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden

class AppListNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = 64
        self.input_size = 100
        self.embeddings = nn.Embedding.from_pretrained(app_list_weight)
        self.layer_norm = LayerNorm(self.input_size)
        
#         self.attention_layer = DotProductAttention(0.)
#         self.attention_layer = MLPAttention(self.input_size, self.input_size, 256)

        self.attention_layer = MLPAttentionPool(self.input_size, config.hidden)
        self.dense_hidden = Dense(self.input_size, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def forward(self, input_dict):
        app_list_ids = input_dict['app_list'].to(args.device)
        app_list_len = input_dict['app_list_len'].to(args.device)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)

        app_list = self.embeddings(app_list_ids)
        app_list = self.layer_norm(app_list)

        x = self.attention_layer(app_list, app_list_len)
        
        hidden = self.dense_hidden(x)        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)

        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden

class AppBehaveNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden
        self.embeddings = nn.Embedding.from_pretrained(app_behave_weight)        
        for i in self.embeddings.parameters():
            i.requires_grad=False
        
        self.qcut_time_embeddings = nn.Embedding(64, 16)
        self.cut_time_embeddings = nn.Embedding(64, 16)
#         self.action_embeddings = nn.Embedding(2, 4)

        
        self.layer_norm = LayerNorm(100)
        self.rnn = nn.GRU(16 + 16 + 100,
                          hidden_size = config.hidden,
                          num_layers = 1,
                          dropout = 0,
                          bidirectional = False, 
                          batch_first=True)
        
#         self.attention_layer = DotProductAttention(0.)
#         self.attention_layer = MLPAttention(config.hidden, config.hidden, config.hidden)
        self.attention_layer = MLPAttentionPool(config.hidden, config.hidden)

        self.dense_hidden = Dense(config.hidden, config.hidden)

        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)

    def rnn_forward(self, x, x_lens):
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        hidden, _= self.rnn(X)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden,total_length=x.shape[1],batch_first=True)
        return hidden
   
    def forward(self, input_dict):
        
        app_behave_ids = input_dict['app_behave'].to(args.device)
        app_behave_len = input_dict['app_behave_len'].to(args.device)
        app_behave_time_cut = input_dict['app_behave_time_cut'].to(args.device)
        app_behave_time_qcut = input_dict['app_behave_time_qcut'].to(args.device)
#         app_behave_action = input_dict['app_behave_action'].to(args.device)
        
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        app_behave = self.embeddings(app_behave_ids)
#         app_behave = self.layer_norm(app_behave)
        cut_time_embed = self.cut_time_embeddings(app_behave_time_cut)
        qcut_time_embed = self.cut_time_embeddings(app_behave_time_qcut)
#         action_embed = self.action_embeddings(app_behave_action)
        
        seq_data = torch.cat([
            app_behave, 
            cut_time_embed,
            qcut_time_embed,
#             action_embed,
        ], dim = -1)
        
        
        rnn_out = self.rnn_forward(seq_data, app_behave_len)
        
        x = self.attention_layer(rnn_out, app_behave_len)
#         mask = make_mask(x, app_behave_len)
#         x = x.masked_fill_(~mask, 0).sum(1)

        
        hidden = self.dense_hidden(x)
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden

class UserlogNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden
        self.embeddings = nn.Embedding.from_pretrained(userlog_weight)        
#         for i in self.embeddings.parameters():
#             i.requires_grad=False
        self.embeddings_day_qcut = nn.Embedding(9, 8)
        self.embeddings_day_cut = nn.Embedding(9, 8)
        self.embeddings_second_qcut = nn.Embedding(33, 16)
        self.embeddings_second_cut = nn.Embedding(33, 16)

        
        self.layer_norm = LayerNorm(100)
        self.rnn = nn.GRU(100 + 8 + 16 + 8 + 16,
                          hidden_size = config.hidden,
                          num_layers = 1,
                          dropout = 0,
                          bidirectional = False, 
                          batch_first=True)
        
#         self.attention_layer = DotProductAttention(0.)
#         self.attention_layer = MLPAttention(config.hidden, config.hidden, config.hidden)
        self.attention_layer = MLPAttentionPool(config.hidden, config.hidden)

        self.dense_hidden = Dense(config.hidden, config.hidden)

        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)

    def rnn_forward(self, x, x_lens):
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        hidden, _= self.rnn(X)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden,total_length=x.shape[1],batch_first=True)
        return hidden
   
    def forward(self, input_dict):
        
        userlog_action_id = input_dict['userlog'].to(args.device)
        userlog_len = input_dict['userlog_len'].to(args.device)
        userlog_day_qcut_id = input_dict['userlog_day_qcut'].to(args.device)
        userlog_day_cut_id = input_dict['userlog_day_cut'].to(args.device)
        userlog_second_qcut_id = input_dict['userlog_second_qcut'].to(args.device)
        userlog_second_cut_id = input_dict['userlog_second_cut'].to(args.device)

        
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        userlog_action = self.embeddings(userlog_action_id)
#         app_behave = self.layer_norm(app_behave)
        userlog_day_qcut = self.embeddings_day_qcut(userlog_day_qcut_id)
        userlog_day_cut = self.embeddings_day_cut(userlog_day_cut_id)
        userlog_second_qcut = self.embeddings_second_qcut(userlog_second_qcut_id)
        userlog_second_cut = self.embeddings_second_cut(userlog_second_cut_id)
        hidden = torch.cat([
            userlog_action,
            userlog_day_qcut,
            userlog_day_cut,
            userlog_second_qcut,
            userlog_second_cut], dim = -1)
        
    
        rnn_out = self.rnn_forward(hidden, userlog_len)
        
        x = self.attention_layer(rnn_out, userlog_len)
#         mask = make_mask(x, app_behave_len)
#         x = x.masked_fill_(~mask, 0).sum(1)

        
        hidden = self.dense_hidden(x)
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
         
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden


# ### interactive model

# In[16]:


class Alignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(config.hidden)))
        self.summary = {}

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature
    
    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b

class MappedAlignment(Alignment):
    def __init__(self, config):
        super().__init__(config)
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            Dense(config.hidden, config.hidden),
        )

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)

class FullFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = args.dropout_rate
        self.fusion1 = Dense(config.hidden * 2, config.hidden)
        self.fusion2 = Dense(config.hidden * 2, config.hidden)
        self.fusion3 = Dense(config.hidden * 2, config.hidden)
        self.fusion = Dense(config.hidden * 3, config.hidden)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.dropout(x, self.dropout, self.training)
        return self.fusion(x)

class AppConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.applist_embeddings = nn.Embedding.from_pretrained(app_list_weight)
        self.appbehave_embeddings = nn.Embedding.from_pretrained(app_behave_weight)
        for i in self.applist_embeddings.parameters():
            i.requires_grad=False
        for i in self.appbehave_embeddings.parameters():
            i.requires_grad=False
        

        self.qcut_time_embeddings = nn.Embedding(64, 16)
        self.cut_time_embeddings = nn.Embedding(64, 16)
        self.action_embeddings = nn.Embedding(2, 4)
        
        self.app_list_qcut_embed_list = nn.ModuleList([nn.Embedding(17, 16) for _ in range(4)])
        self.app_behave_qcut_embed_list = nn.ModuleList([nn.Embedding(17, 16) for _ in range(4)])

        self.app_list_encoder = Dense(100, config.hidden)
#         self.app_list_encoder = Dense(16 * 4, config.hidden)
        self.attention_layer1 = MLPAttentionPool(config.hidden, config.hidden)
        self.attention_layer2 = MLPAttentionPool(config.hidden, config.hidden)


        self.rnn = nn.GRU(16 + 16 + 100,
#         self.rnn = nn.GRU(16 * 4,
                          hidden_size = config.hidden,
                          num_layers = 1,
                          dropout = 0,
                          bidirectional = False, 
                          batch_first=True)        
        
        
        self.dense = Dense(config.hidden * 2, config.hidden)
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def rnn_forward(self, x, x_lens):
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        hidden, _= self.rnn(X)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden,total_length=x.shape[1],batch_first=True)
        return hidden
    
    def maxpool(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]

    def app_behave_encode(self, input_dict):
        app_behave_ids = input_dict['app_behave'].to(args.device)
        app_behave_len = input_dict['app_behave_len'].to(args.device)
        app_behave_time_cut = input_dict['app_behave_time_cut'].to(args.device)
        app_behave_time_qcut = input_dict['app_behave_time_qcut'].to(args.device)
        app_behave_action = input_dict['app_behave_action'].to(args.device)
   
#         app_behave_target_encode = input_dict['app_behave_target_encode'].to(args.device)
        
        app_behave_target_qcut = input_dict['app_behave_te_qcut'].to(args.device)

#         app_behave_target_qcut_embed1 = self.app_behave_qcut_embed_list[0](app_behave_target_qcut[:, :, 0])
#         app_behave_target_qcut_embed2 = self.app_behave_qcut_embed_list[1](app_behave_target_qcut[:, :, 1])
#         app_behave_target_qcut_embed3 = self.app_behave_qcut_embed_list[2](app_behave_target_qcut[:, :, 2])
#         app_behave_target_qcut_embed4 = self.app_behave_qcut_embed_list[3](app_behave_target_qcut[:, :, 3])

        app_behave = self.appbehave_embeddings(app_behave_ids)
        cut_time_embed = self.cut_time_embeddings(app_behave_time_cut)
        qcut_time_embed = self.cut_time_embeddings(app_behave_time_qcut)
#         action_embed = self.action_embeddings(app_behave_action)
        
        seq_data = torch.cat([
            app_behave, 
            cut_time_embed,
            qcut_time_embed,
#             app_behave_target_qcut_embed1,
#             app_behave_target_qcut_embed2,
#             app_behave_target_qcut_embed3,
#             app_behave_target_qcut_embed4,
#             app_behave_target_encode,
#             action_embed,
        ], dim = -1)
        
        rnn_out = self.rnn_forward(seq_data, app_behave_len)

        return rnn_out, app_behave_len
    
    def app_list_encode(self, input_dict):
        app_list_ids = input_dict['app_list'].to(args.device)
        app_list_len = input_dict['app_list_len'].to(args.device)
        app_list_embed = self.applist_embeddings(app_list_ids)
        
        app_list_target_qcut = input_dict['app_list_te_qcut'].to(args.device)
#         app_list_target_encode = input_dict['app_list_target_encode'].to(args.device)

        app_list_target_qcut_embed1 = self.app_list_qcut_embed_list[0](app_list_target_qcut[:, :, 0])
        app_list_target_qcut_embed2 = self.app_list_qcut_embed_list[1](app_list_target_qcut[:, :, 1])
        app_list_target_qcut_embed3 = self.app_list_qcut_embed_list[2](app_list_target_qcut[:, :, 2])
        app_list_target_qcut_embed4 = self.app_list_qcut_embed_list[3](app_list_target_qcut[:, :, 3])

        seq_data = torch.cat([
            app_list_embed, 
#             app_list_target_encode,
#             app_list_target_qcut_embed1,
#             app_list_target_qcut_embed2,
#             app_list_target_qcut_embed3,
#             app_list_target_qcut_embed4,

        ], dim = -1)
        
        return self.app_list_encoder(seq_data), app_list_len 
    

    def forward(self, input_dict):
        a, a_len = self.app_behave_encode(input_dict)
        b, b_len = self.app_list_encode(input_dict)
#         mask_a = self.make_mask(a, a_len)
#         mask_b = self.make_mask(b, b_len)
                
        a = self.attention_layer1(a, a_len)
        b = self.attention_layer2(b, b_len)
        
        hidden = self.dense(torch.cat([a, b], dim=-1)) 

        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)

        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden
    
class AppInteractiveNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.applist_embeddings = nn.Embedding.from_pretrained(app_list_weight)
        self.appbehave_embeddings = nn.Embedding.from_pretrained(app_behave_weight)
        for i in self.applist_embeddings.parameters():
            i.requires_grad=False
        for i in self.appbehave_embeddings.parameters():
            i.requires_grad=False
        

        self.qcut_time_embeddings = nn.Embedding(64, 16)
        self.cut_time_embeddings = nn.Embedding(64, 16)
        self.action_embeddings = nn.Embedding(2, 4)
        
        self.app_list_qcut_embed_list = nn.ModuleList([nn.Embedding(17, 16) for _ in range(4)])
        self.app_behave_qcut_embed_list = nn.ModuleList([nn.Embedding(17, 16) for _ in range(4)])

        self.app_list_encoder = Dense(100, config.hidden)
#         self.app_list_encoder = Dense(16 * 4, config.hidden)


        self.rnn = nn.GRU(16 + 16 + 100,
#         self.rnn = nn.GRU(16 * 4,
                          hidden_size = config.hidden,
                          num_layers = 1,
                          dropout = 0,
                          bidirectional = False, 
                          batch_first=True)
        
        self.alignment_layer = MappedAlignment(config)
        self.fusion_layer = FullFusion(config)
        
        
        
        self.dense = Dense(config.hidden * 4, config.hidden)
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def rnn_forward(self, x, x_lens):
        X = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        hidden, _= self.rnn(X)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden,total_length=x.shape[1],batch_first=True)
        return hidden
    
    def maxpool(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]

    def app_behave_encode(self, input_dict):
        app_behave_ids = input_dict['app_behave'].to(args.device)
        app_behave_len = input_dict['app_behave_len'].to(args.device)
        app_behave_time_cut = input_dict['app_behave_time_cut'].to(args.device)
        app_behave_time_qcut = input_dict['app_behave_time_qcut'].to(args.device)
        app_behave_action = input_dict['app_behave_action'].to(args.device)
   
#         app_behave_target_encode = input_dict['app_behave_target_encode'].to(args.device)
        
#         app_behave_target_qcut = input_dict['app_behave_te_qcut'].to(args.device)

#         app_behave_target_qcut_embed1 = self.app_behave_qcut_embed_list[0](app_behave_target_qcut[:, :, 0])
#         app_behave_target_qcut_embed2 = self.app_behave_qcut_embed_list[1](app_behave_target_qcut[:, :, 1])
#         app_behave_target_qcut_embed3 = self.app_behave_qcut_embed_list[2](app_behave_target_qcut[:, :, 2])
#         app_behave_target_qcut_embed4 = self.app_behave_qcut_embed_list[3](app_behave_target_qcut[:, :, 3])

        app_behave = self.appbehave_embeddings(app_behave_ids)
        cut_time_embed = self.cut_time_embeddings(app_behave_time_cut)
        qcut_time_embed = self.cut_time_embeddings(app_behave_time_qcut)
#         action_embed = self.action_embeddings(app_behave_action)
        
        seq_data = torch.cat([
            app_behave, 
            cut_time_embed,
            qcut_time_embed,
#             app_behave_target_qcut_embed1,
#             app_behave_target_qcut_embed2,
#             app_behave_target_qcut_embed3,
#             app_behave_target_qcut_embed4,
#             app_behave_target_encode,
#             action_embed,
        ], dim = -1)
        
        rnn_out = self.rnn_forward(seq_data, app_behave_len)

        return rnn_out, app_behave_len
    
    def app_list_encode(self, input_dict):
        app_list_ids = input_dict['app_list'].to(args.device)
        app_list_len = input_dict['app_list_len'].to(args.device)
        app_list_embed = self.applist_embeddings(app_list_ids)
        
        app_list_target_qcut = input_dict['app_list_te_qcut'].to(args.device)
#         app_list_target_encode = input_dict['app_list_target_encode'].to(args.device)

        app_list_target_qcut_embed1 = self.app_list_qcut_embed_list[0](app_list_target_qcut[:, :, 0])
        app_list_target_qcut_embed2 = self.app_list_qcut_embed_list[1](app_list_target_qcut[:, :, 1])
        app_list_target_qcut_embed3 = self.app_list_qcut_embed_list[2](app_list_target_qcut[:, :, 2])
        app_list_target_qcut_embed4 = self.app_list_qcut_embed_list[3](app_list_target_qcut[:, :, 3])

        seq_data = torch.cat([
            app_list_embed, 
#             app_list_target_encode,
#             app_list_target_qcut_embed1,
#             app_list_target_qcut_embed2,
#             app_list_target_qcut_embed3,
#             app_list_target_qcut_embed4,

        ], dim = -1)
        
        return self.app_list_encoder(seq_data), app_list_len 
    
    def make_mask(self, X, valid_len):
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
        return mask.unsqueeze(2).byte()

    def forward(self, input_dict):
        a, a_len = self.app_behave_encode(input_dict)
        b, b_len = self.app_list_encode(input_dict)
        mask_a = self.make_mask(a, a_len)
        mask_b = self.make_mask(b, b_len)
        
        align_a, align_b = self.alignment_layer(a, b, mask_a, mask_b)
        a = self.fusion_layer(a, align_a)
        b = self.fusion_layer(b, align_b)
        
        a = self.maxpool(a, mask_a)
        b = self.maxpool(b, mask_b)
        
        hidden = self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1)) #symmetric


        
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)

        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())
        
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss, y1, y2, y3, y4, hidden


# ### multi-view model

# In[17]:


class MaskMlpAttention(nn.Module):
    def __init__(self, key_size, units):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(key_size,units,bias=False),
                                  nn.Tanh(),
                                  nn.Linear(units,1,bias=False))
        
    def forward(self, key, mask):
        scores = self.proj(key).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill_(~mask.byte(), -float('inf'))
            
        softmax_score = F.softmax(scores, dim=-1)
        seq_out = softmax_score.unsqueeze(-1) * key
        return seq_out.sum(1)

class MaskMultiViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_network = config.user_net
        self.app_list_network = config.app_list_net
        self.app_behave_network = config.app_behave_net
        self.userlog_network = config.userlog_network
        
    def forward(self,input_dict):
        loss_user, y1_user, y2_user, y3_user, y4_user, hidden_user = self.user_network(input_dict)
        loss_list, y1_list, y2_list, y3_list, y4_list, hidden_list = self.app_list_network(input_dict)
        loss_behave, y1_behave, y2_behave, y3_behave, y4_behave, hidden_behave = self.app_behave_network(input_dict)
        loss_userlog, y1_userlog, y2_userlog, y3_userlog, y4_userlog, hidden_userlog = self.userlog_network(input_dict)

        return hidden_user, hidden_list, hidden_behave, hidden_userlog
    
class MaskMultiViewNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoder(config)

        self.attention = MaskMlpAttention(config.hidden, config.hidden)
#         self.attention = MaskMlpAttentionSeperateMap(config.hidden, config.hidden, 3)

        self.dense = Dense(config.hidden, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def forward(self, input_dict):
        
        hidden_user, hidden_list, hidden_behave, hidden_userlog = self.multiview_encoder(input_dict)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        view_mask = input_dict['view_mask'].to(args.device)
        
        multi_view_hidden = torch.cat([hidden_user, hidden_list, hidden_behave, hidden_userlog], dim = 1).view(-1, 4, hidden_user.shape[1])
        hidden = self.attention(multi_view_hidden, view_mask)

        #         hidden = self.attention([hidden_user, hidden_list, hidden_behave], view_mask)
#         hidden = torch.cat([hidden_user, hidden_list, hidden_behave], dim = 1)
        hidden = self.dense(hidden)

        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4

        return loss, y1, y2, y3, y4, _

class MaskMultiViewNetworkFillZero(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoder(config)

        self.dense = Dense(config.hidden * 4, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def forward(self, input_dict):
        
        hidden_user, hidden_list, hidden_behave, hidden_userlog = self.multiview_encoder(input_dict)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        view_mask = input_dict['view_mask'].to(args.device)
        
        hidden_user = view_mask[:, 0:1].float() * hidden_user
        hidden_list = view_mask[:, 1:2].float() * hidden_list
        hidden_behave = view_mask[:, 2:3].float() * hidden_behave
        hidden_userlog = view_mask[:, 3:4].float() * hidden_userlog

        multi_view_hidden = torch.cat([hidden_user, hidden_list, hidden_behave, hidden_userlog], dim = 1)

        #         hidden = self.attention([hidden_user, hidden_list, hidden_behave], view_mask)
#         hidden = torch.cat([hidden_user, hidden_list, hidden_behave], dim = 1)
        hidden = self.dense(multi_view_hidden)

        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4

        return loss, y1, y2, y3, y4, _

class MaskMultiViewNetworkFillZeroIn(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoderAppInteractive(config)

    
        self.dense = Dense(3 * config.hidden, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def get_3view_mask(self, mask):
        ret = torch.zeros((mask.shape[0], 3))
        ret[:, 0] = mask[:, 0]
        ret[:, 1] = (mask[:, 1] + mask[:, 2]) >= 1
        ret[:, -1] = mask[:, -1]
        return ret

    
    def forward(self, input_dict):
        
        hidden_user, hidden_app, hidden_userlog = self.multiview_encoder(input_dict)
        
        view_mask = self.get_3view_mask(input_dict['view_mask']).to(args.device)
        hidden_user = view_mask[:, 0:1].float() * hidden_user
        hidden_app = view_mask[:, 1:2].float() * hidden_app
        hidden_userlog = view_mask[:, 2:3].float() * hidden_userlog

        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        
        multi_view_hidden = torch.cat([hidden_user, hidden_app, hidden_userlog], dim = 1)

        
        hidden = self.dense(multi_view_hidden)
        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4

        return loss, y1, y2, y3, y4, _

class MaskMultiViewNetworkGenerate(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoder(config)

        self.attention_pool1 = MaskMlpAttention(config.hidden, config.hidden)
#         self.attention = MaskMlpAttentionSeperateMap(config.hidden, config.hidden, 3)
        self.attention_pool2 = MaskMlpAttention(config.hidden, config.hidden)

    
        self.decoder_user_info = Dense(config.hidden, config.hidden)
        self.decoder_app_list = Dense(config.hidden, config.hidden)
        self.decoder_app_behave = Dense(config.hidden, config.hidden)
        self.decoder_user_log = Dense(config.hidden, config.hidden)

        self.dense = Dense(config.hidden, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
        
    def forward(self, input_dict):
        
        hidden_user, hidden_list, hidden_behave, hidden_userlog = self.multiview_encoder(input_dict)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        view_mask = input_dict['view_mask'].to(args.device)
        
        multi_view_hidden = torch.stack([hidden_user, hidden_list, hidden_behave, hidden_userlog], dim = 1)

        hidden = self.attention_pool1(multi_view_hidden, view_mask)

        #         hidden = self.attention([hidden_user, hidden_list, hidden_behave], view_mask)
#         hidden = torch.cat([hidden_user, hidden_list, hidden_behave], dim = 1)

        generate_user = self.decoder_user_info(hidden)
        generate_list = self.decoder_app_list(hidden)
        generate_behave = self.decoder_app_behave(hidden)
        generate_userlog = self.decoder_user_log(hidden)
        
        
        loss_rebuild_user = (hidden_user - generate_user) ** 2 * view_mask[:, 0:1].float()
        loss_rebuild_list = (hidden_list - generate_list) ** 2 * view_mask[:, 1:2].float()
        loss_rebuild_behave = (hidden_behave - generate_behave) ** 2 * view_mask[:, 2:3].float()
        loss_rebuild_userlog = (hidden_userlog - generate_userlog) ** 2 * view_mask[:, 3:4].float()
        loss_rebuild = loss_rebuild_user.mean() + loss_rebuild_list.mean() + loss_rebuild_behave.mean() + loss_rebuild_userlog.mean()
        
        multi_view_hidden_generate = torch.stack([generate_user, generate_list, generate_behave, generate_userlog], dim = 1)
        hidden_generate = self.attention_pool2(multi_view_hidden_generate, None)
        hidden = self.dense(hidden_generate)

        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4 + loss_rebuild

        return loss, y1, y2, y3, y4, _

class MaskMultiViewNetworkGenerateWithResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoder(config)

        self.attention_pool1 = MaskMlpAttention(config.hidden, config.hidden)
#         self.attention = MaskMlpAttentionSeperateMap(config.hidden, config.hidden, 3)
        self.attention_pool2 = MaskMlpAttention(config.hidden, config.hidden)

    
        self.decoder_user_info = Dense(config.hidden, config.hidden)
        self.decoder_app_list = Dense(config.hidden, config.hidden)
        self.decoder_app_behave = Dense(config.hidden, config.hidden)
        self.decoder_user_log = Dense(config.hidden, config.hidden)

        self.dense = Dense(config.hidden, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
                
    def forward(self, input_dict):
        
        hidden_user, hidden_list, hidden_behave, hidden_userlog = self.multiview_encoder(input_dict)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        view_mask = input_dict['view_mask'].to(args.device)
        
        multi_view_hidden = torch.stack([hidden_user, hidden_list, hidden_behave, hidden_userlog], dim = 1)

        hidden = self.attention_pool1(multi_view_hidden, view_mask)

        #         hidden = self.attention([hidden_user, hidden_list, hidden_behave], view_mask)
#         hidden = torch.cat([hidden_user, hidden_list, hidden_behave], dim = 1)

        generate_user = self.decoder_user_info(hidden)
        generate_list = self.decoder_app_list(hidden)
        generate_behave = self.decoder_app_behave(hidden)
        generate_userlog = self.decoder_user_log(hidden)
        
        
        loss_rebuild_user = (hidden_user - generate_user) ** 2 * view_mask[:, 0:1].float()
        loss_rebuild_list = (hidden_list - generate_list) ** 2 * view_mask[:, 1:2].float()
        loss_rebuild_behave = (hidden_behave - generate_behave) ** 2 * view_mask[:, 2:3].float()
        loss_rebuild_userlog = (hidden_userlog - generate_userlog) ** 2 * view_mask[:, 3:4].float()
        loss_rebuild = loss_rebuild_user.mean() + loss_rebuild_list.mean() + loss_rebuild_behave.mean() + loss_rebuild_userlog.mean()
        
        multi_view_hidden_generate = torch.stack([generate_user, generate_list, generate_behave, generate_userlog], dim = 1)
        hidden_generate = self.attention_pool2(multi_view_hidden_generate, None)
        
        hidden = hidden + hidden_generate
        hidden = self.dense(hidden)
        
        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4 + loss_rebuild

        return loss, y1, y2, y3, y4, _

class MaskMultiViewEncoderAppInteractive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_network = config.user_net
        self.app_network = config.app_net
        self.userlog_network = config.userlog_network
        
    def forward(self,input_dict):
        loss_user, y1_user, y2_user, y3_user, y4_user, hidden_user = self.user_network(input_dict)
        loss_app, y1_app, y2_app, y3_app, y4_app, hidden_app = self.app_network(input_dict)
        loss_userlog, y1_userlog, y2_userlog, y3_userlog, y4_userlog, hidden_userlog = self.userlog_network(input_dict)

        return hidden_user, hidden_app, hidden_userlog

class MaskMultiViewNetworkGenerateWithResidualAppInteractive(nn.Module):
    def __init__(self, config):
        super().__init__()
        
#         self.user_network = UserNetwork(config)
#         self.app_list_network = AppListNetwork(config)
#         self.app_behave_network = AppBehaveNetwork(config)
        
        self.multiview_encoder = MaskMultiViewEncoderAppInteractive(config)

        self.attention_pool1 = MaskMlpAttention(config.hidden, config.hidden)
#         self.attention = MaskMlpAttentionSeperateMap(config.hidden, config.hidden, 3)
        self.attention_pool2 = MaskMlpAttention(config.hidden, config.hidden)
    
        self.decoder_user_info = Dense(config.hidden, config.hidden)
        self.decoder_app = Dense(config.hidden, config.hidden)
        self.decoder_user_log = Dense(config.hidden, config.hidden)

        self.dense = Dense(config.hidden, config.hidden)
        
        self.dense1 = Dense(config.hidden, 2)
        self.dense2 = Dense(config.hidden, 2)
        self.dense3 = Dense(config.hidden, 2)
        self.dense4 = Dense(config.hidden, 2)
    
    def get_3view_mask(self, mask):
        ret = torch.zeros((mask.shape[0], 3))
        ret[:, 0] = mask[:, 0]
        ret[:, 1] = (mask[:, 1] + mask[:, 2]) >= 1
        ret[:, -1] = mask[:, -1]
        return ret
    
    def forward(self, input_dict):
        
        hidden_user, hidden_app, hidden_userlog = self.multiview_encoder(input_dict)
        labels1 = input_dict['labels1'].to(args.device)
        labels2 = input_dict['labels2'].to(args.device)
        labels3 = input_dict['labels3'].to(args.device)
        labels4 = input_dict['labels4'].to(args.device)
        view_mask = self.get_3view_mask(input_dict['view_mask']).to(args.device)
        
        multi_view_hidden = torch.stack([hidden_user, hidden_app, hidden_userlog], dim = 1)

        hidden = self.attention_pool1(multi_view_hidden, view_mask)

        #         hidden = self.attention([hidden_user, hidden_list, hidden_behave], view_mask)
#         hidden = torch.cat([hidden_user, hidden_list, hidden_behave], dim = 1)

        generate_user = self.decoder_user_info(hidden)
        generate_app = self.decoder_app(hidden)
        generate_userlog = self.decoder_user_log(hidden)
        
        
        loss_rebuild_user = (hidden_user - generate_user) ** 2 * view_mask[:, 0:1].float()
        loss_rebuild_app = (hidden_app - generate_app) ** 2 * view_mask[:, 1:2].float()
        loss_rebuild_userlog = (hidden_userlog - generate_userlog) ** 2 * view_mask[:, 2:3].float()
        loss_rebuild = loss_rebuild_user.mean() + loss_rebuild_app.mean()  + loss_rebuild_userlog.mean()
        
        multi_view_hidden_generate = torch.stack([generate_user, generate_app, generate_userlog], dim = 1)
        hidden_generate = self.attention_pool2(multi_view_hidden_generate, None)
        
        hidden = hidden + hidden_generate
        hidden = self.dense(hidden)

        y1 = self.dense1(hidden)
        y2 = self.dense2(hidden)
        y3 = self.dense3(hidden)
        y4 = self.dense4(hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, labels1.long())
        loss2 = loss_func(y2, labels2.long())
        loss3 = loss_func(y3, labels3.long())
        loss4 = loss_func(y4, labels4.long())

        loss = loss1 + loss2 + loss3 + loss4 + loss_rebuild

        return loss, y1, y2, y3, y4, hidden


# ## trainer

# In[18]:


from sklearn.metrics import f1_score

def best_f1(pre_score, label):
        
    max_score = 0
    r = 0.
    while r < 0.5:
        r += 0.005
        y_pre = pre_score > r
        score = f1_score(y_pre, label)
        max_score = max(max_score, score)
    return max_score

best_f1(np.array([0.02,0.2,0.01]), np.array([0,1,1]))


# In[19]:


data = np.array(
[
    [1,2],
    [3,99]
]
)
softmax(data, 1)


# In[25]:


from sklearn.metrics import f1_score


def eval_data(model, master_ids):
    
    if(len(master_ids) > args.n_eval):
        master_ids = random.sample(master_ids, args.n_eval)

    torch_dataset = AppDataset(master_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
#     for step, data in enumerate(tqdm(data_loader)):
    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader)):
            loss, y1, y2, y3, y4, _ = model(data)

            loss_list.append(loss.item())
            y1_list.append(y1.cpu().detach().numpy())
            y2_list.append(y2.cpu().detach().numpy())
            y3_list.append(y3.cpu().detach().numpy())
            y4_list.append(y4.cpu().detach().numpy())

            label1_list.append(data['labels1'].cpu().detach().numpy())
            label2_list.append(data['labels2'].cpu().detach().numpy())
            label3_list.append(data['labels3'].cpu().detach().numpy())
            label4_list.append(data['labels4'].cpu().detach().numpy())

        loss = np.mean(loss_list)
    
    y1_np = np.concatenate(y1_list,  axis = 0)
    y2_np = np.concatenate(y2_list,  axis = 0)
    y3_np = np.concatenate(y3_list,  axis = 0)
    y4_np = np.concatenate(y4_list,  axis = 0)

#     y1_np = softmax(y1_np, 1)
#     y2_np = softmax(y2_np, 1)
#     y3_np = softmax(y3_np, 1)
#     y4_np = softmax(y4_np, 1)

    labels1_np = np.concatenate(label1_list,  axis = 0)
    labels2_np = np.concatenate(label2_list,  axis = 0)
    labels3_np = np.concatenate(label3_list,  axis = 0)
    labels4_np = np.concatenate(label4_list,  axis = 0)

    auc1 = roc_auc_score(labels1_np, y1_np[:, 1])
    auc2 = roc_auc_score(labels2_np, y2_np[:, 1])
    auc3 = roc_auc_score(labels3_np, y3_np[:, 1])
    auc4 = roc_auc_score(labels4_np, y4_np[:, 1])
    auc_all = [auc1, auc2, auc3, auc4]
    
    new_client = np.array(df_master_records['new_client'].loc[master_ids].values)
    new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

    old_client = ~new_client
    old_client_auc1 = roc_auc_score(labels1_np[old_client], y1_np[:, 1][old_client])
    old_client_auc2 = roc_auc_score(labels2_np[old_client], y2_np[:, 1][old_client])
    old_client_auc3 = roc_auc_score(labels3_np[old_client], y3_np[:, 1][old_client])
    old_client_auc4 = roc_auc_score(labels4_np[old_client], y4_np[:, 1][old_client])
    auc_old_client = [old_client_auc1, old_client_auc2, old_client_auc3, old_client_auc4]

    f11 = best_f1(labels1_np, y1_np[:, 1])
    f12 = best_f1(labels2_np, y2_np[:, 1])
    f13 = best_f1(labels3_np, y3_np[:, 1])
    f14 = best_f1(labels4_np, y4_np[:, 1])
    f1_all = [f11, f12, f13, f14]
    
    new_client = np.array(df_master_records['new_client'].loc[master_ids].values)
    new_client_f1 = best_f1(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_f2 = best_f1(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_f3 = best_f1(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_f4 = best_f1(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_f1, new_client_f2, new_client_f3, new_client_f4]

    old_client = ~new_client
    old_client_f1 = best_f1(labels1_np[old_client], y1_np[:, 1][old_client])
    old_client_f2 = best_f1(labels2_np[old_client], y2_np[:, 1][old_client])
    old_client_f3 = best_f1(labels3_np[old_client], y3_np[:, 1][old_client])
    old_client_f4 = best_f1(labels4_np[old_client], y4_np[:, 1][old_client])
    auc_old_client = [old_client_f1, old_client_f2, old_client_f3, old_client_f4]

    
    return {
        'loss' : loss,
        '1m30+' : auc1,
        '2m30+' : auc2,
        '3m30+' : auc3,
        '4m30+' : auc4,
        'new_1m30+' : new_client_auc1,
        'new_2m30+' : new_client_auc2,
        'new_3m30+' : new_client_auc3,
        'new_4m30+' : new_client_auc4,
        'old_1m30+' : old_client_auc1,
        'old_2m30+' : old_client_auc2,
        'old_3m30+' : old_client_auc3,
        'old_4m30+' : old_client_auc4
    },{
        'f1_1m30+' : f11,
        'f1_2m30+' : f12,
        'f1_3m30+' : f13,
        'f1_4m30+' : f14,
        'f1_new_1m30+' : new_client_f1,
        'f1_new_2m30+' : new_client_f2,
        'f1_new_3m30+' : new_client_f3,
        'f1_new_4m30+' : new_client_f4,
        'f1_old_1m30+' : old_client_f1,
        'f1_old_2m30+' : old_client_f2,
        'f1_old_3m30+' : old_client_f3,
        'f1_old_4m30+' : old_client_f4,

    }

def train(train_ids, test_ids, model_class, config):
        
    torch_dataset = AppDataset(train_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=True,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    model = model_class(config).to(args.device)
#     model = UserNetwork().to(args.device)
#     model = AppBehaveNetwork().to(args.device)
#     model = AppListNetwork().to(args.device)
#     model = MaskMultiViewNetwork(config).to(args.device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

    decay = ["app_network"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in decay)], "weight_decay": args.weight_decay},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(train_ids)//(args.batch_size)), num_training_steps=int(len(train_ids) / args.batch_size * args.epoch)
    )

    for epoch in range(args.epoch):
        model.train()
        for step, data in enumerate(tqdm(data_loader)):
            loss, y1, y2, y3, y4, _ = model(data)            
            #backward
            S = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            optimizer.step()
            scheduler.step()
        
        model.eval()
        train_ret_dict = eval_data(model, train_ids)
        test_ret_dict = eval_data(model, test_ids)
        df_ret = pd.DataFrame([
            train_ret_dict, test_ret_dict
        ], index = ['train', 'test'])
        
        
#         install_behave_test = list(set(install_behave_set) & set(test_ids))
#         install_list_test = list(set(install_list_set) & set(test_ids))
#         user_info_test = list(set(user_info_set) & set(test_ids))
#         user_log_test = list(set(user_log_set) & set(test_ids))
        
#         behave_ret_dict = eval_data(model, install_behave_test)
#         list_ret_dict = eval_data(model, install_list_test)
#         userinfo_ret_dict = eval_data(model, user_info_test)
#         userlog_ret_dict = eval_data(model, user_log_test)

#         df_ret = pd.DataFrame([
#             test_ret_dict, behave_ret_dict, list_ret_dict, userinfo_ret_dict, userlog_ret_dict
#         ], index = ['test', 'behave', 'list', 'user_info', 'user_log'])
        

        logging.info('epoch : %d' % epoch)
        ipd.display(df_ret)
#         torch.save(model, 'model/multiview_net_extend.torch.model_%d' % epoch)

        
    return model


# ## config

# In[26]:


class UserNetConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256

class AppbehaveConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
        
class AppListConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
        
class Userlogconfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256

class MultiviewConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
#         self.app_list_net = torch.load('app_list_net.model.torch')
#         self.app_behave_net =  torch.load('app_list_net.model.torch')
#         self.user_net =  torch.load('user_net.model.torch')
        self.app_list_net = AppListNetwork(AppListConfig())
        self.app_behave_net = AppBehaveNetwork(AppbehaveConfig())
        self.user_net = UserNetwork(UserNetConfig())
        self.userlog_network = UserlogNetwork(Userlogconfig())

class MultiviewAppInteractiveConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
#         self.app_list_net = torch.load('app_list_net.model.torch')
#         self.app_behave_net =  torch.load('app_list_net.model.torch')
#         self.user_net =  torch.load('user_net.model.torch')
        self.app_net = AppInteractiveNetwork(AppListConfig())
        self.user_net = UserNetwork(UserNetConfig())
        self.userlog_network = UserlogNetwork(Userlogconfig())


# ## model-train

# ### user attribute

# In[27]:


logging.info('start training')
        
user_net_config = UserNetConfig()

user_net = train(all_train_id, all_test_id, UserNetwork, user_net_config)
torch.save(user_net, 'user_net.model.torch')
logging.info('finish training')


# In[22]:


user_net = torch.load('user_net.model.torch')


# In[30]:


pd.Series([1,2,3,3,2]).value_counts(1)[1]


# In[31]:


def best_f1(label, pre_score):
    
    ipd.display(pd.DataFrame(np.array(pre_score).reshape(-1, 1)).describe())
    max_score = 0
    select_score = random.sample(list(pre_score), 10000)

    
    defalt_rate = pd.Series(label).value_counts(1)[0]
    r = sorted(pre_score)[int(len(pre_score)*defalt_rate)]
    y_pre = pre_score > r
    score = f1_score(y_pre, label, average='micro')
    print(score)
    
    return score

auc_ret_dict, f1_ret_dict = eval_data(user_net, all_test_id)
df_ret = pd.DataFrame([
    f1_ret_dict
], index = ['test'])
ipd.display(df_ret)


# ### App behave

# In[28]:


app_behave_ids = list(df_install_behave.index)
train_app_behave_ids = list( set(app_behave_ids) & set(all_train_id) )
test_app_behave_ids = list( set(app_behave_ids) & set(all_test_id) )
len(train_app_behave_ids), len(test_app_behave_ids)


# In[29]:


logging.info('start training')

app_behave_net = train(train_app_behave_ids, test_app_behave_ids, AppBehaveNetwork, AppbehaveConfig())
torch.save(app_behave_net, 'app_behave_net.model.torch')
logging.info('finish training')


# ### App list

# In[30]:


app_list_ids = se_id_install_list.index
train_app_list_ids = list( set(app_list_ids) & set(all_train_id) )
test_app_list_ids = list( set(app_list_ids) & set(all_test_id) )
len(train_app_list_ids), len(test_app_list_ids)


# In[31]:


logging.info('start training')
app_list_net = train(train_app_list_ids, test_app_list_ids, AppListNetwork, AppListConfig())
torch.save(app_list_net, 'app_list_net.model.torch')
logging.info('finish training')


# ### App interation

# In[17]:


app_ids = set(se_id_install_list.index) | set(df_install_behave.index)
train_app_ids = list( set(app_ids) & set(all_train_id) )
test_app_ids = list( set(app_ids) & set(all_test_id) )
len(train_app_ids), len(test_app_ids)


# #### origin

# In[27]:


logging.info('start training')
app_interactive_net = train(train_app_ids, test_app_ids, AppInteractiveNetwork, AppListConfig())
torch.save(app_interactive_net, 'app_interactive_net.model.torch')
logging.info('finish training')


# #### embedding concat

# In[33]:


logging.info('start training')
app_interactive_net = train(train_app_ids, test_app_ids, AppConcat, AppListConfig())
# torch.save(app_interactive_net, 'app_interactive_net.model.torch')
logging.info('finish training')


# ### userlog

# In[32]:


userlog_ids = se_userlog_cross.index
train_userlog_ids = list( set(userlog_ids) & set(all_train_id) )
test_userlog_ids = list( set(userlog_ids) & set(all_test_id) )
len(train_userlog_ids), len(test_userlog_ids)


# In[33]:


logging.info('start training')
userlog_net = train(train_userlog_ids, test_userlog_ids, UserlogNetwork, Userlogconfig())
torch.save(userlog_net, 'userlog_net.model.torch')
logging.info('finish training')


# In[28]:


logging.info('start training')
userlog_net = train(train_userlog_ids, test_userlog_ids, UserlogNetwork, Userlogconfig())
torch.save(userlog_net, 'userlog_net.time.model.torch')
logging.info('finish training')


# ### multi-view origin

# In[18]:


logging.info('start training')
multiview_net = train(all_train_id, all_test_id, MaskMultiViewNetwork, MultiviewConfig())
logging.info('finish training')


# ### multi-view fill zeros

# In[19]:


logging.info('start training')
multiview_net_fill_zeros = train(all_train_id, all_test_id, MaskMultiViewNetworkFillZero, MultiviewConfig())
torch.save(multiview_net_fill_zeros, 'multiview_net.model.torch')

logging.info('finish training')


# In[18]:


logging.info('start training')
multiview_net_fill_zeros_in = train(all_train_id, all_test_id, MaskMultiViewNetworkFillZeroIn, MultiviewAppInteractiveConfig())
torch.save(multiview_net_fill_zeros_in, 'multiview_net_fill_zeros_in.model.torch')
logging.info('finish training')


# ### multi-view generate

# In[45]:


logging.info('start training')
multiview_net = train(all_train_id, all_test_id, MaskMultiViewNetworkGenerate, MultiviewConfig())
logging.info('finish training')


# ### multi-view generate with residual

# In[38]:


logging.info('start training')
multiview_net = train(all_train_id, all_test_id, MaskMultiViewNetworkGenerateWithResidual, MultiviewConfig())
logging.info('finish training')


# ### multi-view generate with residual App interactive

# In[23]:


logging.info('start training')
multiview_net = train(all_train_id, all_test_id, MaskMultiViewNetworkGenerateWithResidualAppInteractive, MultiviewAppInteractiveConfig())
logging.info('finish training') 


# In[22]:


#with user log time
logging.  info('start training')
multiview_net = train(all_train_id, all_test_id, MaskMultiViewNetworkGenerateWithResidualAppInteractive, MultiviewAppInteractiveConfig())
logging.info('finish training') 


# In[2]:


a = [1, 2, 3, 4, 5]
a[::-1]


# ### drop all miss data

# In[27]:


select_train_id = list(set(se_id_install_list.index) & set(df_install_behave.index) & set(se_userlog_cross.index) & set(all_train_id))
select_test_id = list(set(se_id_install_list.index) & set(df_install_behave.index) & set(se_userlog_cross.index) & set(all_test_id))
len(select_train_id), len(select_test_id)


# In[28]:


class MultiviewConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
#         self.app_list_net = torch.load('app_list_net.model.torch')
#         self.app_behave_net =  torch.load('app_list_net.model.torch')
#         self.user_net =  torch.load('user_net.model.torch')
        self.app_list_net = AppListNetwork(AppListConfig())
        self.app_behave_net = AppBehaveNetwork(AppbehaveConfig())
        self.user_net = UserNetwork(UserNetConfig())
        self.userlog_network = UserlogNetwork(Userlogconfig())

# torch.save(MultiviewConfig(), '../data_sortout/MultiviewConfig.torch')
logging.info('start training')
multiview_net = train(select_train_id, select_test_id, MaskMultiViewNetworkGenerateWithResidual, MultiviewConfig())
logging.info('finish training')


# ### handle missing view with less data

# In[29]:


def reduce_set_number(id_set):
    not_missing_rate = len(id_set) / df_master_records.shape[0]
    return set(random.sample(id_set, int(len(id_set) * not_missing_rate)))

install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))

install_behave_set = reduce_set_number(install_behave_set)
install_list_set = reduce_set_number(install_list_set)
user_info_set = reduce_set_number(user_info_set)
user_log_set = reduce_set_number(user_log_set)

logging.info('start training')
multiview_net = train(select_train_id, select_test_id, MaskMultiViewNetworkGenerateWithResidual, MultiviewConfig())
logging.info('finish training')

install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))


# In[34]:


start = time.time() 


# In[35]:


time.time()  - start


# ### multi-view mean

# In[34]:


user_net = torch.load('user_net.model.torch')
app_behave_net = torch.load('app_behave_net.model.torch')
app_list_net = torch.load('app_list_net.model.torch')
userlog_net = torch.load('userlog_net.model.torch')


# In[35]:


train_user_ids, test_user_ids = all_train_id, all_test_id

app_behave_ids = list(df_install_behave.index)
train_app_behave_ids = list( set(app_behave_ids) & set(all_train_id) )
test_app_behave_ids = list( set(app_behave_ids) & set(all_test_id) )

app_list_ids = se_id_install_list.index
train_app_list_ids = list( set(app_list_ids) & set(all_train_id) )
test_app_list_ids = list( set(app_list_ids) & set(all_test_id) )

userlog_ids = se_userlog_cross.index
train_userlog_ids = list( set(userlog_ids) & set(all_train_id) )
test_userlog_ids = list( set(userlog_ids) & set(all_test_id) )


# In[36]:


def predict_batch(model, master_ids):
    
    if(len(master_ids) > args.n_eval):
        master_ids = random.sample(master_ids, args.n_eval)

    torch_dataset = AppDataset(master_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
#     for step, data in enumerate(tqdm(data_loader)):
    for step, data in enumerate(data_loader):
        loss, y1, y2, y3, y4, _ = model(data)
        
        loss_list.append(loss.item())
        y1_list.append(y1.cpu().detach().numpy())
        y2_list.append(y2.cpu().detach().numpy())
        y3_list.append(y3.cpu().detach().numpy())
        y4_list.append(y4.cpu().detach().numpy())

        label1_list.append(data['labels1'].cpu().detach().numpy())
        label2_list.append(data['labels2'].cpu().detach().numpy())
        label3_list.append(data['labels3'].cpu().detach().numpy())
        label4_list.append(data['labels4'].cpu().detach().numpy())

    loss = np.mean(loss_list)
    
    y1_np = np.concatenate(y1_list,  axis = 0)
    y2_np = np.concatenate(y2_list,  axis = 0)
    y3_np = np.concatenate(y3_list,  axis = 0)
    y4_np = np.concatenate(y4_list,  axis = 0)
    
    y1_np = softmax(y1_np, axis=1)
    y2_np = softmax(y2_np, axis=1)
    y3_np = softmax(y3_np, axis=1)
    y4_np = softmax(y4_np, axis=1)

    return y1_np, y2_np, y3_np, y4_np


# In[37]:


user_info_y1, user_info_y2, user_info_y3, user_info_y4 = predict_batch(user_net.to(args.device), test_user_ids)
app_behave_y1, app_behave_y2, app_behave_y3, app_behave_y4 = predict_batch(app_behave_net.to(args.device), test_app_behave_ids)
app_list_y1, app_list_y2, app_list_y3, app_list_y4 = predict_batch(app_list_net.to(args.device), test_app_list_ids)
user_log_y1, user_log_y2, user_log_y3, user_log_y4 = predict_batch(userlog_net.to(args.device), test_userlog_ids)


# In[38]:


df_score = pd.DataFrame(np.zeros((len(all_test_id), 17)) - 1,
                        columns = ['n_view', 
                                   'user_info_y1', 'user_info_y2', 'user_info_y3', 'user_info_y4',
                                   'app_behave_y1', 'app_behave_y2', 'app_behave_y3', 'app_behave_y4',
                                   'app_list_y1', 'app_list_y2', 'app_list_y3', 'app_list_y4',
                                   'user_log_y1', 'user_log_y2', 'user_log_y3', 'user_log_y4',],
                        index = all_test_id)


# In[39]:


for master_id in test_user_ids:
    cnt = (master_id in install_behave_set) + (master_id in install_list_set) + (master_id in user_info_set) + (master_id in user_log_set)
    df_score.at[master_id, 'n_view'] = cnt


# In[40]:


for master_id, y1, y2, y3, y4 in zip(test_user_ids, user_info_y1, user_info_y2, user_info_y3, user_info_y4):
    df_score.at[master_id, 'user_info_y1'] = y1[1]
    df_score.at[master_id, 'user_info_y2'] = y2[1]
    df_score.at[master_id, 'user_info_y3'] = y3[1]
    df_score.at[master_id, 'user_info_y4'] = y4[1]

for master_id, y1, y2, y3, y4 in zip(test_app_behave_ids, app_behave_y1, app_behave_y2, app_behave_y3, app_behave_y4):
    df_score.at[master_id, 'app_behave_y1'] = y1[1]
    df_score.at[master_id, 'app_behave_y2'] = y2[1]
    df_score.at[master_id, 'app_behave_y3'] = y3[1]
    df_score.at[master_id, 'app_behave_y4'] = y4[1]

for master_id, y1, y2, y3, y4 in zip(test_app_list_ids, app_list_y1, app_list_y2, app_list_y3, app_list_y4):
    df_score.at[master_id, 'app_list_y1'] = y1[1]
    df_score.at[master_id, 'app_list_y2'] = y2[1]
    df_score.at[master_id, 'app_list_y3'] = y3[1]
    df_score.at[master_id, 'app_list_y4'] = y4[1]

for master_id, y1, y2, y3, y4 in zip(test_userlog_ids, user_log_y1, user_log_y2, user_log_y3, user_log_y4):
    df_score.at[master_id, 'user_log_y1'] = y1[1]
    df_score.at[master_id, 'user_log_y2'] = y2[1]
    df_score.at[master_id, 'user_log_y3'] = y3[1]
    df_score.at[master_id, 'user_log_y4'] = y4[1]


# In[41]:


df_score


# In[42]:


df_score['n_view'].value_counts() / df_score.shape[0]


# In[43]:


score_names = ['user_info_y1', 'user_info_y2', 'user_info_y3', 'user_info_y4',
               'app_behave_y1', 'app_behave_y2', 'app_behave_y3', 'app_behave_y4',
               'app_list_y1', 'app_list_y2', 'app_list_y3', 'app_list_y4',
               'user_log_y1', 'user_log_y2', 'user_log_y3', 'user_log_y4']

def mean_score(x):
    score1, score2, score3, score4 = 0, 0, 0, 0
    for name in ['user_info_y1', 'app_behave_y1', 'app_list_y1', 'user_log_y1']:
        if(x[name] != -1):
            score1 += x[name]
    for name in ['user_info_y2', 'app_behave_y2', 'app_list_y2', 'user_log_y2']:
        if(x[name] != -1):
            score2 += x[name]
    for name in ['user_info_y3', 'app_behave_y3', 'app_list_y3', 'user_log_y3']:
        if(x[name] != -1):
            score3 += x[name]
    for name in ['user_info_y4', 'app_behave_y4', 'app_list_y4', 'user_log_y4']:
        if(x[name] != -1):
            score4 += x[name]

    return pd.Series({
        'y1_mean' : score1 / x['n_view'],
        'y2_mean' : score2 / x['n_view'],
        'y3_mean' : score3 / x['n_view'],
        'y4_mean' : score4 / x['n_view'],
    })
    
df_mean_score = df_score.apply(mean_score, axis = 1)


# In[44]:


df_mean_score


# In[45]:


def max_score(x):
    score1 = max(x[['user_info_y1', 'app_behave_y1', 'app_list_y1', 'user_log_y1']])
    score2 = max(x[['user_info_y2', 'app_behave_y2', 'app_list_y2', 'user_log_y2']])
    score3 = max(x[['user_info_y3', 'app_behave_y3', 'app_list_y3', 'user_log_y3']])
    score4 = max(x[['user_info_y4', 'app_behave_y4', 'app_list_y4', 'user_log_y4']])
    return pd.Series({
        'y1_max' : score1,
        'y2_max' : score2,
        'y3_max' : score3,
        'y4_max' : score4,
    })

df_max_score = df_score.apply(max_score, axis = 1)


# In[46]:


new_ids = set(df_master_records[df_master_records['new_client']].index) & set(df_mean_score.index)
old_ids = set(df_master_records[~df_master_records['new_client']].index) & set(df_mean_score.index)

ret_dict_mean = {
    '1m30+' : roc_auc_score(df_master_records['target_1m30+'].loc[all_test_id], df_mean_score['y1_mean']),
    '2m30+' : roc_auc_score(df_master_records['target_2m30+'].loc[all_test_id], df_mean_score['y2_mean']),
    '3m30+' : roc_auc_score(df_master_records['target_3m30+'].loc[all_test_id], df_mean_score['y3_mean']),
    '4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[all_test_id], df_mean_score['y4_mean']),
    'new_1m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_mean_score['y1_mean'].loc[new_ids]),
    'new_2m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_mean_score['y2_mean'].loc[new_ids]),
    'new_3m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_mean_score['y3_mean'].loc[new_ids]),
    'new_4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_mean_score['y4_mean'].loc[new_ids]),
    'old_1m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_mean_score['y1_mean'].loc[old_ids]),
    'old_2m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_mean_score['y2_mean'].loc[old_ids]),
    'old_3m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_mean_score['y3_mean'].loc[old_ids]),
    'old_4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_mean_score['y4_mean'].loc[old_ids]),
}


ret_dict_max = {
    '1m30+' : roc_auc_score(df_master_records['target_1m30+'].loc[all_test_id], df_max_score['y1_max']),
    '2m30+' : roc_auc_score(df_master_records['target_2m30+'].loc[all_test_id], df_max_score['y2_max']),
    '3m30+' : roc_auc_score(df_master_records['target_3m30+'].loc[all_test_id], df_max_score['y3_max']),
    '4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[all_test_id], df_max_score['y4_max']),
    'new_1m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_max_score['y1_max'].loc[new_ids]),
    'new_2m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_max_score['y2_max'].loc[new_ids]),
    'new_3m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_max_score['y3_max'].loc[new_ids]),
    'new_4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[new_ids], df_max_score['y4_max'].loc[new_ids]),
    'old_1m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_max_score['y1_max'].loc[old_ids]),
    'old_2m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_max_score['y2_max'].loc[old_ids]),
    'old_3m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_max_score['y3_max'].loc[old_ids]),
    'old_4m30+' : roc_auc_score(df_master_records['target_4m30+'].loc[old_ids], df_max_score['y4_max'].loc[old_ids]),
}

pd.DataFrame([ret_dict_mean, ret_dict_max], index = ['mean', 'max'])


# ### train all & test single

# In[19]:


def extend(origin_data, start_extend_id, add_id = True):
    train_data = origin_data.loc[list(set(all_train_id) & set(origin_data.index))]
    extend_id = list(range(start_extend_id, start_extend_id+train_data.shape[0]))
    if add_id:
        start_extend_id += train_data.shape[0]
    
    new_feature = train_data.copy()
    new_label = df_target.loc[train_data.index].copy()
    
    new_feature.index = extend_id
    new_label.index = extend_id
    
    return new_feature, new_label, start_extend_id

start_extend_id = max(df_master_records.index) + 1
extend_user_info, extend_user_info_label, start_extend_id =  extend(df_user_one_hot, start_extend_id)
extend_applist, extend_applist_label, start_extend_id =  extend(se_id_install_list, start_extend_id)
extend_appbehave, extend_appbehave_label, start_extend_id =  extend(df_install_behave, start_extend_id, False)
extend_appbehavetime, extend_appbehavetime_label, start_extend_id =  extend(df_behave_time, start_extend_id)
extend_userlog, extend_userlog_label, start_extend_id =  extend(se_userlog_cross, start_extend_id)

df_user_one_hot = df_user_one_hot.append(extend_user_info)
se_id_install_list = se_id_install_list.append(extend_applist)
df_install_behave = df_install_behave.append(extend_appbehave)
df_behave_time = df_behave_time.append(extend_appbehavetime)
se_userlog_cross = se_userlog_cross.append(extend_userlog)

df_target = df_target.append(extend_user_info_label)
df_target = df_target.append(extend_applist_label)
df_target = df_target.append(extend_appbehave_label)
df_target = df_target.append(extend_userlog_label)
extend_train_ids = list(range(max(df_master_records.index) + 1, max(df_target.index) + 1)) + all_train_id


# In[20]:


install_behave_set = set(df_install_behave.index) & (set(extend_train_ids) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(extend_train_ids) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(extend_train_ids) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(extend_train_ids) | set(all_test_id))


# In[21]:


USE_VIEW = ""
def collate_fn_single(master_ids):
    
    name = USE_VIEW
    master_ids = np.array(master_ids)

#     sub_master_id = se_id_install_list.loc[master_ids]
#     df_sub_behave = df_install_behave.loc[master_ids]
#     df_sub_time = df_behave_time.loc[master_ids]
    for i, master_id in enumerate(master_ids):
        
        if master_id in user_info_set and name == 'user_info':
            x_dict['user_info'][i] = df_user_one_hot.loc[master_id].values
            x_dict['view_mask'][i][0] = 1
        else:
            x_dict['user_info'][i] = 0
            x_dict['view_mask'][i][0] = 0

        if master_id in install_list_set and name == 'install_list':
            app_list = se_id_install_list.at[master_id][:args.app_install_list_max_length]
            x_dict['app_list_len'][i] = len(app_list) + 1
            x_dict['app_list'][i][1 : x_dict['app_list_len'][i]] = app_list
            x_dict['app_list'][i][x_dict['app_list_len'][i] :] = 0
            x_dict['view_mask'][i][1] = 1
        else:
            x_dict['app_list_len'][i] = 1
            x_dict['app_list'][i] = 0
            x_dict['view_mask'][i][1] = 0

        if master_id in install_behave_set and name == 'install_behave':
            app_behave = df_install_behave['pkg_id'].at[master_id][-args.app_behave_max_length:]
            len_app = len(app_behave) + 1
            x_dict['app_behave_len'][i] = len_app
            x_dict['app_behave'][i][1: len_app] = app_behave
            x_dict['app_behave'][i][len_app :] = 0

            x_dict['app_behave_time_cut'][i][1:len_app] = df_behave_time['cut_id'].at[master_id][-args.app_behave_max_length:]
            x_dict['app_behave_time_qcut'][i][1:len_app] = df_behave_time['qcut_id'].at[master_id][-args.app_behave_max_length:]
            x_dict['app_behave_action'][i][1:len_app] = df_install_behave['action'].at[master_id][-args.app_behave_max_length:]
            x_dict['view_mask'][i][2] = 1
        else:
            x_dict['app_behave_len'][i] = 1
            x_dict['app_behave'][i] = 0
            x_dict['app_behave'][i] = 0

            x_dict['app_behave_time_cut'][i] = 0
            x_dict['app_behave_time_qcut'][i] = 0
            x_dict['app_behave_action'][i] = 0
            x_dict['view_mask'][i][2] = 0
        
        
        if master_id in user_log_set and name == 'user_log':
            userlog_list = se_userlog_cross.at[master_id][:args.userlog_max_length]
            x_dict['userlog_len'][i] = len(userlog_list) + 1
            x_dict['userlog'][i][1 : x_dict['userlog_len'][i]] = userlog_list
            x_dict['userlog'][i][x_dict['userlog_len'][i] :] = 0
            x_dict['view_mask'][i][3] = 1
        else:
            x_dict['userlog_len'][i] = 1
            x_dict['userlog'][i] = 0
            x_dict['view_mask'][i][3] = 0
            
    len_id = master_ids.shape[0]
    x_dict['app_list'][len_id:] = 0
    x_dict['app_behave'][len_id:] = 0
    return {
        
        'user_info' : torch.tensor(x_dict['user_info'][:len_id]).float(),
        
        'app_list' : torch.tensor(x_dict['app_list'][:len_id]).long(),
        'app_list_te_qcut' : torch.tensor(x_dict['app_list_te_qcut'][:len_id]).long(),
        'app_list_len' : torch.tensor(x_dict['app_list_len'][:len_id]).long(),
        
        'app_behave' : torch.tensor(x_dict['app_behave'][:len_id]).long(),
        'app_behave_te_qcut' : torch.tensor(x_dict['app_behave_te_qcut'][:len_id]).long(),
        'app_behave_len' : torch.tensor(x_dict['app_behave_len'][:len_id]).long(),
        
        'app_behave_time_cut' : torch.tensor(x_dict['app_behave_time_cut'][:len_id]).long(),
        'app_behave_time_qcut' : torch.tensor(x_dict['app_behave_time_qcut'][:len_id]).long(),
        'app_behave_action' : torch.tensor(x_dict['app_behave_action'][:len_id]).long(),
        'userlog' : torch.tensor(x_dict['userlog'][:len_id]).long(),
        'userlog_len' : torch.tensor(x_dict['userlog_len'][:len_id]).long(),
        'view_mask' : torch.tensor(x_dict['view_mask'][:len_id]).long(),
        'labels1' : torch.tensor(df_target.loc[master_ids]['target_1m30+'].values).long(),
        'labels2' : torch.tensor(df_target.loc[master_ids]['target_2m30+'].values).long(),
        'labels3' : torch.tensor(df_target.loc[master_ids]['target_3m30+'].values).long(),
        'labels4' : torch.tensor(df_target.loc[master_ids]['target_4m30+'].values).long(),
    }
def eval_data_single(model, master_ids):
    
    if(len(master_ids) > args.n_eval):
        master_ids = random.sample(master_ids, args.n_eval)

    torch_dataset = AppDataset(master_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        collate_fn=collate_fn_single,
        num_workers = args.n_worker,
    )
    
    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
    for step, data in enumerate(tqdm(data_loader)):
#     for step, data in enumerate(data_loader):
        loss, y1, y2, y3, y4, _ = model(data)
        
        loss_list.append(loss.item())
        y1_list.append(y1.cpu().detach().numpy())
        y2_list.append(y2.cpu().detach().numpy())
        y3_list.append(y3.cpu().detach().numpy())
        y4_list.append(y4.cpu().detach().numpy())

        label1_list.append(data['labels1'].cpu().detach().numpy())
        label2_list.append(data['labels2'].cpu().detach().numpy())
        label3_list.append(data['labels3'].cpu().detach().numpy())
        label4_list.append(data['labels4'].cpu().detach().numpy())

    loss = np.mean(loss_list)
    
    y1_np = np.concatenate(y1_list,  axis = 0)
    y2_np = np.concatenate(y2_list,  axis = 0)
    y3_np = np.concatenate(y3_list,  axis = 0)
    y4_np = np.concatenate(y4_list,  axis = 0)

    labels1_np = np.concatenate(label1_list,  axis = 0)
    labels2_np = np.concatenate(label2_list,  axis = 0)
    labels3_np = np.concatenate(label3_list,  axis = 0)
    labels4_np = np.concatenate(label4_list,  axis = 0)

    auc1 = roc_auc_score(labels1_np, y1_np[:, 1])
    auc2 = roc_auc_score(labels2_np, y2_np[:, 1])
    auc3 = roc_auc_score(labels3_np, y3_np[:, 1])
    auc4 = roc_auc_score(labels4_np, y4_np[:, 1])
    auc_all = [auc1, auc2, auc3, auc4]
    
    new_client = np.array(df_master_records['new_client'].loc[master_ids].values)
    new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

    old_client = ~new_client
    old_client_auc1 = roc_auc_score(labels1_np[old_client], y1_np[:, 1][old_client])
    old_client_auc2 = roc_auc_score(labels2_np[old_client], y2_np[:, 1][old_client])
    old_client_auc3 = roc_auc_score(labels3_np[old_client], y3_np[:, 1][old_client])
    old_client_auc4 = roc_auc_score(labels4_np[old_client], y4_np[:, 1][old_client])
    auc_old_client = [old_client_auc1, old_client_auc2, old_client_auc3, old_client_auc4]

    return {
        'loss' : loss,
        '1m30+' : auc1,
        '2m30+' : auc2,
        '3m30+' : auc3,
        '4m30+' : auc4,
        'new_1m30+' : new_client_auc1,
        'new_2m30+' : new_client_auc2,
        'new_3m30+' : new_client_auc3,
        'new_4m30+' : new_client_auc4,
        'old_1m30+' : old_client_auc1,
        'old_2m30+' : old_client_auc2,
        'old_3m30+' : old_client_auc3,
        'old_4m30+' : old_client_auc4,
    }

def train_multi_test_signle(train_ids, test_ids, model_class, config):
        
    torch_dataset = AppDataset(train_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=True,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    model = model_class(config).to(args.device)
#     model = UserNetwork().to(args.device)
#     model = AppBehaveNetwork().to(args.device)
#     model = AppListNetwork().to(args.device)
#     model = MaskMultiViewNetwork(config).to(args.device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

    decay = ["app_network"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in decay)], "weight_decay": args.weight_decay},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(train_ids)//(args.batch_size)), num_training_steps=int(len(train_ids) / args.batch_size * args.epoch)
    )

    for epoch in range(args.epoch):
        model.train()
        for step, data in enumerate(tqdm(data_loader)):
            loss, y1, y2, y3, y4, _ = model(data)            
            #backward
            S = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            optimizer.step()
            scheduler.step()
        
        model.eval()
#         train_ret_dict = eval_data(model, train_ids)
        test_ret_dict = eval_data(model, test_ids)
#         df_ret = pd.DataFrame([
#             train_ret_dict, test_ret_dict
#         ], index = ['train', 'test'])
        
        
        install_behave_test = list(set(install_behave_set) & set(test_ids))
        install_list_test = list(set(install_list_set) & set(test_ids))
        user_info_test = list(set(user_info_set) & set(test_ids))
        user_log_test = list(set(user_log_set) & set(test_ids))
        
        behave_ret_dict = eval_data(model, install_behave_test)
        list_ret_dict = eval_data(model, install_list_test)
        userinfo_ret_dict = eval_data(model, user_info_test)
        userlog_ret_dict = eval_data(model, user_log_test)

        df_ret = pd.DataFrame([
            test_ret_dict, behave_ret_dict, list_ret_dict, userinfo_ret_dict, userlog_ret_dict
        ], index = ['test', 'behave', 'list', 'user_info', 'user_log'])
        

        logging.info('epoch : %d' % epoch)
        ipd.display(df_ret)
        torch.save(model, 'model/multiview_net_extend.torch.model_%d' % epoch)

        
    return model


# In[26]:


class MultiviewConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
#         self.app_list_net = torch.load('app_list_net.model.torch')
#         self.app_behave_net =  torch.load('app_list_net.model.torch')
#         self.user_net =  torch.load('user_net.model.torch')
        self.app_list_net = AppListNetwork(AppListConfig())
        self.app_behave_net = AppBehaveNetwork(AppbehaveConfig())
        self.user_net = UserNetwork(UserNetConfig())
        self.userlog_network = UserlogNetwork(Userlogconfig())
# torch.save(MultiviewConfig(), '../data_sortout/MultiviewConfig.torch')
logging.info('start training')
# sample_ids = random.sample(extend_train_ids, int(1e4))
multiview_net = train_multi_test_signle(extend_train_ids, all_test_id, MaskMultiViewNetworkGenerateWithResidualAppInteractive, MultiviewAppInteractiveConfig())
logging.info('finish training')


# In[45]:


len(all_train_id)


# In[22]:


for i in range(12):
    
    model = torch.load('model/multiview_net_extend.torch.model_%d' % i)
    
    install_behave_test = list(set(install_behave_set) & set(all_test_id))
    install_list_test = list(set(install_list_set) & set(all_test_id))
    user_info_test = list(set(user_info_set) & set(all_test_id))
    user_log_test = list(set(user_log_set) & set(all_test_id))
    USE_VIEW = "install_behave"
    behave_ret_dict = eval_data_single(model, install_behave_test)
    USE_VIEW = "install_list"
    list_ret_dict = eval_data_single(model, install_list_test)
    USE_VIEW = "user_info"
    userinfo_ret_dict = eval_data_single(model, user_info_test)
    USE_VIEW = "user_log"
    userlog_ret_dict = eval_data_single(model, user_log_test)
    
    df_ret = pd.DataFrame([
        behave_ret_dict, list_ret_dict, userinfo_ret_dict, userlog_ret_dict
    ], index = ['behave', 'list', 'user_info', 'user_log'])
    
    ipd.display(df_ret)


# ### fit more

# In[19]:


class MultiviewConfig():
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.hidden = 256
#         self.app_list_net = torch.load('app_list_net.model.torch')
#         self.app_behave_net =  torch.load('app_list_net.model.torch')
#         self.user_net =  torch.load('user_net.model.torch')
        self.app_list_net = AppListNetwork(AppListConfig())
        self.app_behave_net = AppBehaveNetwork(AppbehaveConfig())
        self.user_net = UserNetwork(UserNetConfig())
        self.userlog_network = UserlogNetwork(Userlogconfig())
# torch.save(MultiviewConfig(), '../data_sortout/MultiviewConfig.torch')
logging.info('start training')
multiview_net = train(extend_train_ids, all_test_id, MaskMultiViewNetwork, MultiviewConfig())
logging.info('finish training')


# In[22]:


for i in range(12):
    
    model = torch.load('model/multiview_net_extend.torch.model_%d' % i)
    
    install_behave_test = list(set(install_behave_set) & set(all_test_id))
    install_list_test = list(set(install_list_set) & set(all_test_id))
    user_info_test = list(set(user_info_set) & set(all_test_id))
    user_log_test = list(set(user_log_set) & set(all_test_id))
    USE_VIEW = "install_behave"
    behave_ret_dict = eval_data_single(model, install_behave_test)
    USE_VIEW = "install_list"
    list_ret_dict = eval_data_single(model, install_list_test)
    USE_VIEW = "user_info"
    userinfo_ret_dict = eval_data_single(model, user_info_test)
    USE_VIEW = "user_log"
    userlog_ret_dict = eval_data_single(model, user_log_test)
    
    df_ret = pd.DataFrame([
        behave_ret_dict, list_ret_dict, userinfo_ret_dict, userlog_ret_dict
    ], index = ['behave', 'list', 'user_info', 'user_log'])
    
    ipd.display(df_ret)


# ### ret log

# app list
# 1.111740	0.664095	0.657518	0.639428	0.630501
# 
# app behave
# 0.663820	0.638084	0.627594	0.623197	
# 0.624079	0.600459	0.592388	0.591546	
# 0.695377	0.652742	0.635372	0.625435
# 
# add time
# 0.682645	0.656571	0.646690	0.634663
# 0.636268	0.618727	0.611120	0.603158
# 0.732818	0.667061	0.652106	0.634041
# 
# add time & action
# 0.678883	0.653626	0.641810	0.628926	
# 0.636229	0.619752	0.609672	0.60488	
# 0.718630	0.658465	0.643400	0.619151

# # debug code

# In[ ]:


# class PositionalEncoding(nn.Module):
#     def __init__(self, config):
#         super(PositionalEncoding, self).__init__()
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         max_len=config.max_position_embeddings
#         self.P=torch.zeros(1,max_len,config.hidden_size)
#         X=torch.arange(0,max_len).view(-1,1).float()/torch.pow(10000,torch.arange(0,config.hidden_size,2).float()/config.hidden_size)
#         self.P[:,:,0::2]=torch.sin(X.float())
#         self.P[:,:,1::2]=torch.cos(X.float())
#         self.Dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.layerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#     def forward(self,X,position_ids=None):
#         input_shape = X.size()[:-1]
#         seq_length = input_shape[1]
#         device = X.device
#         inputs_embeds=X
#         if position_ids is None:
#             position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
#             position_ids = position_ids.unsqueeze(0).expand(input_shape)
#         position_embeddings = self.position_embeddings(position_ids)

#         embeddings = inputs_embeds + position_embeddings
#         embeddings = self.layerNorm(embeddings)
#         embeddings = self.Dropout(embeddings)
#         return embeddings
# class Transformer(nn.Module):
#     def __init__(self,config):
#         super(Transformer,self).__init__()
#         self.config=config
#         self.Encoder=BertEncoder(config=config)
#         self.P=PositionalEncoding(config)
#         self.rnns=nn.ModuleList()
#         for i in range(1):
#             hidden_size=config.hidden_size
#             self.rnns.append(nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=1,\
#                         bidirectional=True,batch_first=True))
#         self.one=nn.Parameter(torch.tensor([1]).float())
#         for n,e in self.Encoder.named_modules():
#             self._init_weights(e)
#         for n,e in self.P.named_modules():
#             self._init_weights(e)
            
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         elif isinstance(module, BertLayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#     def make_mask(self,X,valid_len):
#             shape=X.shape
#             if valid_len.dim()==1:
#                 valid_len=valid_len.view(-1,1).repeat(1,shape[1])
#             mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
#             return mask
#     def forward(self, X, length, encoder_hidden_states=None,encoder_extended_attention_mask=None):

#         attention_mask=self.make_mask(X,length)
#         embedding_output=X
#         if attention_mask.dim() == 3:
#             extended_attention_mask = attention_mask[:, None, :, :]
#         elif attention_mask.dim() == 2:
#             extended_attention_mask = attention_mask[:, None, None, :]
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
#         #make head mask
#         head_mask = [None] * self.config.num_hidden_layers
#         outputs=self.Encoder(  embedding_output,
#                 attention_mask=extended_attention_mask,
#                 head_mask=head_mask,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_extended_attention_mask,)
#         return outputs[0]
    
# class LayerNorm(nn.Module):
#     def __init__(self,features,eps=1e-6):
#         super(LayerNorm,self).__init__()
#         self.gamma=nn.Parameter(torch.ones(features))
#         self.beta=nn.Parameter(torch.zeros(features))
#         self.eps=eps
#     def forward(self,X):
#         mean=X.mean(-1,keepdim=True)
#         std=X.std(-1,keepdim=True)
#         return self.gamma*(X-mean)/(std+self.eps)+self.beta

    
# config = {
#   "attention_probs_dropout_prob": 0, 
#   "directionality": "bidi", 
#   "hidden_act": "gelu", 
#   "hidden_dropout_prob": 0, 
#   "hidden_size": 4, 
#   "initializer_range": 0.02, 
#   "intermediate_size": 840, 
#   "max_position_embeddings": 128, 
#   "num_attention_heads": 1, 
#   "num_hidden_layers": 1, 
#   "type_vocab_size": 2, 
#   "vocab_size": 21128,
# }
# config=BertConfig(**config, output_hidden_states=True, output_attentions=True)

# X = torch.tensor([
#     [
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [0, 0, 0, 0],
#     ],
#     [
#         [2, 2, 3, 4],
#         [1, 2, 3, 4],
#         [2, 6, 0, 0],
#     ]
# ]).float()

# Length = torch.tensor([2, 3]).long()
# model = Transformer(config)

# model(X, Length)

