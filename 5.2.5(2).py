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
import json


# # Read

# In[5]:


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


# In[6]:


df_user_one_hot.shape


# ## word2vec

# In[9]:


wv_model_app_list = pickle.load(open('../data_sortout/wv_model_app_list.pickle', 'rb'))
wv_model_app_behave = pickle.load(open('../data_sortout/wv_model_app_behave.pickle', 'rb'))
wv_model_userlog = pickle.load(open('../data_sortout/wv_model_userlog_cross.pickle', 'rb'))


# # train

# ## 调参

# In[10]:


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
    epoch = 10,
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

# In[11]:


install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))


# In[12]:


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

# In[13]:


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

# In[14]:


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

# In[15]:


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

# In[16]:


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

        return loss, y1, y2, y3, y4, {
            'v_final' : hidden,
            
            'v_user_attribute' : hidden_user,
            'v_app' : hidden_app,
            'v_app_inlog' : hidden_userlog,
    
            'v_g_user_attribute' : generate_user,
            'v_g_app' : generate_app,
            'v_g_app_inlog' : generate_userlog,
        }


# ## trainer

# In[17]:


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
    hidden_dict_list = []
#     for step, data in enumerate(tqdm(data_loader)):
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            loss, y1, y2, y3, y4, hidden_dict = model(data)

            loss_list.append(loss.item())
            y1_list.append(y1.cpu().detach().numpy())
            y2_list.append(y2.cpu().detach().numpy())
            y3_list.append(y3.cpu().detach().numpy())
            y4_list.append(y4.cpu().detach().numpy())

            label1_list.append(data['labels1'].cpu().detach().numpy())
            label2_list.append(data['labels2'].cpu().detach().numpy())
            label3_list.append(data['labels3'].cpu().detach().numpy())
            label4_list.append(data['labels4'].cpu().detach().numpy())
            
            hidden_dict_list.append(hidden_dict)
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

def train(train_ids, test_ids, model_class, config, tune_log):
        
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
        tune_log.write(epoch)
        tune_log.write_df(df_ret)
        
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

# In[18]:


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

# In[19]:


class TuneLog():
    def __init__(self, model_name):
        time_str = time.strftime("%mm_%dd_%Hh_%Mm_%Ss")
        self.file_path = 'log/' + model_name + '_' + time_str + '_log'
        self.file = open(self.file_path, 'w+')
        self.file.write(model_name + '\n')
        arg_dict = args._asdict()
        arg_dict['device'] = str(arg_dict['device'])
        self.file.write(json.dumps(arg_dict, indent = 4) + '\n')
        
    def write(self, epoch, loss=None, df_ret=None):
        self.file.write('%s\nepoch:%d\n' % (time.strftime("%Y-%m-%d %H:%M:%S"), epoch))
#         self.file.write(str(df_ret))
        self.file.write('\n')
    
    def write_df(self, df_ret):
        self.file.write(str(df_ret))
        self.file.write('\n')
    
    def close(self):
        self.file.close()


# In[28]:


install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))


# In[29]:


mp_name_set_origin = {
    'install_behave_set' : install_behave_set.copy(),
    'install_list_set' : install_list_set.copy(),
    'user_info_set' : user_info_set.copy(),
    'user_log_set' : user_log_set.copy()
}


# In[30]:


mp_name_set = {
    'install_behave_set' : install_behave_set,
    'install_list_set' : install_list_set,
    'user_info_set' : user_info_set,
    'user_log_set' : user_log_set
}


# ## App List 和App behaviors同时缺失

# In[31]:


max_rate = len(mp_name_set['install_behave_set'] & mp_name_set['install_list_set']) / len(user_info_set)
rates = list(range(5, int(max_rate*100), 5))
rates.append(max_rate*100)
for rate in rates:
    print(rate)
    n_used = int(len(user_info_set) * rate/100.)
    new_items = random.sample(list(mp_name_set['install_behave_set'] & mp_name_set['install_list_set']), n_used)
    old_items = list(mp_name_set['install_behave_set'] & mp_name_set['install_list_set'])
    for item in old_items:
        mp_name_set['install_behave_set'].remove(item)
        mp_name_set['install_list_set'].remove(item)

    for item in new_items:
        mp_name_set['install_behave_set'].add(item)
        mp_name_set['install_list_set'].add(item)
        
    tune_log = TuneLog('install_list_behave_%f' % rate)

    multiview_net = train(all_train_id, all_test_id, 
                          MaskMultiViewNetworkGenerateWithResidualAppInteractive, 
                          MultiviewAppInteractiveConfig(), tune_log)
    tune_log.close()

    for item in mp_name_set_origin['install_behave_set']:
        mp_name_set['install_behave_set'].add(item)

    for item in mp_name_set_origin['install_list_set']:
        mp_name_set['install_list_set'].add(item)
        
    print(rate / 100., n_used)


# ## 各个view自己缺失

# In[22]:


for view_name in ['install_behave_set', 'install_list_set', 'user_log_set']:
    max_rate = len(mp_name_set[view_name]) / len(user_info_set)
    rates = list(range(10, int(max_rate*100), 10))
    rates.append(max_rate*100)
    for rate in rates:
        n_used = int(len(user_info_set) * rate/100.)
        new_items = random.sample(list(mp_name_set[view_name]), n_used)
        old_items = list(mp_name_set[view_name])
        print(view_name, len(mp_name_set[view_name]))
        for item in old_items:
            mp_name_set[view_name].remove(item)
        for item in new_items:
            mp_name_set[view_name].add(item)
        print(view_name, len(mp_name_set[view_name]))
        tune_log = TuneLog('%s_%f'%(view_name, rate))

        multiview_net = train(all_train_id, all_test_id, 
                              MaskMultiViewNetworkGenerateWithResidualAppInteractive, 
                              MultiviewAppInteractiveConfig(), tune_log)
        tune_log.close()
        
        for item in mp_name_set_origin[view_name]:
            mp_name_set[view_name].add(item)
        print(rate / 100., n_used)


# # 分析

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
sns.set(color_codes=True)


# In[4]:


def draw(complete_rate, customers, save_path, xaxis_name, names=['1R30', '2R30', '3R30', '4R30']):
    complete_rate = [str(i) + '%' for i in complete_rate]
    l1=plt.plot(complete_rate, customers[: ,0], color='blue', marker = 'o', label=names[0])
    l2=plt.plot(complete_rate, customers[: ,1], color='green', marker = 'v', label=names[1])
    l3=plt.plot(complete_rate, customers[: ,2], color='grey', marker = '^', label=names[2])
    l4=plt.plot(complete_rate, customers[: ,3], color='navy', marker = 's', label=names[3])
#     l5=plt.plot(complete_rate, ys[4], color='red', marker = 'p', label=names[4])
    
    size = 12
    plt.xlabel('Proportion of samples with %s' % xaxis_name, fontsize=size)
    plt.ylabel('AUC', fontsize=size)

    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.legend(fontsize=size)
    plt.savefig('pdf/%s.pdf' % save_path, bbox_inches='tight')
    plt.show()


# In[14]:


#install_behave_list_set
complete_rate = [5, 10, 15, 20, 25, 30, 35, 40]
all_customers = np.array([
    [ 0.733458,  0.722556 , 0.691586 , 0.676559],
    [ 0.729430,  0.723967 , 0.692527,  0.675286],
    [0.732478 , 0.722905 , 0.692182,  0.675574],
    [0.725755 , 0.718861 , 0.688496 , 0.674525],
    [0.725871 , 0.722579 , 0.691715 , 0.677471],
    [ 0.735530 , 0.724719,  0.697042 , 0.683351],
    [0.741042 , 0.724117,  0.697670,  0.682066],
    [0.753084, 0.741381, 0.710574, 0.692068]
])

new_customers = np.array([
    [0.685973,   0.683886 , 0.657839,   0.648048],
    [ 0.681231 ,  0.684802 ,   0.658567,   0.648294  ],
    [0.684422 ,  0.686330,   0.659816,   0.648023],
    [0.673039 ,  0.676500 , 0.650134,   0.641839],
    [ 0.666899,   0.679930, 0.653381,   0.644363],
    [0.685887 ,  0.685565, 0.656981,   0.648419],
    [0.685965 ,  0.678420, 0.655041,   0.645466],
    [0.693948, 0.695046, 0.668302, 0.656762]
])

regular_customers = np.array([
    [0.766452,   0.740284,   0.697695,   0.676253 ],
    [0.766456,   0.744278,   0.700236,   0.676163  ],
    [ 0.766605,   0.738493,   0.697228,   0.672771],
    [0.762427,   0.736658,   0.694296 ,  0.673095],
    [ 0.766602 ,  0.738357 ,  0.696139,   0.676818 ],
    [0.759425,   0.738805 ,  0.705383,   0.687229 ],
    [0.773204 ,  0.740614,   0.706832,   0.683237 ],
    [0.799148, 0.758816, 0.716508, 0.689202],
])

draw(complete_rate, all_customers, 'ALL_behave_list_list', 'app installed lists and app installation behaviors')
draw(complete_rate, new_customers, 'NEW_behave_list_list', 'app installed lists and app installation behaviors')
draw(complete_rate, regular_customers, 'REGULAR_behave_list_list', 'app installed lists and app installation behaviors')


# In[18]:


#install_list_set
complete_rate = [10, 20, 30, 40, 45]
all_customers = np.array([
    [0.73214, 0.722154,  0.693085,  0.677281],
    [ 0.737725, 0.724247 , 0.693778,  0.680566],
    [0.740075, 0.727352 ,0.699360,  0.682379],
    [ 0.742131, 0.725014,0.694025,  0.682170],
    [0.753084, 0.741381, 0.710574, 0.692068]
])

new_customers = np.array([
    [0.675739,    0.67827, 0.651393,   0.641896],
    [ 0.687269,   0.677432,  0.652538,   0.643672 ],
    [0.687828,   0.686190,   0.660731,   0.651952],
    [0.687343,   0.684698, 0.655300,   0.647338],
    [0.693948, 0.695046, 0.668302, 0.656762]
])

regular_customers = np.array([
    [ 0.776720,   0.741906,   0.703727,   0.682819  ],
    [0.755140,   0.742776,   0.700312,   0.682809  ],
    [0.768732,   0.740766,   0.704699,   0.679072  ],
    [0.779872,   0.736812,   0.696968,   0.679700  ],
    [0.799148, 0.758816, 0.716508, 0.689202]
])

draw(complete_rate, all_customers, 'ALL_install_list', 'app installed lists')
draw(complete_rate, new_customers, 'NEW_install_list', 'app installed lists')
draw(complete_rate, regular_customers, 'REGULAR_install_list', 'app installed lists')


# In[15]:


#install_behave_set
complete_rate = [10, 20, 30, 40, 50]
all_customers = np.array([
    [0.735911, 0.728049, 0.697188, 0.681079],
    [0.738768,0.729331,0.697724,0.679471],
    [0.736772, 0.726012, 0.695391, 0.680469],
    [0.734851, 0.723852, 0.698022, 0.681692],
    [0.753084, 0.741381, 0.710574, 0.692068]
])

new_customers = np.array([
    [0.683730, 0.689954, 0.663529, 0.651941],
    [0.687803, 0.686284, 0.654941, 0.642369],
    [0.684129,0.684084,0.658202,0.647121],
    [0.682366, 0.679486, 0.658860, 0.647018],
    [0.693948, 0.695046, 0.668302, 0.656762]
])

regular_customers = np.array([
    [0.771165, 0.742280, 0.702029, 0.677957],
    [0.767036, 0.745989, 0.706785, 0.681823],
    [0.775164,0.743774,0.701364,0.679957],
    [0.758271,0.739912,0.703115,0.681376],
    [0.799148, 0.758816, 0.716508, 0.689202]
])

draw(complete_rate, all_customers, 'ALL_install_behave', 'app installation behaviors')
draw(complete_rate, new_customers, 'NEW_install_behave', 'app installation behaviors')
draw(complete_rate, regular_customers, 'REGULAR_install_behave', 'app installation behaviors')


# In[17]:


#app in log _set
complete_rate = [10, 20, 30, 40, 50, 60, 70, 80]

all_customers = np.array([
    [ 0.653405,  0.637749,  0.625857,  0.623452],
    [ 0.669636 , 0.651415 , 0.633407,  0.626657],
    [ 0.671152 , 0.652319 , 0.635360,  0.631226],
    [0.677273 , 0.664526 , 0.641977 , 0.638659],
    [0.687167 , 0.673230,  0.650560 , 0.641009],
    [0.703735 , 0.691594,  0.672280 , 0.659739],
    [0.718075 , 0.699508,  0.677832 , 0.666433],
    [0.753084, 0.741381, 0.710574, 0.692068]
])

new_customers = np.array([
    [0.627515 ,  0.617978 , 0.604914  , 0.601008],
    [0.639424 ,  0.629102,  0.610257,   0.605741],
    [0.642967  , 0.631179, 0.613598 ,  0.612753],
    [0.637002 ,  0.636012, 0.614506 ,  0.61264],
    [0.653892,   0.649187, 0.627729  , 0.620903],
    [0.658502 ,  0.656188,  0.638372 ,  0.631688],
    [ 0.669198  , 0.657363,  0.637474 ,  0.633497 ],
    [0.693948, 0.695046, 0.668302, 0.656762]
])

regular_customers = np.array([
    [0.675397 ,  0.645219  , 0.630802 ,  0.622781  ],
    [0.702432 ,  0.659095  , 0.636552 ,  0.623374],
    [0.693552,   0.658884,   0.636758,   0.626013],
    [ 0.714992 ,  0.671378  , 0.641811  , 0.634123 ],
    [ 0.701900,   0.677473 ,  0.647938 ,  0.633786],
    [0.744929,   0.706313 ,  0.681264 ,  0.658289],
    [0.751224 ,  0.716922,   0.687605 ,  0.666498  ],
    [0.799148, 0.758816, 0.716508, 0.689202]
])

draw(complete_rate, all_customers, 'ALL_app-in_log', 'app-in logs')
draw(complete_rate, new_customers, 'NEW_app-in_log', 'app-in logs')
draw(complete_rate, regular_customers, 'REGULAR_app-in_log', 'app-in logs')


# In[ ]:




