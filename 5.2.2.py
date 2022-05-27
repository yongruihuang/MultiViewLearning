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
# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize(progress_bar=True)
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


# In[124]:


device=torch.device("cuda:2")
# device=torch.device("cpu")


# # Get views

# In[ ]:


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


# ## 数据划分

# In[3]:


split_date = datetime.datetime(2019, 8, 31)
end_date = datetime.datetime(2019, 9, 30)

df_master_records = df_master_records.dropna(axis=0, how='any')
df_train_master = df_master_records.query('loan_date <= @split_date')
df_test_master = df_master_records.query('loan_date > @split_date & loan_date <= @end_date')
all_train_id = list(df_train_master.index)
all_test_id = list(df_test_master.index)
logging.info('all_train_id len :%d, all_test_id: %d' % (len(all_train_id), len(all_test_id)))
df_target = df_master_records[['target_1m30+', 'target_2m30+', 'target_3m30+', 'target_4m30+']]


# In[4]:


max_app_list_id = max(se_id_install_list.apply(max))
max_app_behave_id = max(df_install_behave['pkg_id'].apply(max))
max_uselog_id = max(se_userlog_cross.apply(max))
start_app_list_id = max_app_list_id + 1
start_app_behave_id = max_app_behave_id + 1 
start_uselog_id = max_uselog_id + 1


# ## Feature

# In[5]:


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


# ## Model

# In[6]:


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
    epoch = 16,
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
#     device=torch.device("cuda:1"),
    device=torch.device("cpu")

)


# ## dataset

# In[7]:


install_behave_set = set(df_install_behave.index) & (set(all_train_id) | set(all_test_id))
install_list_set = set(se_id_install_list.index) & (set(all_train_id) | set(all_test_id))
user_info_set = set(df_user_one_hot.index) & (set(all_train_id) | set(all_test_id))
user_log_set = set(se_userlog_cross.index) & (set(all_train_id) | set(all_test_id))


# In[8]:


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


# ## view generate

# In[9]:


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


# In[10]:


def view_generate(model, master_ids):
    
    torch_dataset = AppDataset(master_ids)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    hidden_list = [] 
    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader)):
            loss, y1, y2, y3, y4, hidden = model(data)
            hidden_list.append(hidden.cpu().detach().numpy())
            
    return np.concatenate(hidden_list,  axis = 0)


# ### user attribute

# In[11]:


logging.info('start')

user_train_id = all_train_id
user_test_id = all_test_id
user_net = torch.load('../code_new_model/user_net.model.torch').to(args.device)
hidden_train = view_generate(user_net, user_train_id)
hidden_test = view_generate(user_net, user_test_id)
user_ret_dict = {
    'train_id' : user_train_id,
    'test_id' : user_test_id,
    'hidden_train' : hidden_train,
    'hidden_test' : hidden_test,
}
pickle.dump(user_ret_dict, open('5_data/user_attribute.pickle', 'wb'))
logging.info('finish')


# ### App behave

# In[12]:


app_behave_ids = list(df_install_behave.index)
train_app_behave_ids = list( set(app_behave_ids) & set(all_train_id) )
test_app_behave_ids = list( set(app_behave_ids) & set(all_test_id) )
len(train_app_behave_ids), len(test_app_behave_ids)


# In[13]:


logging.info('start')

app_behave_net = torch.load('../code_new_model/app_behave_net.model.torch').to(args.device)
hidden_train = view_generate(app_behave_net, train_app_behave_ids)
hidden_test = view_generate(app_behave_net, test_app_behave_ids)
app_behavior_dict = {
    'train_id' : train_app_behave_ids,
    'test_id' : test_app_behave_ids,
    'hidden_train' : hidden_train,
    'hidden_test' : hidden_test,
}
pickle.dump(app_behavior_dict, open('5_data/app_behaviors.pickle', 'wb'))
logging.info('finish')


# ### App list

# In[14]:


app_list_ids = se_id_install_list.index
train_app_list_ids = list( set(app_list_ids) & set(all_train_id) )
test_app_list_ids = list( set(app_list_ids) & set(all_test_id) )
len(train_app_list_ids), len(test_app_list_ids)


# In[17]:


logging.info('start')

app_list_net = torch.load('../code_new_model/app_list_net.model.torch').to(args.device)
hidden_train = view_generate(app_list_net, train_app_list_ids)
hidden_test = view_generate(app_list_net, test_app_list_ids)
app_list_dict = {
    'train_id' : train_app_list_ids,
    'test_id' : test_app_list_ids,
    'hidden_train' : hidden_train,
    'hidden_test' : hidden_test,
}
pickle.dump(app_list_dict, open('5_data/app_list.pickle', 'wb'))
logging.info('finish')


# ### App in-log

# In[18]:


userlog_ids = se_userlog_cross.index
train_userlog_ids = list( set(userlog_ids) & set(all_train_id) )
test_userlog_ids = list( set(userlog_ids) & set(all_test_id) )
len(train_userlog_ids), len(test_userlog_ids)


# In[20]:


logging.info('start')

user_log_net = torch.load('../code_new_model/userlog_net.model.torch').to(args.device)
hidden_train = view_generate(user_log_net, train_userlog_ids)
hidden_test = view_generate(user_log_net, test_userlog_ids)
user_log_dict = {
    'train_id' : train_userlog_ids,
    'test_id' : test_userlog_ids,
    'hidden_train' : hidden_train,
    'hidden_test' : hidden_test,
}
pickle.dump(user_log_dict, open('5_data/userlog.pickle', 'wb'))
logging.info('finish')


# ## 验证

# In[27]:


train_x, train_y, test_x, test_y = user_log_dict['hidden_train'], df_master_records['target_1m30+'].loc[user_log_dict['train_id']], user_log_dict['hidden_test'], df_master_records['target_1m30+'].loc[user_log_dict['test_id']]


# In[29]:


from sklearn import linear_model, svm, neural_network, ensemble
logging.info('start')
clf = ensemble.GradientBoostingClassifier(random_state=0)
clf.fit(train_x, train_y)
logging.info('finish')


# In[30]:


predict_test = clf.predict_proba(test_x)
auc_test = roc_auc_score(test_y, predict_test[:, 1])
auc_test


# # Load views

# ## load

# In[47]:


user_attribute_dict = pickle.load(open('5_data/user_attribute.pickle', 'rb'))
app_behavior_dict = pickle.load(open('5_data/app_behaviors.pickle', 'rb'))
app_list_dict = pickle.load(open('5_data/app_list.pickle', 'rb'))
user_log_dict = pickle.load(open('5_data/userlog.pickle', 'rb'))


# In[48]:


user_attribute_matrix = np.concatenate([user_attribute_dict['hidden_train'], user_attribute_dict['hidden_test']], axis = 0).astype('float32')
master_ids = user_attribute_dict['train_id'] + user_attribute_dict['test_id']
mp_master_id_idx = dict(zip(master_ids, range(len(master_ids))))
app_behaviors_matrix = np.zeros(user_attribute_matrix.shape).astype('float32')
app_list_matrix = np.zeros(user_attribute_matrix.shape).astype('float32')
user_log_matrix = np.zeros(user_attribute_matrix.shape).astype('float32')


# In[49]:


def set_matrix(matrix, ret_dict):
    exsit_id = ret_dict['train_id'] + ret_dict['test_id']
    exsit_idx = [mp_master_id_idx[master_id] for master_id in exsit_id]
    matrix[exsit_idx] = np.concatenate([ret_dict['hidden_train'], ret_dict['hidden_test']], axis = 0)
    return exsit_idx
app_behaviors_exsit_idx = set_matrix(app_behaviors_matrix, app_behavior_dict)
app_list_exsit_idx = set_matrix(app_list_matrix, app_list_dict)
user_log_exsit_idx = set_matrix(user_log_matrix, user_log_dict)


# In[50]:


user_attribute_exsit_idx = list(range(user_attribute_matrix.shape[0]))


# In[51]:


len(user_attribute_exsit_idx), len(app_behaviors_exsit_idx), len(app_list_exsit_idx), len(user_log_exsit_idx)


# In[52]:


df_master_records = pickle.load(open('../data_sortout/df_master_records.pickle', 'rb'))
new_client = (df_master_records['loan_sequence'] == 1).loc[user_attribute_dict['test_id']].values
old_client = ~new_client


# In[53]:


y1 = df_master_records.loc[user_attribute_dict['train_id'] + user_attribute_dict['test_id']]['target_1m30+'].values
y2 = df_master_records.loc[user_attribute_dict['train_id'] + user_attribute_dict['test_id']]['target_2m30+'].values
y3 = df_master_records.loc[user_attribute_dict['train_id'] + user_attribute_dict['test_id']]['target_3m30+'].values
y4 = df_master_records.loc[user_attribute_dict['train_id'] + user_attribute_dict['test_id']]['target_4m30+'].values


# In[54]:


y1, y2, y3, y4


# # Multi-View Learning With Incomplete Views

# ## generation

# In[11]:


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_items, n_hidden, n_view, n_factors=20):
        super().__init__()

        self.u_list = nn.ModuleList([nn.Embedding(n_items, n_factors) for i in range(n_view) ])
        self.w = nn.Embedding(n_hidden, n_factors)

    def forward(self, i_s, j_s, view_idx):
        feat_i = self.u_list[view_idx](i_s)
        feat_j = self.w(j_s).transpose(1, 0)
        result = torch.mm(feat_i, feat_j)
        
        return result
    
# device=torch.device("cpu")


# In[12]:


model = MatrixFactorization(user_attribute_matrix.shape[0], user_attribute_matrix.shape[1], 4).to(device)


# In[13]:


epoch = 1000
exsit_idxs = [user_attribute_exsit_idx, app_behaviors_exsit_idx, app_list_exsit_idx, user_log_exsit_idx]
matrixs = [
    torch.tensor(user_attribute_matrix).to(device), 
    torch.tensor(app_behaviors_matrix).to(device),
    torch.tensor(app_list_matrix).to(device),
    torch.tensor(user_log_matrix).to(device)
]

j_s = torch.arange(user_attribute_matrix.shape[1]).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for i in range(epoch):
    loss = 0
    for view_idx in range(4):
        ret = model(torch.tensor(exsit_idxs[view_idx]).to(device), j_s, view_idx)
        loss += torch.mean((ret - matrixs[view_idx][exsit_idxs[view_idx]]) ** 2)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
    optimizer.step()
    if ((i + 1) % 2000 == 0):
        logging.info('epoch:%d : loss: %f' % (i, loss.item()))


# In[17]:


all_idx_tensor = torch.tensor(exsit_idxs[0]).to(device)
user_attribute_generation = model(all_idx_tensor, j_s, 0)
app_behaviors_generation = model(all_idx_tensor, j_s, 1)
app_list_exsit_generation = model(all_idx_tensor, j_s, 2)
user_log_exsit_generation = model(all_idx_tensor, j_s, 3)


# In[18]:


user_attribute_generation.shape, app_behaviors_generation.shape, app_list_exsit_generation.shape, user_log_exsit_generation.shape


# In[14]:


generation_dict = {
    'user_attribute' : user_attribute_generation.cpu().detach().numpy(),
    'app_behaviors' : app_behaviors_generation.cpu().detach().numpy(),
    'app_list' : app_list_exsit_generation.cpu().detach().numpy(),
    'user_log' : user_log_exsit_generation.cpu().detach().numpy(),
    'y1' : y1,
    'y2' : y2,
    'y3' : y3,
    'y4' : y4,
}
pickle.dump(generation_dict, open('5_data/Chang_Xu2015_generation_views.pickle', 'wb'))


# ## classification

# In[11]:


generation_dict = pickle.load(open('5_data/Chang_Xu2015_generation_views.pickle', 'rb'))
user_attribute_generation = generation_dict['user_attribute']
app_behaviors_generation = generation_dict['app_behaviors']
app_list_exsit_generation = generation_dict['app_list']
user_log_exsit_generation = generation_dict['user_log']
# y1 = generation_dict['y1']
# y2 = generation_dict['y2']
# y3 = generation_dict['y3']
# y4 = generation_dict['y4']


# In[12]:


user_attribute_generation[user_attribute_exsit_idx] = user_attribute_matrix[user_attribute_exsit_idx] 
app_behaviors_generation[app_behaviors_exsit_idx] = app_behaviors_matrix[app_behaviors_exsit_idx]
app_list_exsit_generation[app_list_exsit_idx] = app_list_matrix[app_list_exsit_idx]
user_log_exsit_generation[user_log_exsit_idx]  = user_log_matrix[user_log_exsit_idx] 


# In[13]:


full_x = np.concatenate([
    user_attribute_generation,
    app_behaviors_generation,
    app_list_exsit_generation,
    user_log_exsit_generation,
], axis=1)

label1, label2, label3, label4 = y1.astype('float32'), y2.astype('float32'), y3.astype('float32'), y4.astype('float32')
full_x.shape


# In[14]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 64
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)
    
class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_dim = 1024
        self.dense_hidden = Dense(n_dim, 64)

        self.dense1 = Dense(64, 2)
        self.dense2 = Dense(64, 2)
        self.dense3 = Dense(64, 2)
        self.dense4 = Dense(64, 2)
        
    def forward(self, x, labels1, labels2, labels3, labels4 ):
                
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


# In[15]:


num_train = len(user_attribute_dict['train_id'])
# train_x, test_x = tensor_x[:num_train].clone(), tensor_x[num_train:].clone()
# train_y1, test_y1 = label1[:num_train].clone(), label1[num_train:].clone()
# train_y2, test_y2 = label2[:num_train].clone(), label2[num_train:].clone()
# train_y3, test_y3 = label3[:num_train].clone(), label3[num_train:].clone()
# train_y4, test_y4 = label4[:num_train].clone(), label4[num_train:].clone()

# train_x, test_x = tensor_x[:num_train].clone().detach().requires_grad_(True), tensor_x[num_train:].clone().detach().requires_grad_(True)
# train_y1, test_y1 = label1[:num_train].clone().detach().requires_grad_(True), label1[num_train:].clone().detach().requires_grad_(True)
# train_y2, test_y2 = label2[:num_train].clone().detach().requires_grad_(True), label2[num_train:].clone().detach().requires_grad_(True)
# train_y3, test_y3 = label3[:num_train].clone().detach().requires_grad_(True), label3[num_train:].clone().detach().requires_grad_(True)
# train_y4, test_y4 = label4[:num_train].clone().detach().requires_grad_(True), label4[num_train:].clone().detach().requires_grad_(True)

train_x, test_x = torch.tensor(full_x[:num_train]), torch.tensor(full_x[num_train:])
train_y1, test_y1 = torch.tensor(label1[:num_train]), torch.tensor(label1[num_train:])
train_y2, test_y2 = torch.tensor(label2[:num_train]), torch.tensor(label2[num_train:])
train_y3, test_y3 = torch.tensor(label3[:num_train]), torch.tensor(label3[num_train:])
train_y4, test_y4 = torch.tensor(label4[:num_train]), torch.tensor(label4[num_train:])


# In[16]:


train_dataset = torch.utils.data.TensorDataset(train_x, train_y1, train_y2, train_y3, train_y4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 0)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y1, test_y2, test_y3, test_y4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 0)


# In[17]:


def eval_data(model):
    global new_client, old_client

    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
    with torch.no_grad():
        for x, l1, l2, l3, l4 in test_loader:
            loss, y1, y2, y3, y4, _ = model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))

            loss_list.append(loss.item())
            y1_list.append(y1.cpu().detach().numpy())
            y2_list.append(y2.cpu().detach().numpy())
            y3_list.append(y3.cpu().detach().numpy())
            y4_list.append(y4.cpu().detach().numpy())

            label1_list.append(l1.long().cpu().detach().numpy())
            label2_list.append(l2.long().cpu().detach().numpy())
            label3_list.append(l3.long().cpu().detach().numpy())
            label4_list.append(l4.long().cpu().detach().numpy())

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
    
    new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

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
    
epoch = 12
output_model = OutputLayer().to(device)
optimizer = AdamW(output_model.parameters(), lr = 0.001, weight_decay = 0)

for i in range(epoch):
    output_model.train()

    for x, l1, l2, l3, l4 in tqdm(train_loader):
        loss, y1, y2, y3, y4, _ = output_model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))
        optimizer.zero_grad()

#         loss.backward(retain_graph=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(output_model.parameters(), max_norm = 2)

        optimizer.step()
        
    output_model.eval()
    
#     train_ret_dict = eval_data(output_model)
    test_ret_dict = eval_data(output_model)
    df_ret = pd.DataFrame([
        test_ret_dict
    ], index = ['test'])
    ipd.display(df_ret)


# # CPM-Nets: Cross Partial Multi-View Networks

# In[14]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 64
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)

class CPM(torch.nn.Module):
    def __init__(self, n_train, n_test, n_hidden, n_view):
        super().__init__()

        n_factors = 64
        self.n_view = n_view
        self.h_train = nn.Embedding(n_train, n_factors)
        self.h_test = nn.Embedding(n_test, n_factors)
        self.pro_list = nn.ModuleList([Dense(n_factors, n_hidden) for i in range(n_view)])
        
        self.dense1 = Dense(n_factors, 2)
        self.dense2 = Dense(n_factors, 2)
        self.dense3 = Dense(n_factors, 2)
        self.dense4 = Dense(n_factors, 2)
    
    def train_forward(self, **param):

        train_hidden = self.h_train(param['train_idx'])
        test_hidden = self.h_test(param['test_idx'])
        reconstruction_loss = 0
        for i in range(self.n_view):
            rebuild_train = self.pro_list[i](train_hidden)
            rebuild_test = self.pro_list[i](test_hidden)
            train_exist_idx_v_i = param['train_exist_idx_%d'%i]
            test_exist_idx_v_i = param['test_exist_idx_%d'%i]

            loss_rebuild_train = (rebuild_train[train_exist_idx_v_i] - param['view_train_%d' % i][train_exist_idx_v_i]) ** 2
            loss_rebuild_test = (rebuild_train[test_exist_idx_v_i] - param['view_test_%d' % i][test_exist_idx_v_i]) ** 2
            reconstruction_loss += torch.mean(loss_rebuild_train) + torch.mean(loss_rebuild_test)
            
        y1 = self.dense1(train_hidden)
        y2 = self.dense2(train_hidden)
        y3 = self.dense3(train_hidden)
        y4 = self.dense4(train_hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, param['labels1'].long())
        loss2 = loss_func(y2, param['labels2'].long())
        loss3 = loss_func(y3, param['labels3'].long())
        loss4 = loss_func(y4, param['labels4'].long())
        
        classification_loss = loss1 + loss2 + loss3 + loss4
        
        return reconstruction_loss, classification_loss, reconstruction_loss + classification_loss 

    def test_forward(self, **param):
        test_hidden = self.h_test(param['test_idx'])
        
        y1 = self.dense1(test_hidden)
        y2 = self.dense2(test_hidden)
        y3 = self.dense3(test_hidden)
        y4 = self.dense4(test_hidden)
        
        return y1, y2, y3, y4
    


# In[15]:


num_train = len(user_attribute_dict['train_id'])
num_test = len(user_attribute_dict['test_id'])
v0_exist_set = set(user_attribute_exsit_idx)
v1_exist_set = set(app_behaviors_exsit_idx)
v2_exist_set = set(app_list_exsit_idx)
v3_exist_set = set(user_log_exsit_idx)

v0_train_m, v0_test_m = user_attribute_matrix[:num_train], user_attribute_matrix[num_train:]
v1_train_m, v1_test_m = app_behaviors_matrix[:num_train], app_behaviors_matrix[num_train:]
v2_train_m, v2_test_m = app_list_matrix[:num_train], app_list_matrix[num_train:]
v3_train_m, v3_test_m = user_log_matrix[:num_train], user_log_matrix[num_train:]


# In[13]:


# tensor_view_train_0 = torch.tensor(v0_train_m).to(device),
# tensor_view_train_1 = torch.tensor(v1_train_m).to(device),
# tensor_view_train_2 = torch.tensor(v2_train_m).to(device),
# tensor_view_train_3 = torch.tensor(v3_train_m).to(device),

# tensor_view_test_0 = torch.tensor(v0_test_m).to(device),
# tensor_view_test_1 = torch.tensor(v1_test_m).to(device),
# tensor_view_test_2 = torch.tensor(v2_test_m).to(device),
# tensor_view_test_3 = torch.tensor(v3_test_m).to(device),


# In[16]:


model = CPM(num_train, num_test, 256, 4).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# In[ ]:


logging.info('start')

def train():
    batch_size = 256
    epoch = 1024

    for epoch in range(epoch):
        model.train()
        idxs = list(range(user_attribute_matrix.shape[0]))
        random.shuffle(idxs)

#         for i in tqdm(range(0, user_attribute_matrix.shape[0], batch_size)):
        for i in (range(0, user_attribute_matrix.shape[0], batch_size)):
            select_idx = idxs[i:i+batch_size]
            train_idx = [idx for idx in select_idx if idx < num_train]
            test_idx = [idx - num_train for idx in select_idx if idx >= num_train]

            loss1, loss2, loss = model.train_forward(
                train_exist_idx_0 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v0_exist_set]).long().to(device),
                train_exist_idx_1 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v1_exist_set]).long().to(device),
                train_exist_idx_2 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v2_exist_set]).long().to(device),
                train_exist_idx_3 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v3_exist_set]).long().to(device),

                test_exist_idx_0 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v0_exist_set]).long().to(device),
                test_exist_idx_1 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v1_exist_set]).long().to(device),
                test_exist_idx_2 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v2_exist_set]).long().to(device),
                test_exist_idx_3 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v3_exist_set]).long().to(device),

                view_train_0 = torch.tensor(v0_train_m[train_idx].astype('float32')).to(device),
                view_train_1 = torch.tensor(v1_train_m[train_idx].astype('float32')).to(device),
                view_train_2 = torch.tensor(v2_train_m[train_idx].astype('float32')).to(device),
                view_train_3 = torch.tensor(v3_train_m[train_idx].astype('float32')).to(device),

                view_test_0 = torch.tensor(v0_test_m[test_idx].astype('float32')).to(device),
                view_test_1 = torch.tensor(v1_test_m[test_idx].astype('float32')).to(device),
                view_test_2 = torch.tensor(v2_test_m[test_idx].astype('float32')).to(device),
                view_test_3 = torch.tensor(v3_test_m[test_idx].astype('float32')).to(device),

                train_idx = torch.tensor(train_idx).long().to(device),
                test_idx = torch.tensor(test_idx).long().to(device),
                labels1 = torch.tensor(y1[train_idx]).long().to(device),
                labels2 = torch.tensor(y2[train_idx]).long().to(device),
                labels3 = torch.tensor(y3[train_idx]).long().to(device),
                labels4 = torch.tensor(y4[train_idx]).long().to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            optimizer.step()
        model.eval()

        y1_list, y2_list, y3_list, y4_list = [], [], [], []
        with torch.no_grad():
            idxs = list(range(num_test))
            for i in range(0, num_test, batch_size):
                select_idx = idxs[i:i+batch_size]
                y1_tensor, y2_tensor, y3_tensor, y4_tensor= model.test_forward(test_idx = torch.tensor(select_idx).long().to(device))
                y1_list.append(y1_tensor.cpu().detach().numpy())
                y2_list.append(y2_tensor.cpu().detach().numpy())
                y3_list.append(y3_tensor.cpu().detach().numpy())
                y4_list.append(y4_tensor.cpu().detach().numpy())

        y1_np = np.concatenate(y1_list,  axis = 0)
        y2_np = np.concatenate(y2_list,  axis = 0)
        y3_np = np.concatenate(y3_list,  axis = 0)
        y4_np = np.concatenate(y4_list,  axis = 0)
        
        labels1_np = y1[-num_test:]
        labels2_np = y2[-num_test:]
        labels3_np = y3[-num_test:]
        labels4_np = y4[-num_test:]

        auc1 = roc_auc_score(labels1_np, y1_np[:, 1])
        auc2 = roc_auc_score(labels2_np, y2_np[:, 1])
        auc3 = roc_auc_score(labels3_np, y3_np[:, 1])
        auc4 = roc_auc_score(labels4_np, y4_np[:, 1])
        auc_all = [auc1, auc2, auc3, auc4]

        new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
        new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
        new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
        new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
        auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

        old_client_auc1 = roc_auc_score(labels1_np[old_client], y1_np[:, 1][old_client])
        old_client_auc2 = roc_auc_score(labels2_np[old_client], y2_np[:, 1][old_client])
        old_client_auc3 = roc_auc_score(labels3_np[old_client], y3_np[:, 1][old_client])
        old_client_auc4 = roc_auc_score(labels4_np[old_client], y4_np[:, 1][old_client])
        auc_old_client = [old_client_auc1, old_client_auc2, old_client_auc3, old_client_auc4]

        df_ret = pd.DataFrame([{
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
        }], index = ['test'])
        
        if epoch % 32 == 0:
            logging.info('epoch: %d' % epoch)
            ipd.display(df_ret)
train()
logging.info('end')


# In[26]:


len(select_idx)


# # GAN

# ## generation

# In[125]:


user_attribute_matrix.shape, app_behaviors_matrix.shape, app_list_matrix.shape, user_log_matrix.shape


# In[126]:


class Generator(nn.Module):
    def __init__(self, input_size = 256, output_size = 256, n_view = 4):
        super(Generator, self).__init__()
        self.map_list = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(n_view)])
        self.map = nn.Linear(input_size, output_size)
        
    def forward(self, v1, v2, v3, v4):
#         v1_generation = self.map_list[0](v1)
#         v2_generation = self.map_list[1](v2)
#         v3_generation = self.map_list[2](v3)
#         v4_generation = self.map_list[3](v4)
        v1_generation = self.map(v1)
        v2_generation = self.map(v2)
        v3_generation = self.map(v3)
        v4_generation = self.map(v4)

        return torch.cat([v1_generation, v2_generation, v3_generation, v4_generation])
    
class Discriminator(nn.Module):
    def __init__(self, input_size = 256):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, 1)
 
    def forward(self, x):
        x = self.map1(x)
        return F.sigmoid(x)


# In[128]:


def sample_real_data(batch_size):
    global user_attribute_matrix, app_behaviors_matrix, app_list_matrix, user_log_matrix,    user_attribute_exsit_idx, app_behaviors_exsit_idx, app_list_exsit_idx, user_log_exsit_idx
    assert batch_size % 4 == 0
    idx1 = random.sample(user_attribute_exsit_idx, batch_size // 4)
    idx2 = random.sample(app_behaviors_exsit_idx, batch_size // 4)
    idx3 = random.sample(app_list_exsit_idx, batch_size // 4)
    idx4 = random.sample(user_log_exsit_idx, batch_size // 4)
    v1 = torch.tensor(user_attribute_matrix[idx1]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[idx2]).to(device)
    v3 = torch.tensor(app_list_matrix[idx3]).to(device)
    v4 = torch.tensor(user_log_matrix[idx4]).to(device)

    return torch.cat([v1, v2, v3, v4])

def sample_generate_data(G_net, batch_size):
    global user_attribute_matrix, app_behaviors_matrix, app_list_matrix, user_log_matrix
    assert batch_size % 4 == 0
    
    idx = random.sample(range(user_attribute_matrix.shape[0]), batch_size // 4)
    v1 = torch.tensor(user_attribute_matrix[idx]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[idx]).to(device)
    v3 = torch.tensor(app_list_matrix[idx]).to(device)
    v4 = torch.tensor(user_log_matrix[idx]).to(device)
    
    v_stack = G_net(v1, v2, v3, v4)
    return v_stack


# In[135]:


G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()

d_learning_rate = 2e-4  
g_learning_rate = 2e-4  
optim_betas = (0.9, 0.999)
num_epochs = 1024
d_steps = 1
g_steps = 1
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
batch_size = 256

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = sample_real_data(batch_size)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, torch.ones((batch_size, 1)).to(device))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = sample_generate_data(G, batch_size)
        d_fake_data = d_gen_input.detach()  # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        d_fake_error = criterion(d_fake_decision, torch.zeros((batch_size, 1)).to(device)) # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = sample_generate_data(G, batch_size)
        g_fake_data = gen_input
        dg_fake_decision = D(g_fake_data)
        g_error = criterion(dg_fake_decision, torch.ones((batch_size, 1)).to(device)) # we want to fool, so pretend it's all genuine

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if epoch % 108 == 0:
        logging.info('epoch:%d D loss:%f G loss:%f' % (epoch, d_fake_error.item(), g_error.item()))


# In[136]:


batch = 256
v1_list, v2_list, v3_list, v4_list = [], [], [], []
for i in range(0, user_attribute_matrix.shape[0], batch):
    v1 = torch.tensor(user_attribute_matrix[i : i + batch]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[i : i + batch]).to(device)
    v3 = torch.tensor(app_list_matrix[i : i + batch]).to(device)
    v4 = torch.tensor(user_log_matrix[i : i + batch]).to(device)
    
    v_generation = G(v1, v2, v3, v4)
    n = v_generation.shape[0] // 4
    v1_list.append(v_generation[:n].cpu().detach().numpy())
    v2_list.append(v_generation[n:n*2].cpu().detach().numpy())
    v3_list.append(v_generation[n*2:n*3].cpu().detach().numpy())
    v4_list.append(v_generation[n*3:].cpu().detach().numpy())


# In[137]:


user_attribute_generation = np.concatenate(v1_list)
app_behaviors_generation = np.concatenate(v2_list)
app_list_exsit_generation = np.concatenate(v3_list)
user_log_exsit_generation = np.concatenate(v4_list)
user_attribute_generation[user_attribute_exsit_idx] = user_attribute_matrix[user_attribute_exsit_idx] 
app_behaviors_generation[app_behaviors_exsit_idx] = app_behaviors_matrix[app_behaviors_exsit_idx]
app_list_exsit_generation[app_list_exsit_idx] = app_list_matrix[app_list_exsit_idx]
user_log_exsit_generation[user_log_exsit_idx]  = user_log_matrix[user_log_exsit_idx] 


# In[138]:


generation_dict = {
    'user_attribute' : user_attribute_generation,
    'app_behaviors' : app_behaviors_generation,
    'app_list' : app_list_exsit_generation,
    'user_log' : user_log_exsit_generation,
    'y1' : y1,
    'y2' : y2,
    'y3' : y3,
    'y4' : y4,
}
pickle.dump(generation_dict, open('5_data/GAN_generation_views.pickle', 'wb'))


# ## classificaiton

# In[139]:


generation_dict = pickle.load(open('5_data/GAN_generation_views.pickle', 'rb'))
user_attribute_generation = generation_dict['user_attribute']
app_behaviors_generation = generation_dict['app_behaviors']
app_list_exsit_generation = generation_dict['app_list']
user_log_exsit_generation = generation_dict['user_log']


# In[142]:


full_x = np.concatenate([
    user_attribute_generation,
    app_behaviors_generation,
    app_list_exsit_generation,
    user_log_exsit_generation,
], axis=1)

label1, label2, label3, label4 = y1.astype('float32'), y2.astype('float32'), y3.astype('float32'), y4.astype('float32')
full_x.shape


# In[143]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 64
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)
    
class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_dim = 1024
        self.dense_hidden = Dense(n_dim, 64)

        self.dense1 = Dense(64, 2)
        self.dense2 = Dense(64, 2)
        self.dense3 = Dense(64, 2)
        self.dense4 = Dense(64, 2)
        
    def forward(self, x, labels1, labels2, labels3, labels4 ):
                
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


# In[144]:


num_train = len(user_attribute_dict['train_id'])
# train_x, test_x = tensor_x[:num_train].clone(), tensor_x[num_train:].clone()
# train_y1, test_y1 = label1[:num_train].clone(), label1[num_train:].clone()
# train_y2, test_y2 = label2[:num_train].clone(), label2[num_train:].clone()
# train_y3, test_y3 = label3[:num_train].clone(), label3[num_train:].clone()
# train_y4, test_y4 = label4[:num_train].clone(), label4[num_train:].clone()

# train_x, test_x = tensor_x[:num_train].clone().detach().requires_grad_(True), tensor_x[num_train:].clone().detach().requires_grad_(True)
# train_y1, test_y1 = label1[:num_train].clone().detach().requires_grad_(True), label1[num_train:].clone().detach().requires_grad_(True)
# train_y2, test_y2 = label2[:num_train].clone().detach().requires_grad_(True), label2[num_train:].clone().detach().requires_grad_(True)
# train_y3, test_y3 = label3[:num_train].clone().detach().requires_grad_(True), label3[num_train:].clone().detach().requires_grad_(True)
# train_y4, test_y4 = label4[:num_train].clone().detach().requires_grad_(True), label4[num_train:].clone().detach().requires_grad_(True)

train_x, test_x = torch.tensor(full_x[:num_train]), torch.tensor(full_x[num_train:])
train_y1, test_y1 = torch.tensor(label1[:num_train]), torch.tensor(label1[num_train:])
train_y2, test_y2 = torch.tensor(label2[:num_train]), torch.tensor(label2[num_train:])
train_y3, test_y3 = torch.tensor(label3[:num_train]), torch.tensor(label3[num_train:])
train_y4, test_y4 = torch.tensor(label4[:num_train]), torch.tensor(label4[num_train:])


# In[145]:


train_dataset = torch.utils.data.TensorDataset(train_x, train_y1, train_y2, train_y3, train_y4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 0)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y1, test_y2, test_y3, test_y4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 0)


# In[146]:


def eval_data(model):
    global new_client, old_client

    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
    with torch.no_grad():
        for x, l1, l2, l3, l4 in test_loader:
            loss, y1, y2, y3, y4, _ = model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))

            loss_list.append(loss.item())
            y1_list.append(y1.cpu().detach().numpy())
            y2_list.append(y2.cpu().detach().numpy())
            y3_list.append(y3.cpu().detach().numpy())
            y4_list.append(y4.cpu().detach().numpy())

            label1_list.append(l1.long().cpu().detach().numpy())
            label2_list.append(l2.long().cpu().detach().numpy())
            label3_list.append(l3.long().cpu().detach().numpy())
            label4_list.append(l4.long().cpu().detach().numpy())

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
    
    new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

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
    
epoch = 12
output_model = OutputLayer().to(device)
optimizer = AdamW(output_model.parameters(), lr = 0.001, weight_decay = 0)

for i in range(epoch):
    output_model.train()

    for x, l1, l2, l3, l4 in tqdm(train_loader):
        loss, y1, y2, y3, y4, _ = output_model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))
        optimizer.zero_grad()

#         loss.backward(retain_graph=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(output_model.parameters(), max_norm = 2)

        optimizer.step()
        
    output_model.eval()
    
#     train_ret_dict = eval_data(output_model)
    test_ret_dict = eval_data(output_model)
    df_ret = pd.DataFrame([
        test_ret_dict
    ], index = ['test'])
    ipd.display(df_ret)


# # Self-paced_Multi-view_Co-training

# In[ ]:


import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC, NuSVC
from copy import deepcopy
from sklearn import linear_model, svm, neural_network, ensemble
from sklearn.linear_model import LogisticRegression

def sel_ids_y(score, add_num = 10):
    ids_sort = np.argsort(score)
    add_id = np.zeros(score.shape[0])
    add_id[ids_sort[:add_num]] = -1
    add_id[ids_sort[-add_num:]] = 1
    return add_id
    
def update_train_untrain(sel_ids, train_data, train_labels, untrain_data, weights=None):
#     sel_ids = np.array(sel_ids, dtype='bool')
    add_ids = np.where(np.array(sel_ids) != 0)[0]
    untrain_ids = np.where(np.array(sel_ids) == 0)[0]
    add_datas = [d[add_ids] for d in untrain_data]
    new_train_data = [np.concatenate([d1, d2]) for d1,d2 in zip(train_data, add_datas)]
    add_y = [1 if sel_ids[idx] > 0 else 0 for idx in add_ids]
    new_train_y = np.concatenate([train_labels, add_y])
    new_untrain_data = [d[untrain_ids] for d in untrain_data]
    return new_train_data, new_train_y, new_untrain_data


def cotrain(labeled_data, labels, unlabeled_data, iter_step=1):
    lbls = copy.deepcopy(labels)
    for step in range(iter_step):
        scores = []
        add_ids = []
        add_ys = []
        clfs = []
        for view in range(2):
            clfs.append(LinearSVC())
            clfs[view].fit(labeled_data[view], lbls)
            scores.append(clfs[view].decision_function(unlabeled_data[view]))
            add_id = sel_ids_y(scores[view], 6)
            add_ids.append(add_id)
        add_id = sum(add_ids)
        labeled_data, lbls, unlabeled_data = update_train_untrain(add_id, labeled_data, lbls, unlabeled_data)
        if len(unlabeled_data[view]) <= 0:
            break
    return clfs
        


def update_train(sel_ids, train_data, train_labels, untrain_data, pred_y):
    add_ids = np.where(np.array(sel_ids) != 0)[0]
    add_data = [d[add_ids] for d in untrain_data]
    new_train_data = [np.concatenate([d1, d2]) for d1,d2 in zip(train_data, add_data)]
    add_y = pred_y[add_ids]
    new_train_y = np.concatenate([train_labels, pred_y[add_ids]])
    return new_train_data, new_train_y


def spaco(l_data, lbls, u_data, iter_step = 1, gamma = 0.5):
    
    # initiate classifier
    clfs = []
    scores = []
    add_ids = []
    add_num = 6
    clfss = []
    for view in range(4):
        clfs.append(ensemble.GradientBoostingClassifier())
        clfs[view].fit(l_data[view], lbls)
        scores.append(clfs[view].decision_function(u_data[view]))
        add_ids.append(sel_ids_y(scores[view], add_num))
        py = [0  if s < 0 else 1 for s in scores[view]]
    score = sum(scores)
    pred_y = np.array([0  if s < 0 else 1 for s in score])
    for step in range(iter_step):
        for view in range(4):
            if add_num * 2 > u_data[0].shape[0]: break
            #update v
            ov = np.where(add_ids[1-view] != 0)[0]
            scores[view][ov] += add_ids[1-view][ov] * gamma
            add_ids[view] = sel_ids_y(scores[view], add_num)
            
            
            #update w
            nl_data, nlbls = update_train(add_ids[view], l_data, lbls, u_data, pred_y)
            clfs[view].fit(nl_data[view], nlbls)
            
            # update y, v
            scores[view] = clfs[view].decision_function(u_data[view])
            add_num += 6
            scores[view][ov] += add_ids[1-view][ov] * gamma
            add_ids[view] = sel_ids_y(scores[view], add_num)
            
            
            score = sum(scores)
            
            pred_y = np.array([0  if s < 0 else 1 for s in score])
            py = [0  if s < 0 else 1 for s in scores[view]]
    return clfs


# In[ ]:


train_new_id = np.array(df_master_records.loc[user_attribute_dict['train_id']]['loan_sequence']==1)
test_new_id = np.array(df_master_records.loc[user_attribute_dict['test_id']]['loan_sequence']==1)


# In[ ]:


n_used = -1
train_y_list = [train_y1, train_y2, train_y3, train_y4]
test_y_list = [test_y1, test_y2, test_y3, test_y4]

train_x_list = [train_x_user_attribute[:n_used], train_x_app_list[:n_used], train_x_app_behavior[:n_used], train_x_user_log[:n_used]]
test_x_list = [test_x_user_attribute[:n_used], test_x_app_list[:n_used], test_x_app_behavior[:n_used], test_x_user_log[:n_used]]

for i, y_n in enumerate(train_y_list): 
    clfs = spaco(
        train_x_list, 
        y_n[:n_used], 
        test_x_list, 
        iter_step=5, gamma=3)

    score = 0
    for t, view in enumerate(test_x_list):
        score += clfs[t].decision_function(view)
    auc = roc_auc_score(test_y_list[i][:n_used].astype('int'), score)
    new_auc = roc_auc_score(test_y_list[i][:n_used].astype('int')[test_new_id[:n_used]], score[test_new_id[:n_used]])
    old_auc = roc_auc_score(test_y_list[i][:n_used].astype('int')[~test_new_id[:n_used]], score[~test_new_id[:n_used]])

    print('%d auc:%f, new auc:%f, old auc:%f'%(i, auc, new_auc, old_auc))


# # Deep Partial Multi-View Learning

# In[ ]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 64
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)

class CPM(torch.nn.Module):
    def __init__(self, n_train, n_test, n_hidden, n_view):
        super().__init__()

        n_factors = 64
        self.n_view = n_view
        self.h_train = nn.Embedding(n_train, n_factors)
        self.h_test = nn.Embedding(n_test, n_factors)
        self.pro_list = nn.ModuleList([Dense(n_factors, n_hidden) for i in range(n_view)])
        
        self.dense1 = Dense(1088, 2)
        self.dense2 = Dense(1088, 2)
        self.dense3 = Dense(1088, 2)
        self.dense4 = Dense(1088, 2)
    
    def train_forward(self, **param):

        train_hidden = self.h_train(param['train_idx'])
        test_hidden = self.h_test(param['test_idx'])
        reconstruction_loss = 0
        for i in range(self.n_view):
            rebuild_train = self.pro_list[i](train_hidden)
            rebuild_test = self.pro_list[i](test_hidden)
            train_exist_idx_v_i = param['train_exist_idx_%d'%i]
            test_exist_idx_v_i = param['test_exist_idx_%d'%i]

            loss_rebuild_train = (rebuild_train[train_exist_idx_v_i] - param['view_train_%d' % i][train_exist_idx_v_i]) ** 2
            loss_rebuild_test = (rebuild_train[test_exist_idx_v_i] - param['view_test_%d' % i][test_exist_idx_v_i]) ** 2
            reconstruction_loss += torch.mean(loss_rebuild_train) + torch.mean(loss_rebuild_test)
        
        train_hidden = torch.cat([train_hidden, param['view_train_0'], 
                                     param['view_train_1'], param['view_train_2'],
                                     param['view_train_3']], -1)
        
        y1 = self.dense1(train_hidden)
        y2 = self.dense2(train_hidden)
        y3 = self.dense3(train_hidden)
        y4 = self.dense4(train_hidden)
        
        loss_func = nn.CrossEntropyLoss()
        loss1 = loss_func(y1, param['labels1'].long())
        loss2 = loss_func(y2, param['labels2'].long())
        loss3 = loss_func(y3, param['labels3'].long())
        loss4 = loss_func(y4, param['labels4'].long())
        
        classification_loss = loss1 + loss2 + loss3 + loss4
        
        return reconstruction_loss, classification_loss, reconstruction_loss + classification_loss 

    def test_forward(self, **param):
        test_hidden = self.h_test(param['test_idx'])

        test_hidden = torch.cat([test_hidden, param['view_test_0'], 
                                    param['view_test_1'], 
                                    param['view_test_2'],
                                    param['view_test_3']], -1)        
        y1 = self.dense1(test_hidden)
        y2 = self.dense2(test_hidden)
        y3 = self.dense3(test_hidden)
        y4 = self.dense4(test_hidden)
        
        return y1, y2, y3, y4
    


# In[ ]:


num_train = len(user_attribute_dict['train_id'])
num_test = len(user_attribute_dict['test_id'])
v0_exist_set = set(user_attribute_exsit_idx)
v1_exist_set = set(app_behaviors_exsit_idx)
v2_exist_set = set(app_list_exsit_idx)
v3_exist_set = set(user_log_exsit_idx)

v0_train_m, v0_test_m = user_attribute_matrix[:num_train], user_attribute_matrix[num_train:]
v1_train_m, v1_test_m = app_behaviors_matrix[:num_train], app_behaviors_matrix[num_train:]
v2_train_m, v2_test_m = app_list_matrix[:num_train], app_list_matrix[num_train:]
v3_train_m, v3_test_m = user_log_matrix[:num_train], user_log_matrix[num_train:]


# In[ ]:


device=torch.device("cuda:1")
model = CPM(num_train, num_test, 256, 4).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# In[ ]:


logging.info('start')

def train():
    batch_size = 256
    epoch = 12

    for epoch in range(epoch):
        model.train()
        idxs = list(range(user_attribute_matrix.shape[0]))
        random.shuffle(idxs)

        for i in tqdm(range(0, user_attribute_matrix.shape[0], batch_size)):
#         for i in (range(0, user_attribute_matrix.shape[0], batch_size)):
            select_idx = idxs[i:i+batch_size]
            train_idx = [idx for idx in select_idx if idx < num_train]
            test_idx = [idx - num_train for idx in select_idx if idx >= num_train]

            loss1, loss2, loss = model.train_forward(
                train_exist_idx_0 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v0_exist_set]).long().to(device),
                train_exist_idx_1 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v1_exist_set]).long().to(device),
                train_exist_idx_2 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v2_exist_set]).long().to(device),
                train_exist_idx_3 = torch.tensor([i for i, idx in enumerate(train_idx) if idx in v3_exist_set]).long().to(device),

                test_exist_idx_0 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v0_exist_set]).long().to(device),
                test_exist_idx_1 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v1_exist_set]).long().to(device),
                test_exist_idx_2 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v2_exist_set]).long().to(device),
                test_exist_idx_3 = torch.tensor([i for i, idx in enumerate(test_idx) if idx in v3_exist_set]).long().to(device),

                view_train_0 = torch.tensor(v0_train_m[train_idx].astype('float32')).to(device),
                view_train_1 = torch.tensor(v1_train_m[train_idx].astype('float32')).to(device),
                view_train_2 = torch.tensor(v2_train_m[train_idx].astype('float32')).to(device),
                view_train_3 = torch.tensor(v3_train_m[train_idx].astype('float32')).to(device),

                view_test_0 = torch.tensor(v0_test_m[test_idx].astype('float32')).to(device),
                view_test_1 = torch.tensor(v1_test_m[test_idx].astype('float32')).to(device),
                view_test_2 = torch.tensor(v2_test_m[test_idx].astype('float32')).to(device),
                view_test_3 = torch.tensor(v3_test_m[test_idx].astype('float32')).to(device),

                train_idx = torch.tensor(train_idx).long().to(device),
                test_idx = torch.tensor(test_idx).long().to(device),
                labels1 = torch.tensor(y1[train_idx]).long().to(device),
                labels2 = torch.tensor(y2[train_idx]).long().to(device),
                labels3 = torch.tensor(y3[train_idx]).long().to(device),
                labels4 = torch.tensor(y4[train_idx]).long().to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            optimizer.step()
        model.eval()

        y1_list, y2_list, y3_list, y4_list = [], [], [], []
        with torch.no_grad():
            idxs = list(range(num_test))
            for i in range(0, num_test, batch_size):
                select_idx = idxs[i:i+batch_size]
                y1_tensor, y2_tensor, y3_tensor, y4_tensor= model.test_forward(
                    test_idx = torch.tensor(select_idx).long().to(device),
                    view_test_0 = torch.tensor(v0_test_m[select_idx].astype('float32')).to(device),
                    view_test_1 = torch.tensor(v1_test_m[select_idx].astype('float32')).to(device),
                    view_test_2 = torch.tensor(v2_test_m[select_idx].astype('float32')).to(device),
                    view_test_3 = torch.tensor(v3_test_m[select_idx].astype('float32')).to(device)

                )
                y1_list.append(y1_tensor.cpu().detach().numpy())
                y2_list.append(y2_tensor.cpu().detach().numpy())
                y3_list.append(y3_tensor.cpu().detach().numpy())
                y4_list.append(y4_tensor.cpu().detach().numpy())

        y1_np = np.concatenate(y1_list,  axis = 0)
        y2_np = np.concatenate(y2_list,  axis = 0)
        y3_np = np.concatenate(y3_list,  axis = 0)
        y4_np = np.concatenate(y4_list,  axis = 0)
        
        labels1_np = y1[-num_test:]
        labels2_np = y2[-num_test:]
        labels3_np = y3[-num_test:]
        labels4_np = y4[-num_test:]

        auc1 = roc_auc_score(labels1_np, y1_np[:, 1])
        auc2 = roc_auc_score(labels2_np, y2_np[:, 1])
        auc3 = roc_auc_score(labels3_np, y3_np[:, 1])
        auc4 = roc_auc_score(labels4_np, y4_np[:, 1])
        auc_all = [auc1, auc2, auc3, auc4]

        new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
        new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
        new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
        new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
        auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

        old_client_auc1 = roc_auc_score(labels1_np[old_client], y1_np[:, 1][old_client])
        old_client_auc2 = roc_auc_score(labels2_np[old_client], y2_np[:, 1][old_client])
        old_client_auc3 = roc_auc_score(labels3_np[old_client], y3_np[:, 1][old_client])
        old_client_auc4 = roc_auc_score(labels4_np[old_client], y4_np[:, 1][old_client])
        auc_old_client = [old_client_auc1, old_client_auc2, old_client_auc3, old_client_auc4]

        df_ret = pd.DataFrame([{
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
        }], index = ['test'])
        
#         if epoch % 32 == 0:
        logging.info('epoch: %d' % epoch)
        ipd.display(df_ret)
train()
logging.info('end')


# # Generative Incomplete Multi-View Prognosis Predictor for Breast Cancer: GIMPP

# In[ ]:


class Generator(nn.Module):
    def __init__(self, input_size = 256, output_size = 256, n_view = 4):
        super(Generator, self).__init__()
        self.map_list = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(n_view)])
        self.map = nn.Linear(input_size, output_size)
        
    def forward(self, v1, v2, v3, v4):
#         v1_generation = self.map_list[0](v1)
#         v2_generation = self.map_list[1](v2)
#         v3_generation = self.map_list[2](v3)
#         v4_generation = self.map_list[3](v4)
        v1_generation = self.map(v1)
        v2_generation = self.map(v2)
        v3_generation = self.map(v3)
        v4_generation = self.map(v4)

        return torch.cat([v1_generation, v2_generation, v3_generation, v4_generation])
    
class Discriminator(nn.Module):
    def __init__(self, input_size = 256):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, 1)
 
    def forward(self, x):
        x = self.map1(x)
        return F.sigmoid(x)


# In[ ]:


def sample_real_data(batch_size):
    global user_attribute_matrix, app_behaviors_matrix, app_list_matrix, user_log_matrix,    user_attribute_exsit_idx, app_behaviors_exsit_idx, app_list_exsit_idx, user_log_exsit_idx
    assert batch_size % 4 == 0
    idx1 = random.sample(user_attribute_exsit_idx, batch_size // 4)
    idx2 = random.sample(app_behaviors_exsit_idx, batch_size // 4)
    idx3 = random.sample(app_list_exsit_idx, batch_size // 4)
    idx4 = random.sample(user_log_exsit_idx, batch_size // 4)
    v1 = torch.tensor(user_attribute_matrix[idx1]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[idx2]).to(device)
    v3 = torch.tensor(app_list_matrix[idx3]).to(device)
    v4 = torch.tensor(user_log_matrix[idx4]).to(device)

    return torch.cat([v1, v2, v3, v4])

def sample_generate_data(G_net, batch_size):
    global user_attribute_matrix, app_behaviors_matrix, app_list_matrix, user_log_matrix
    assert batch_size % 4 == 0
    
    idx = random.sample(range(user_attribute_matrix.shape[0]), batch_size // 4)
    v1 = torch.tensor(user_attribute_matrix[idx]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[idx]).to(device)
    v3 = torch.tensor(app_list_matrix[idx]).to(device)
    v4 = torch.tensor(user_log_matrix[idx]).to(device)
    
    v_stack = G_net(v1, v2, v3, v4)
    return v_stack


# In[ ]:


G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()

d_learning_rate = 2e-4  
g_learning_rate = 2e-4  
optim_betas = (0.9, 0.999)
num_epochs = 1024
d_steps = 1
g_steps = 1
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
batch_size = 256

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = sample_real_data(batch_size)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, torch.ones((batch_size, 1)).to(device))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = sample_generate_data(G, batch_size)
        d_fake_data = d_gen_input.detach()  # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        d_fake_error = criterion(d_fake_decision, torch.zeros((batch_size, 1)).to(device)) # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = sample_generate_data(G, batch_size)
        g_fake_data = gen_input
        dg_fake_decision = D(g_fake_data)
        #gan loss
        g_error = criterion(dg_fake_decision, torch.ones((batch_size, 1)).to(device)) # we want to fool, so pretend it's all genuine
        
        #mse loss
        g_error += torch.sum((gen_input-gen_input)**2)
        
        #STACKED RF loss 
        labels = torch.ones((batch_size, 1))
        g_error += criterion(dg_fake_decision, labels.to(device))
        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if epoch % 108 == 0:
        logging.info('epoch:%d D loss:%f G loss:%f' % (epoch, d_fake_error.item(), g_error.item()))


# In[ ]:


batch = 256
v1_list, v2_list, v3_list, v4_list = [], [], [], []
for i in range(0, user_attribute_matrix.shape[0], batch):
    v1 = torch.tensor(user_attribute_matrix[i : i + batch]).to(device)
    v2 = torch.tensor(app_behaviors_matrix[i : i + batch]).to(device)
    v3 = torch.tensor(app_list_matrix[i : i + batch]).to(device)
    v4 = torch.tensor(user_log_matrix[i : i + batch]).to(device)
    
    v_generation = G(v1, v2, v3, v4)
    n = v_generation.shape[0] // 4
    v1_list.append(v_generation[:n].cpu().detach().numpy())
    v2_list.append(v_generation[n:n*2].cpu().detach().numpy())
    v3_list.append(v_generation[n*2:n*3].cpu().detach().numpy())
    v4_list.append(v_generation[n*3:].cpu().detach().numpy())


# In[ ]:


user_attribute_generation = np.concatenate(v1_list)
app_behaviors_generation = np.concatenate(v2_list)
app_list_exsit_generation = np.concatenate(v3_list)
user_log_exsit_generation = np.concatenate(v4_list)
user_attribute_generation[user_attribute_exsit_idx] = user_attribute_matrix[user_attribute_exsit_idx] 
app_behaviors_generation[app_behaviors_exsit_idx] = app_behaviors_matrix[app_behaviors_exsit_idx]
app_list_exsit_generation[app_list_exsit_idx] = app_list_matrix[app_list_exsit_idx]
user_log_exsit_generation[user_log_exsit_idx]  = user_log_matrix[user_log_exsit_idx] 


# In[ ]:


generation_dict = {
    'user_attribute' : user_attribute_generation,
    'app_behaviors' : app_behaviors_generation,
    'app_list' : app_list_exsit_generation,
    'user_log' : user_log_exsit_generation,
    'y1' : y1,
    'y2' : y2,
    'y3' : y3,
    'y4' : y4,
}
# pickle.dump(generation_dict, open('5_data/GAN_generation_views.pickle', 'wb'))


# In[ ]:


# generation_dict = pickle.load(open('5_data/GAN_generation_views.pickle', 'rb'))
user_attribute_generation = generation_dict['user_attribute']
app_behaviors_generation = generation_dict['app_behaviors']
app_list_exsit_generation = generation_dict['app_list']
user_log_exsit_generation = generation_dict['user_log']


# In[ ]:


full_x = np.concatenate([
    user_attribute_generation,
    app_behaviors_generation,
    app_list_exsit_generation,
    user_log_exsit_generation,
], axis=1)

label1, label2, label3, label4 = y1.astype('float32'), y2.astype('float32'), y3.astype('float32'), y4.astype('float32')
full_x.shape


# In[ ]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        hidden = 64
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            GeLU(),
            nn.Dropout(0),
            nn.Linear(hidden, out_feature)
        )
        self.dense.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.dense(x)
    
class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_dim = 1024
        self.dense_hidden = Dense(n_dim, 64)

        self.dense1 = Dense(64, 2)
        self.dense2 = Dense(64, 2)
        self.dense3 = Dense(64, 2)
        self.dense4 = Dense(64, 2)
        
    def forward(self, x, labels1, labels2, labels3, labels4 ):
                
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


# In[ ]:


num_train = len(user_attribute_dict['train_id'])
# train_x, test_x = tensor_x[:num_train].clone(), tensor_x[num_train:].clone()
# train_y1, test_y1 = label1[:num_train].clone(), label1[num_train:].clone()
# train_y2, test_y2 = label2[:num_train].clone(), label2[num_train:].clone()
# train_y3, test_y3 = label3[:num_train].clone(), label3[num_train:].clone()
# train_y4, test_y4 = label4[:num_train].clone(), label4[num_train:].clone()

# train_x, test_x = tensor_x[:num_train].clone().detach().requires_grad_(True), tensor_x[num_train:].clone().detach().requires_grad_(True)
# train_y1, test_y1 = label1[:num_train].clone().detach().requires_grad_(True), label1[num_train:].clone().detach().requires_grad_(True)
# train_y2, test_y2 = label2[:num_train].clone().detach().requires_grad_(True), label2[num_train:].clone().detach().requires_grad_(True)
# train_y3, test_y3 = label3[:num_train].clone().detach().requires_grad_(True), label3[num_train:].clone().detach().requires_grad_(True)
# train_y4, test_y4 = label4[:num_train].clone().detach().requires_grad_(True), label4[num_train:].clone().detach().requires_grad_(True)

train_x, test_x = torch.tensor(full_x[:num_train]), torch.tensor(full_x[num_train:])
train_y1, test_y1 = torch.tensor(label1[:num_train]), torch.tensor(label1[num_train:])
train_y2, test_y2 = torch.tensor(label2[:num_train]), torch.tensor(label2[num_train:])
train_y3, test_y3 = torch.tensor(label3[:num_train]), torch.tensor(label3[num_train:])
train_y4, test_y4 = torch.tensor(label4[:num_train]), torch.tensor(label4[num_train:])


# In[ ]:


train_dataset = torch.utils.data.TensorDataset(train_x, train_y1, train_y2, train_y3, train_y4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 0)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y1, test_y2, test_y3, test_y4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 0)


# In[ ]:


def eval_data(model):
    global new_client, old_client

    loss_list = []
    y1_list, y2_list, y3_list, y4_list = [], [], [], []
    label1_list, label2_list, label3_list, label4_list = [], [], [], []
    with torch.no_grad():
        for x, l1, l2, l3, l4 in test_loader:
            loss, y1, y2, y3, y4, _ = model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))

            loss_list.append(loss.item())
            y1_list.append(y1.cpu().detach().numpy())
            y2_list.append(y2.cpu().detach().numpy())
            y3_list.append(y3.cpu().detach().numpy())
            y4_list.append(y4.cpu().detach().numpy())

            label1_list.append(l1.long().cpu().detach().numpy())
            label2_list.append(l2.long().cpu().detach().numpy())
            label3_list.append(l3.long().cpu().detach().numpy())
            label4_list.append(l4.long().cpu().detach().numpy())

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
    
    new_client_auc1 = roc_auc_score(labels1_np[new_client], y1_np[:, 1][new_client])
    new_client_auc2 = roc_auc_score(labels2_np[new_client], y2_np[:, 1][new_client])
    new_client_auc3 = roc_auc_score(labels3_np[new_client], y3_np[:, 1][new_client])
    new_client_auc4 = roc_auc_score(labels4_np[new_client], y4_np[:, 1][new_client])
    auc_new_client = [new_client_auc1, new_client_auc2, new_client_auc3, new_client_auc4]

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
    
epoch = 12
output_model = OutputLayer().to(device)
optimizer = AdamW(output_model.parameters(), lr = 0.001, weight_decay = 0)

for i in range(epoch):
    output_model.train()

    for x, l1, l2, l3, l4 in tqdm(train_loader):
        loss, y1, y2, y3, y4, _ = output_model(x.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device))
        optimizer.zero_grad()

#         loss.backward(retain_graph=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(output_model.parameters(), max_norm = 2)

        optimizer.step()
        
    output_model.eval()
    
#     train_ret_dict = eval_data(output_model)
    test_ret_dict = eval_data(output_model)
    df_ret = pd.DataFrame([
        test_ret_dict
    ], index = ['test'])
    ipd.display(df_ret)

