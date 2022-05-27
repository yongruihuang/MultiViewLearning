#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import pickle
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
from tqdm import tqdm_notebook as tqdm
import time
import IPython.display as ipd


# # df_master

# In[26]:


df_master_records = pd.read_csv('../data/202002/master_records.csv')
df_map_income = pd.read_csv('../data/202002/map_income.csv')
mp_income_number = {}
for income, real_money in zip(df_map_income['income'], df_map_income['近12个月月收入平均区间']):
    real_money = str(real_money)
    two_ends = real_money.split('-')
    min_money = float(two_ends[0].replace('+', ''))
    if len(two_ends) > 1:
        mp_income_number[income] = [min_money, two_ends[1]] 
    else:
        mp_income_number[income] = [min_money, min_money] 
df_income_map = pd.DataFrame(pd.DataFrame(mp_income_number).T)
df_income_map.columns = ['min_income', 'max_income']
df_master_records = pd.merge(df_master_records, df_income_map, how = 'left', left_on = 'income', right_index = True)

df_master_records['loan_date'] = pd.to_datetime(df_master_records['loan_date'], format='%Y%m%d', errors='ignore')
loan_sequence = pd.read_csv('../data/202002/loan_sequence.csv')
loan_sequence.index = loan_sequence.id
df_master_records.index = df_master_records.id
df_master_records['loan_sequence'] = loan_sequence['loan_sequence']

target_update_0514 = pd.read_csv('../data/202005/target_update_0514.csv')
target_update_0514.index = target_update_0514.id
df_master_records = pd.merge(df_master_records, target_update_0514, how = 'left', left_index = True, right_index = True)
df_master_records = df_master_records.drop(['id_x', 'id_y', 'Unnamed: 11'], axis = 1)
pickle.dump(df_master_records, open('../data_sortout/df_master_records.pickle', 'wb'))


# # app_install_list

# In[2]:


df_app_install_records = pd.read_csv('../data/202002/app_installed_records.csv', header=None, names=['id', 'loan_date', 'pkg', 'date'])
df_app_install_records['loan_date'] = pd.to_datetime(df_app_install_records['loan_date'], format='%Y%m%d', errors='ignore')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
pkg_id = le.fit_transform(df_app_install_records['pkg'])
df_app_install_records['pkg_id'] = pkg_id
df_app_install_records['pkg_id'] = df_app_install_records['pkg_id'] + 1


# In[12]:


pickle.dump(le.classes_, open('../data_sortout/mp_install_list_pkg_id2pkg.pickle', 'wb'))
logging.info('start se_id_install_list')
se_id_install_list = df_app_install_records.groupby('id').apply(lambda x : list(x['pkg_id']))
pickle.dump(se_id_install_list, open('../data_sortout/se_id_install_list.pickle', 'wb'))
logging.info('finish se_id_install_list')


# # app install behave

# In[2]:


df_install_behave = pd.read_csv('../data/202002/app_inunstall_records.csv', header=None, names=['id', 'loan_date', 'pkg', 'date', 'action_type'])
df_install_behave['loan_date'] = pd.to_datetime(df_install_behave['loan_date'], format='%Y%m%d', errors='ignore')
df_install_behave['date'] = pd.to_datetime(df_install_behave['date'], format='%Y%m%d', errors='ignore')
df_install_behave_sort = df_install_behave.sort_values(by = ['date'])


# In[3]:


df_install_behave_sort_less = df_install_behave_sort[df_install_behave_sort.loan_date > df_install_behave_sort.date]


# In[4]:


df_install_behave_sort_less.shape, df_install_behave_sort.shape


# In[5]:


len(set(df_install_behave_sort_less.id)), len(set(df_install_behave_sort.id)),


# In[6]:


df_install_behave_sort = df_install_behave_sort_less


# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
pkg_id = le.fit_transform(df_install_behave_sort['pkg'])
pickle.dump(le.classes_, open('../data_sortout/mp_install_behave_pkg_id2pkg.pickle', 'wb'))


# In[9]:


df_install_behave_sort['pkg_id'] = pkg_id + 1


# In[10]:


logging.info('start df_install_behave_sort')
df_install_behave = df_install_behave_sort.groupby('id').apply(lambda x : pd.Series({
    'pkg_id' : list(x['pkg_id'].values),
    'date' : list(x['date'].values),
    'action' : list(x['action_type'].map({'install': 1, 'unstall':0}) ),
}))
logging.info('finish df_install_behave_sort')


# In[5]:


df_install_behave


# In[11]:


logging.info('start dump df_install_behave_no_date')
pickle.dump(df_install_behave[['pkg_id', 'action']], open('../data_sortout/df_install_behave_no_date.pickle', 'wb'))
logging.info('finish dump df_install_behave_no_date')

logging.info('start dump df_install_behave')
pickle.dump(df_install_behave, open('../data_sortout/df_install_behave.pickle', 'wb'))
logging.info('finish dump df_install_behave')


# In[9]:


df_day = df_install_behave['date'].apply(lambda date_list : pd.Series({
    'year' : list(pd.DatetimeIndex(date_list).year - 2000),
    'month' : list(pd.DatetimeIndex(date_list).month),
    'day' : list(pd.DatetimeIndex(date_list).day),
}))
logging.info('start dump df_install_behave_date')
pickle.dump(df_day, open('../data_sortout/df_install_behave_date.pickle', 'wb'))
logging.info('finish dump df_install_behave_date')


# In[2]:


df_install_behave = pickle.load(open('../data_sortout/df_install_behave_no_date.pickle', 'rb'))
new_actions = []
for actions in tqdm( df_install_behave['action']):
    new_action = list(map(lambda x : x if x > 0 else 0, actions) )
    new_actions.append(new_action)
df_install_behave['action'] = new_actions


# In[4]:


pickle.dump(df_install_behave, open('../data_sortout/df_install_behave_no_date.pickle', 'wb'))


# ## time

# In[ ]:


df_install_behave = pd.read_csv('../data/202002/app_inunstall_records.csv', header=None, names=['id', 'loan_date', 'pkg', 'date', 'action_type'])
df_install_behave['loan_date'] = pd.to_datetime(df_install_behave['loan_date'], format='%Y%m%d', errors='ignore')
df_install_behave['date'] = pd.to_datetime(df_install_behave['date'], format='%Y%m%d', errors='ignore')
df_install_behave_sort = df_install_behave.sort_values(by = ['date'])
df_install_behave_sort_less = df_install_behave_sort[df_install_behave_sort.loan_date > df_install_behave_sort.date]


# In[21]:


qcut_time = pd.qcut(df_install_behave_sort_less['date'], 32)
cut_time = pd.cut(df_install_behave_sort_less['date'], 32)


# In[39]:


mp_qcut_id = dict(zip(list(qcut_time.value_counts().sort_index().index), list(range(32))))
qcut_time_id = qcut_time.map(mp_qcut_id)


# In[40]:


mp_cut_id = dict(zip(list(cut_time.value_counts().sort_index().index), list(range(32))))
cut_time_id = cut_time.map(mp_cut_id)


# In[45]:


df_install_behave_sort_less['qcut_time_id'] = qcut_time_id
df_install_behave_sort_less['cut_time_id'] = cut_time_id


# In[46]:


df_install_behave_sort_less


# In[49]:


df_time = df_install_behave_sort_less.groupby('id').apply(lambda x : pd.Series({
    'qcut_id' : list(x['qcut_time_id'].values),
    'cut_id' : list(x['cut_time_id'].values),
}))


# In[50]:


pickle.dump(df_time, open('../data_sortout/df_time_cut.pickle', 'wb'))


# In[51]:


df_time


# # App user log

# In[2]:


df_user_log = pickle.load(open('../data/202005/df_user_log.pickle','rb'))


# In[3]:


df_user_log


# In[7]:


df_user_log['page'].count() / df_user_log['page'].shape[0], df_user_log['tgt_event_id'].count() / df_user_log['tgt_event_id'].shape[0]


# In[3]:


df_user_log['time'] = pd.to_datetime(df_user_log['time'], format='%Y-%m-%d %H:%M:%S', errors='ignore')


# In[4]:


df_user_log = df_user_log.sort_values(by = ['time'])


# In[5]:


df_master_records = pickle.load(open('../data_sortout/df_master_records.pickle', 'rb'))


# In[6]:


df_user_log = df_user_log.fillna(0)


# In[7]:


gby_userid_log = df_user_log.groupby('userid')
mp_userid_log = {}
for userid, df in gby_userid_log:
    mp_userid_log[userid] = df


# In[8]:


df_master_records_with_log = df_master_records[df_master_records.userid.isin(set(df_user_log['userid']))]


# In[9]:


df_master_records_with_log.shape, df_master_records.shape


# In[10]:


df_user_log['page'] = df_user_log['page'].apply(int)
df_user_log['tgt_event_id'] = df_user_log['tgt_event_id'].apply(int)


# In[11]:


userlog_before_loan = []
useids_log = []
cnt = 0
for i in tqdm(range(df_master_records_with_log.shape[0])):
    se = df_master_records_with_log.iloc[i]
    userid = se['userid']
    loan_date = se['loan_date']
    df_item = mp_userid_log[userid].query('time <= @loan_date')
#     ipd.display(df_item)
    useids_log.append(userid)
    userlog_before_loan.append({
        'time' : df_item['time'].values,
        'session_id' : df_item['session_id'].values,
        'page' : df_item['page'].values,
        'tgt_event_id' : df_item['tgt_event_id'].values,
    })
    


# In[12]:


df_userlog_sequence = pd.DataFrame(userlog_before_loan, index = df_master_records_with_log.index)


# In[13]:


pickle.dump(df_userlog_sequence, open('../data_sortout/df_userlog_sequence.pickle', 'wb'))


# In[14]:


df_userlog_sequence = df_userlog_sequence[df_userlog_sequence['time'].apply(len) > 0]
pickle.dump(df_userlog_sequence[['page', 'tgt_event_id']], open('../data_sortout/df_userlog_sequence_less.pickle', 'wb'))


# ## Time

# In[2]:


df_userlog_sequence = pickle.load(open('../data_sortout/df_userlog_sequence.pickle', 'rb'))


# In[7]:


df_master_records = pickle.load(open('../data_sortout/df_master_records.pickle', 'rb'))


# In[22]:


df_userlog_sequence_data = df_userlog_sequence[df_userlog_sequence['time'].apply(len) > 0]
df_userlog_sequence_with_loan_date = pd.merge(df_userlog_sequence_data, df_master_records[['loan_date']], how = 'left', left_index=True, right_index=True)


# In[111]:


day_list, second_list, id_list = [], [], []
for i in tqdm(range(df_userlog_sequence_with_loan_date.shape[0])):
    a_row = df_userlog_sequence_with_loan_date.iloc[i]
    time_list = list(map(lambda x : ((a_row['loan_date'] - x).seconds, (a_row['loan_date'] - x).days), a_row['time']))
    time_list = list(zip(*time_list))
    second_list.extend(time_list[0])
    day_list.extend(time_list[1])
    id_list.extend([a_row.name] * len(time_list[1]))


# In[112]:


def cut(se_time, cut_piece):
    qcut_time = pd.qcut(se_time, cut_piece)
    mp_qcut_id = dict(zip(list(qcut_time.value_counts().sort_index().index), list(range(cut_piece))))
    qcut_time_id = qcut_time.map(mp_qcut_id)
    
    cut_time = pd.cut(se_time, cut_piece)
    mp_cut_id = dict(zip(list(cut_time.value_counts().sort_index().index), list(range(cut_piece))))
    cut_time_id = cut_time.map(mp_cut_id)
    
    return qcut_time_id, cut_time_id

df_time = pd.DataFrame({'day' :day_list, 'second' : second_list, 'id' : id_list})

qcut_day_id, cut_day_id = cut(df_time['day'], 8)
qcut_second_id, cut_second_id = cut(df_time['second'], 32)
df_time['qcut_day_id'] = qcut_day_id
df_time['cut_day_id'] = cut_day_id
df_time['qcut_second_id'] = qcut_second_id
df_time['cut_second_id'] = cut_second_id


# In[113]:


df_time_seq = df_time.groupby('id').apply(lambda x : pd.Series({
    'qcut_day_id' : list(x['qcut_day_id']),
    'cut_day_id' : list(x['cut_day_id']),
    'qcut_second_id' : list(x['qcut_second_id']),
    'cut_second_id' : list(x['cut_second_id']),
}))


# In[116]:


# df_time_seq


# In[114]:


pickle.dump(df_time_seq, open('../data_sortout/df_userlog_time_seq.pickle', 'wb'))


# ## cross

# In[2]:


df_userlog_sequence = pickle.load(open('../data_sortout/df_userlog_sequence_less.pickle', 'rb'))


# In[3]:


cross_list = []
for page_list, tgt_list in tqdm(zip(df_userlog_sequence['page'], df_userlog_sequence['tgt_event_id'])):
    cross_list.append([str(int(page))+ '_' + str(int(tgt)) for (page, tgt) in zip(page_list, tgt_list)])
se_userlog_cross = pd.Series(cross_list)


# In[4]:


se_userlog_cross.index = df_userlog_sequence.index
pickle.dump(se_userlog_cross, open('../data_sortout/se_userlog_cross.pickle', 'wb'))


# In[32]:


se_userlog_cross = pickle.load(open('../data_sortout/se_userlog_cross.pickle', 'rb'))


# In[5]:


mp_cross_id = {}
cnt = 1
for i in range(se_userlog_cross.shape[0]):
    user_log_list = se_userlog_cross.iloc[i]
    for action in user_log_list: 
        if (action not in mp_cross_id):
            mp_cross_id[action] = cnt
            cnt += 1
    if(cnt == 138):
        break
#     print(user_log_list)
#     break


# In[6]:


se_userlog_cross_id = se_userlog_cross.apply(lambda x : list(map(lambda item : mp_cross_id[item], x)))
pickle.dump(se_userlog_cross_id, open('../data_sortout/se_userlog_cross_id.pickle', 'wb'))


# In[37]:


se_userlog_cross.apply(len).sum()


# In[33]:


se_userlog_cross


# ## load

# In[2]:


df_userlog_sequence = pickle.load(open('../data_sortout/df_userlog_sequence_less.pickle', 'rb'))


# In[5]:


pad_start_session = []
userids = []
start_token = '#'
for i in tqdm(range(df_userlog_sequence.shape[0])):
    se_item = df_userlog_sequence.iloc[i]
    
    session_ids = se_item['session_id']
    
    if(len(session_ids) == 0):
        continue
        
    time_list, page_list, tgt_list = [start_token], [start_token], [start_token]
    for j in range(len(session_ids)):
        if (j > 0 and session_ids[j] != session_ids[j-1]):
            time_list.append(start_token)
            page_list.append(start_token)
            tgt_list.append(start_token)
        time_list.append(se_item['time'][j])
        page_list.append(se_item['page'][j])
        tgt_list.append(se_item['tgt_event_id'][j])

    pad_start_session.append({
        'time' : time_list,
        'page' : page_list,
        'tgt' : tgt_list,
    })
    userids.append(se_item.name)


# In[ ]:


df_userlog_sequence_start_token = pd.DataFrame(pad_start_session, index = userids)
pickle.dump(df_userlog_sequence_start_token, open('../data_sortout/df_userlog_sequence_start_token.pickle', 'wb'))


# In[7]:


df_userlog_sequence_start_token


# In[ ]:




