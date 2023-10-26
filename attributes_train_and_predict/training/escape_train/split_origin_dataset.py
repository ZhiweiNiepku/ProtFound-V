import random
import numpy as np
import pandas as pd

# we split the dataset into 11 csvs due to the memory limit, including 10 csvs for training and 1 for validation

# dataset from *Imprinted SARS-CoV-2 humoral immunity induces convergent Omicron RBD evolution*
raw=pd.read_csv('use_res_clean.csv')

if 1:
   # split training set and test set
   test_idx=np.array(random.sample(np.arange(len(raw)),9600))
   train_idx=[]
   for i in range(len(raw)):
      if i not in test_idx:
         train_idx.append(i)
   train_idx=np.array(train_idx)

   np.save('data/train_index.npy',train_idx)
   np.save('data/test_index.npy',test_idx)


train_idx=np.load('data/train_index.npy')
train_idx=np.array(random.sample(train_idx.tolist(),len(train_idx)))
raw=raw.iloc[train_idx]
print(raw.shape)
print(raw.head())
raw.to_csv('_raw/raw_train.csv',index=False)

# we split training and validation set in advance due to the memory limit
# valid index
df=pd.read_csv('_raw/raw_train.csv')
label=df['mut_escape'].values
pos_val_idx=random.sample(np.where(label>0.4)[0].tolist(),10500)
neg_val_idx=random.sample(np.where(label<0.4)[0].tolist(),10500)
val_idx=pos_val_idx+neg_val_idx
val_idx.sort()
np.save('val_index.npy',val_idx)

# train index
val_idx=np.load('val_index.npy')
val_label=label[val_idx]
train_idx=[]
for i in range(len(df)):
   if i not in val_idx:
       train_idx.append(i)
print(len(train_idx))
np.save('train_index.npy',train_idx)


val_idx=np.load('val_index.npy')
df_val=df.iloc[val_idx]
print(df_val.shape)
df_val.to_csv('_raw/val.csv',index=False)


train_idx=np.load('train_index.npy')
train_idx=np.random.permutation(train_idx)
file_n=10
pernum=len(train_idx)//file_n
for i in range(file_n):
   tp_=df.iloc[np.array(train_idx)[i*pernum:i*pernum+pernum]]
   tp_.to_csv('_raw/train-{}.csv'.format(i),index=False)
   print(tp_.shape)
