# For multi-site mutation benchmark
# The blind test set only contains multi-site mutation sequences
import random
import numpy as np
import pandas as pd
import pickle

random.seed(27)

# get train and test index
df=pd.read_csv('variant_data.csv')
p_mask=np.where((df['KD_ratio']>1).values)[0].tolist()
n_mask=np.where((df['KD_ratio']<1).values)[0].tolist()

p_idx=random.sample(p_mask,300)
n_idx=random.sample(n_mask,300)
chosen_idx=p_idx+n_idx

print(len(chosen_idx))
np.save('test_index.npy',chosen_idx)



idx=np.load('test_index.npy')

mask_train=np.zeros(len(df))
mask_train[idx]=1
mask_train=(1-mask_train).astype(bool)

mask_train=mask_train&(1-df['KD_ratio'].isna().values)
mask_train=mask_train&(df['KD_ratio'].values!=1)

mask_train=np.where(mask_train)[0]

print(mask_train.shape)
np.save('train_index.npy',mask_train)




# get train and test data

train_idx=np.load('train_index.npy') #idx is related to the raw csv
test_idx=np.load('test_index.npy')


#85 backbone
train1_data=np.load('../../85_backbone/data/4KwPtest_train_85_embedding_data.npy')
train1_label=np.load('../../85_backbone/data/4KwPtest_train_85_embedding_label.npy')

train2_data=np.load('../../85_backbone/data/pnas_test_85_embedding_data.npy')
train2_label=np.load('../../85_backbone/data/pnas_test_85_embedding_label.npy')

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values

train3_data=open('../../1212test/variant_data.csv_embedding.pickle','rb')
train3_data=pickle.load(train3_data)
train3_data=np.array(train3_data)

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train2_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train2_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('85/85_embedding_train_data.npy',train_data)
np.save('85/85_embedding_train_label.npy',train_label)
np.save('85/85_embedding_test_data.npy',test_data)
np.save('85/85_embedding_test_label.npy',test_label)


# other backbones
'''
#esm1b
train1_data=np.load('../../esm2/1b/data/4KwPtest_train_esm1b_embedding_data.npy')
train1_label=np.load('../../esm2/1b/data/4KwPtest_train_esm1b_embedding_label.npy')

train2_data=np.load('../../esm2/1b/data/pnas_test_esm1b_embedding_data.npy')
train2_label=np.load('../../esm2/1b/data/pnas_test_esm1b_embedding_label.npy')

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values

train3_data=np.load('esm1b/esm1b_variant_embedding_all_data.npy')

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train2_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train2_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('esm1b/esm1b_embedding_train_data.npy',train_data)
np.save('esm1b/esm1b_embedding_train_label.npy',train_label)
np.save('esm1b/esm1b_embedding_test_data.npy',test_data)
np.save('esm1b/esm1b_embedding_test_label.npy',test_label)
'''

'''
#t5
print('t5')
train1_data=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_data_T5-XL-UNI.npy')
train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy').flatten()

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('t5/t5_variant_embedding_all_data.npy')

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('t5/t5_embedding_train_data.npy',train_data)
np.save('t5/t5_embedding_train_label.npy',train_label)
np.save('t5/t5_embedding_test_data.npy',test_data)
np.save('t5/t5_embedding_test_label.npy',test_label)
'''

'''
#esm1v1
name='esm1v1'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''



'''
#esm1v2
name='esm1v2'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''


'''
#esm1v3
name='esm1v3'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''

'''
#esm1v4
name='esm1v4'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''


'''
#esm1v5
name='esm1v5'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)

del train_data,train_label,test_data,test_label
'''

'''
#esm2
name='esm2'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''

'''
#esm215b
name='esm215b'
print(name)
train1_data=np.load('{}/{}_4ksel_embedding_all_data.npy'.format(name,name))
#train1_label=np.load('../../finetune_predict_cpp_update/_embedding_train/data/single_embedding_label_T5-XL-UNI.npy')
train1_label=pd.read_csv('sel_4003.csv')['KD_ratio'].values.flatten()
assert len(train1_data)==len(train1_label)

df=pd.read_csv('variant_data.csv')
train3_label=df['KD_ratio'].values.flatten()

train3_data=np.load('{}/{}_variant_embedding_all_data.npy'.format(name,name))

test_data=train3_data[test_idx]
test_label=train3_label[test_idx]

train3_data=train3_data[train_idx]
train3_label=train3_label[train_idx]

print('train3 data',train3_data.shape)

train_data=np.concatenate([train1_data,train3_data],axis=0)
train_label=np.concatenate([train1_label,train3_label],axis=0)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

np.save('{}/{}_embedding_train_data.npy'.format(name,name),train_data)
np.save('{}/{}_embedding_train_label.npy'.format(name,name),train_label)
np.save('{}/{}_embedding_test_data.npy'.format(name,name),test_data)
np.save('{}/{}_embedding_test_label.npy'.format(name,name),test_label)
'''