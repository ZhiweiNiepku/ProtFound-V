# create training dataset for escape training and prediction
import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm


# load feature extracted by ProtFound
f=open('RBD_sequence.csv_embedding.pickle','rb')
RBD_emb=np.array(pickle.load(f)).astype(np.float16)

f=open('antibody_heavy_sequence.csv_embedding.pickle','rb')
HSEQ_emb=np.array(pickle.load(f)).astype(np.float16)

f=open('antibody_light_sequence.csv_embedding.pickle','rb')
LSEQ_emb=np.array(pickle.load(f)).astype(np.float16)


RBD_df=pd.read_csv('RBD_sequences.csv')
HSEQ_df=pd.read_csv('antibody_heavy_sequence.csv')
LSEQ_df=pd.read_csv('antibody_light_sequence.csv')

# we split the dataset into 11 dataframes due to the memory limit

# the sequence order in dataframes is the same as the order in pickles
def get_RBD_emb(site,mutat):
    tp_=RBD_df[(RBD_df['site']==site)&(RBD_df['mutation']==mutat)].index
    assert len(tp_)==1
    return RBD_emb[tp_[0],:,:]

def get_HSEQ_emb(name):
    tp_=HSEQ_df[HSEQ_df['name']==name].index
    assert len(tp_)==1
    return HSEQ_emb[tp_[0],:,:]

def get_LSEQ_emb(name):
    tp_=LSEQ_df[LSEQ_df['name']==name].index
    assert len(tp_)==1
    return LSEQ_emb[tp_[0],:,:]

adds_=['train-0','train-1','train-2','train-3','train-4',
      'train-5','train-6','train-7','train-8','train-9','val']

for add_ in adds_:
    df=pd.read_csv('_raw/{}.csv'.format(add_))

    sites=df['site']
    mutations=df['mutation']
    antibodies=df['antibody']
    label=df['mut_escape']

    label_all=[]
    RBD_embedding_all=[]
    HSEQ_embedding_all=[]
    LSEQ_embedding_all=[]

    for i in tqdm(range(len(df))):
        RBD_embedding_all.append(get_RBD_emb(sites[i],mutations[i]))
        HSEQ_embedding_all.append(get_HSEQ_emb(antibodies[i]))
        LSEQ_embedding_all.append(get_LSEQ_emb(antibodies[i]))
        label_all.append(label[i])


    print(len(RBD_embedding_all))
    print(RBD_embedding_all[0].shape)
    print(len(HSEQ_embedding_all))
    print(HSEQ_embedding_all[0].shape)
    print(len(LSEQ_embedding_all))
    print(LSEQ_embedding_all[0].shape)
    print(len(label_all))

    np.save('RBD_embedding_{}.npy'.format(add_),RBD_embedding_all)
    np.save('HSEQ_embedding_{}.npy'.format(add_),HSEQ_embedding_all)
    np.save('LSEQ_embedding_{}.npy'.format(add_),LSEQ_embedding_all)
    np.save('label_{}.npy'.format(add_),label_all)
