# data process for the training and prediction of bind, expr
import pandas as pd
import numpy as np

import random
import pickle

###################################################################
# pickle to npy

# bind data
if 1:
    df_bind=pd.read_csv('bind_variant_data.csv')
    # only use samples of BA.1 and BA.2
    mask_bind=(df_bind['target']=='Omicron_BA1')|(df_bind['target']=='Omicron_BA2')
    label_bind=df_bind[mask_bind]['bind_ratio'].values
    # load features extracted by ProtFound
    feature_bind=pickle.load(open('omicron_rbd_sequence.txt_embedding.pickle','rb'))
    feature_bind=np.array(feature_bind)

    assert len(feature_bind)==len(label_bind)

    mask=np.isnan(label_bind)
    feature_bind=feature_bind[~mask]
    label_bind=label_bind[~mask]

    print(feature_bind.shape)
    print(label_bind.shape)
    np.save('all_bind_data.npy',feature_bind)
    np.save('all_bind_label.npy',label_bind)

    del feature_bind

# expr data
if 1:
    df_expr=pd.read_csv('expr_variant_data.csv')
    # only use samples of BA.1 and BA.2
    mask_expr=(df_expr['target']=='Omicron_BA1')|(df_expr['target']=='Omicron_BA2')
    label_expr=df_expr[mask_expr]['expr_ratio'].values
    # load features extracted by ProtFound
    feature_expr=pickle.load(open('omicron_rbd_sequence.txt_embedding.pickle','rb'))
    feature_expr=np.array(feature_expr)

    assert len(feature_expr)==len(label_expr)

    mask=np.isnan(label_expr)
    feature_expr=feature_expr[~mask]
    label_expr=label_expr[~mask]

    print(feature_expr.shape)
    print(label_expr.shape)
    np.save('all_expr_data.npy',feature_expr)
    np.save('all_expr_label.npy',label_expr)

###################################################################


###################################################################
# train test data split

if 1:
    #bind
    label=np.load('all_bind_label.npy')
    mask_=np.where(np.isnan(label))[0]
    pidx=random.sample(np.where(label>=1)[0].tolist(),79)
    nidx=random.sample(np.where(label<1)[0].tolist(),79)
    test_idx=np.array(pidx+nidx)
    train_idx=[]
    for i in range(len(label)):
        if i not in test_idx and i not in mask_:
            train_idx.append(i)
    train_idx=np.array(train_idx)

    print(train_idx.shape)
    print(test_idx.shape)
    np.save('bind_train_idx.npy',train_idx)
    np.save('bind_test_idx.npy',test_idx)

    #expr
    label=np.load('all_expr_label.npy')
    mask_=np.where(np.isnan(label))[0]
    pidx=random.sample(np.where(label>=1)[0].tolist(),90)
    nidx=random.sample(np.where(label<1)[0].tolist(),90)
    test_idx=np.array(pidx+nidx)
    train_idx=[]
    for i in range(len(label)):
        if i not in test_idx and i not in mask_:
            train_idx.append(i)
    train_idx=np.array(train_idx)

    print(train_idx.shape)
    print(test_idx.shape)
    np.save('expr_train_idx.npy',train_idx)
    np.save('expr_test_idx.npy',test_idx)


if 1:
    feature_all=np.load('all_bind_data.npy')
    label_all=np.load('all_bind_label.npy')
    print(feature_all.shape)
    train_idx=np.load('bind_train_idx.npy')
    test_idx=np.load('bind_test_idx.npy')
    train_feature=feature_all[train_idx]
    train_label=label_all[train_idx]

    test_feature=feature_all[test_idx]
    test_label=label_all[test_idx]

    print(train_feature.shape)
    print(train_label.shape)
    print(test_feature.shape)
    print(test_label.shape)

    np.save('bind_train_data.npy',train_feature)
    np.save('bind_train_label.npy',train_label)
    np.save('bind_test_data.npy',test_feature)
    np.save('bind_test_label.npy',test_label)



    feature_all=np.load('all_expr_data.npy')
    label_all=np.load('all_expr_label.npy')
    print(feature_all.shape)
    train_idx=np.load('expr_train_idx.npy')
    test_idx=np.load('expr_test_idx.npy')
    train_feature=feature_all[train_idx]
    train_label=label_all[train_idx]

    test_feature=feature_all[test_idx]
    test_label=label_all[test_idx]

    print(train_feature.shape)
    print(train_label.shape)
    print(test_feature.shape)
    print(test_label.shape)

    np.save('expr_train_data.npy',train_feature)
    np.save('expr_train_label.npy',train_label)
    np.save('expr_test_data.npy',test_feature)
    np.save('expr_test_label.npy',test_label)
