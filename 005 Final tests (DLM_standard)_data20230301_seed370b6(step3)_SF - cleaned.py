# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:07:51 2021

@author: Junjie Zhu
"""

import numpy  as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = pd.DataFrame(np.array(y_true)), pd.DataFrame(np.array(y_pred))
    df = pd.concat([y_true, y_pred], axis = 1)
    df.columns = ['true', 'pred']
    df2 = df[df['true'] != 0]
    print(len(df2))
    MAPE = np.mean(np.abs((df2.iloc[:,0] - df2.iloc[:,1]) / df2.iloc[:,0])) * 100
    return MAPE

def perf(y_train, y_pred_tr, y_test, y_pred_te):
    R2_tr = metrics.r2_score(y_train, y_pred_tr)
    MAPE_tr = mean_absolute_percentage_error(y_train,y_pred_tr)  
    MAE_tr = metrics.mean_absolute_error(y_train, y_pred_tr)
    RMSE_tr = np.sqrt(metrics.mean_squared_error(y_train, y_pred_tr))
    
    R2_te = metrics.r2_score(y_test, y_pred_te)
    MAPE_te = mean_absolute_percentage_error(y_test,y_pred_te) 
    MAE_te = metrics.mean_absolute_error(y_test, y_pred_te)
    RMSE_te = np.sqrt(metrics.mean_squared_error(y_test, y_pred_te))
    
    Perf = np.array([[R2_tr, RMSE_tr, MAPE_tr, MAE_tr, R2_te, RMSE_te, MAPE_te, MAE_te]])
    Perf2 = np.array([[R2_tr, RMSE_tr, MAPE_tr, MAE_tr], [R2_te, RMSE_te, MAPE_te, MAE_te]]).T
    
    return Perf, Perf2

def rp_nanval(train, test, arg):
    lst = list(train.iloc[:, :-1])
    train2 = train.copy(deep = True)
    test2 = test.copy(deep = True)
    if arg == 'mean':
        for c in lst:
            train2[c].fillna(value=train2[c].mean(), inplace=True)
            test2[c].fillna(value=train2[c].mean(), inplace=True)            
    elif arg == 'median':
        for c in lst:
            train2[c].fillna(value=train2[c].median(), inplace=True)
            test2[c].fillna(value=train2[c].median(), inplace=True)             
    elif arg == 'mode':
        for c in lst:
            train2[c].fillna(value=train2[c].mode()[0], inplace=True)
            test2[c].fillna(value=train2[c].mode()[0], inplace=True)            
    else:
        print('not correctly assigned!!!')      
 
    return train2, test2


# split training and testing data set
#########data leakage management

train00 = pd.read_csv('Preprocessed train (data20230301)_DLM370_0.2b6_SF.csv')
test00 = pd.read_csv('Preprocessed test (data20230301)_DLM370_0.2b6_SF.csv')

train01 = train00.drop(['NO'], axis = 1)
test01 = test00.drop(['NO'], axis = 1)
 
train0a, test0a = train01.iloc[:, 1:-1024], test01.iloc[:, 1:-1024]
train0b, test0b = train01.iloc[:, -1024:], test01.iloc[:, -1024:]

###############################################################################
traina, testa = rp_nanval(train0a, test0a, 'mean')

pca = PCA()
pca.fit_transform(train0b)
evr = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
criterion = 95
n_cp = len(evr[evr<criterion])+1

lst_bitnames = ['PC'+str(e+1) for e in range(n_cp)]     
x_trainb = pd.DataFrame(pca.transform(train0b)[:,:n_cp])  
x_testb = pd.DataFrame(pca.transform(test0b)[:,:n_cp])
x_trainb.columns = x_testb.columns = lst_bitnames

x_traina = traina.drop(columns=['Separation_factor'])
y_train0 = traina['Separation_factor']

x_testa = testa.drop(columns=['Separation_factor'])
y_test0 = testa['Separation_factor']


y_train1 = 10**(y_train0)
y_test1 = 10**(y_test0)

x_train = pd.concat([x_traina,x_trainb], axis = 1)
x_test = pd.concat([x_testa,x_testb], axis = 1)


# standardize data
scale = StandardScaler()
scale.fit(x_train)
x_train2 = scale.transform(x_train)
x_test2 = scale.transform(x_test)


##################################################    
#############################################################################################
model = LGBMRegressor(n_estimators=400, max_depth=12, min_child_samples=5, learning_rate=0.05,
                      reg_lambda=3, feature_fraction=0.4, num_leaves=30)

#############################################################################################
###10 replicates - mean performance #####################
Perf_mean = pd.Series([np.nan]*8)
try:
    n = 1
    Perfs = []
    lst_pred_tr = []
    lst_pred_te = []
    for i in range(n):  
        model.fit(x_train2, np.ravel(y_train0))
        
        y_pred_tr = model.predict(x_train2)
        y_pred_te = model.predict(x_test2)
        
        Perf, Perf2 = perf(y_train0, y_pred_tr, y_test0, y_pred_te)
        Perf_lst = list(Perf[0])
        
        Perfs.append(Perf_lst)
        lst_pred_tr.append(y_pred_tr)
        lst_pred_te.append(y_pred_te)
    
    Perfs2 = pd.DataFrame(Perfs)
    Perf_mean = Perfs2.mean()
except:
    pass


y_pred_tr2 = pd.DataFrame(np.array(lst_pred_tr).mean(axis = 0))
y_pred_te2 = pd.DataFrame(np.array(lst_pred_te).mean(axis = 0))


###########
y_pred_tr2.columns = ['predicted']
y_pred_te2.columns = ['predicted']

x_train3 = pd.DataFrame(x_train2)
x_test3 = pd.DataFrame(x_test2)
x_train3.columns = x_test3.columns = list(x_train)

train_fn = pd.concat([x_train3, y_train0, y_pred_tr2], axis = 1)
test_fn = pd.concat([x_test3, y_test0, y_pred_te2], axis = 1)

# train_fn.to_csv('Final train (data20230301)_DLM370_0.2b6_filled_lgm_normalized.csv', index=False)
# test_fn.to_csv('Final test (data20230301)_DLM370_0.2b6_filled_lgm_normalized.csv', index=False)


y_pred_tr3 = 10**(y_pred_tr2)
y_pred_te3 = 10**(y_pred_te2)

train_fn0 = pd.concat([x_train, y_train1, y_pred_tr3], axis = 1)
test_fn0 = pd.concat([x_test, y_test1, y_pred_te3], axis = 1)

# train_fn0.to_csv('Final train (data20230301)_DLM370_0.2b6_filled_lgm.csv', index=False)
# test_fn0.to_csv('Final test (data20230301)_DLM370_0.2b6_filled_lgm.csv', index=False)


#########################
train_fn1 = pd.concat([train00[['NO','DOI']],train_fn0], axis = 1)
test_fn1 = pd.concat([test00[['NO','DOI']],test_fn0], axis = 1)

# train_fn1.to_csv('Final train (data20230301)_DLM370_0.2b6_filled_lgm_NO.csv', index=False)
# test_fn1.to_csv('Final test (data20230301)_DLM370_0.2b6_filled_lgm_NO.csv', index=False)


