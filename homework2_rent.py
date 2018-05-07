#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24  2017
@author: KunyiLiu & ZhifuXiao
"""

# import library
import numpy as np
import pandas as pd
import io
import fancyimpute
from scipy import stats
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression, Lasso, RidgeCV, LassoCV

class TestClass:
  def preprocessing(self):
    # load data from url
    rent=pd.read_csv('https://ndownloader.figshare.com/files/7586326',sep=',')
    
    # drop data with person
    trial = rent.copy()
    trial = trial.drop(trial.columns[[30,31,32,33,34,35,36,37,38,39,40,41,42,43,47,119,120,121,122,123,131,132,133,134,135,140,141,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,163,164,165,166,167,168,169,170,171,172,173,174,175]], axis=1)
    
    # handle missing value in different columns
    columns=['sc118','sc140','sc143','sc147','sc173','sc171','sc541','sc184','sc542','sc543','sc544']
    for i in columns:
      trial[i] = trial[i].replace(3, np.NaN)
    columns=['sc197','sc114']
    for i in columns:
      trial[i] = trial[i].replace(4, np.NaN)
    columns=['sc571','sc189']
    for i in columns:
        trial[i] = trial[i].replace(5, np.NaN)
    columns=['sc181','rec54']
    for i in columns:
        trial[i] = trial[i].replace(7, np.NaN)
    columns=['rec21','sc541','sc184','sc542','sc543','sc544','sc185','sc186','sc197','sc198','sc187','sc188','sc571','sc189','sc190','sc191','sc192','uf1_1','uf1_2','uf1_3','uf1_4','uf1_5','uf1_6','uf1_7','uf1_8','uf1_9','uf1_10','uf1_11','uf1_12','uf1_13','uf1_14','uf1_15','uf1_16','uf1_17','uf1_18','uf1_19','uf1_20','uf1_21','uf1_22','uf1_35','sc23','sc24','sc36','sc37','sc38','sc120','sc121','sc125','sc140','sc141','sc143','sc144','sc147','sc173','sc171','sc154','sc157','sc174','sc181','sc193','sc194','sc196','sc548','sc549','sc550','sc551','sc199','sc575']
    for i in columns:
        trial[i] = trial[i].replace(8, np.NaN)
    column=['rec53','sc541','sc184','sc542','sc543','sc544','sc115','sc116','sc120','sc121','sc125','sc127','sc140','sc141','sc143','sc144','sc173','sc153','sc154','sc156','sc157','sc181']
    for i in column:
        trial[i] = trial[i].replace(9, np.NaN)
    column=['rec15']
    for i in column:
        trial[i] = trial[i].replace(10, np.NaN)
    column=['rec15']
    for i in column:
        trial[i] = trial[i].replace(11, np.NaN)
    column=['sc27','uf9','uf10']
    for i in column:
        trial[i] = trial[i].replace(98, np.NaN)
    column=['uf9','uf10']
    for i in column:
        trial[i] = trial[i].replace(99, np.NaN)
    column=['sc134','uf7a','uf8','uf64']
    for i in column:
        trial[i] = trial[i].replace(9998, np.NaN)
    column=['uf28','uf27','sc134','uf7a','uf8','uf12','uf13','uf14','uf15','uf64']
    for i in column:
        trial[i] = trial[i].replace(9999, np.NaN)
    column=['uf7','uf17a']
    for i in column:
        trial[i] = trial[i].replace(99998, np.NaN)
    column=['uf7','uf17a','uf16','uf17']
    for i in column:
        trial[i] = trial[i].replace(99999, np.NaN)
    column=['uf5','uf6']
    for i in column:
        trial[i] = trial[i].replace(9999998, np.NaN)
    column=['uf5','uf6']
    for i in column:
        trial[i] = trial[i].replace(9999999, np.NaN)

    # drop columns with more than half na
    trial=trial.dropna(thresh=len(trial)/2,axis=1)

    #drop the observations with na in objective feature
    feature_missing = np.isnan(trial.loc[:, 'uf17'])
    trial=trial.loc[~feature_missing, :]

    #objective columns
    X=trial.ix[:,trial.columns!='uf17']
    y=trial['uf17']
    contin_cols = ['uf5','uf6','uf7','sc134','uf7a','uf8','uf12','uf13','uf14','uf15','uf17a','uf26','uf28','uf27']
    contin_cols = [x for x in contin_cols if x in list(trial.columns)] 
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test, contin_cols
  
  def score_rent(self):
    X_train, X_test, y_train, y_test, contin_cols = self.preprocessing()
    #seperate continous and categorical columns

    con_train = X_train[contin_cols]
    cat_train = X_train[X_train.columns.difference(contin_cols)]
    
    #fancyimpute mice in continous train data
    mice = fancyimpute.MICE(verbose=0)
    con_train=np.asarray(con_train)
    con_train_mice = mice.complete(con_train)

    #fancyimpute mice in categorical train data
    cat_train=np.asarray(cat_train)
    cat_train_fancyknn = fancyimpute.KNN().complete(cat_train)
    cat_train_fancyknn = np.round(cat_train_fancyknn).astype(int)

    #apply boxcox transformation to continuous train data
    con_train_mice_bc=np.empty(con_train_mice.shape)
    from scipy import stats
    for i in range(len(contin_cols)):
      if np.argwhere(con_train_mice[:,i]<0).size==0:
        x = stats.boxcox(con_train_mice[:,i]+ 1e-5)[0]
        x = np.asarray([x])
        con_train_mice_bc[:,i]=x
      else:
        con_train_mice_bc[:,i]=con_train_mice[:,i]

    # apply onehot to categorical train data
    enc = OneHotEncoder()
    enc=enc.fit(cat_train_fancyknn)
    oh = enc.transform(cat_train_fancyknn).toarray()
    cat_train_fancyknn_onehot = np.round(oh).astype(int)

    #concatenate imputed train data
    X_train_imp=np.concatenate((cat_train_fancyknn_onehot, con_train_mice_bc), axis=1)
    
    # Feature selection using Lasso
    select_lassocv = SelectFromModel(LassoCV())
    select_lassocv = select_lassocv.fit(X_train_imp, y_train)

    #LassoCV
    param_grid = {'alpha': np.logspace(-3, 0, 14)}
    print(param_grid)
    grid = GridSearchCV(Lasso(normalize=True, max_iter=1e6), param_grid, cv=10)

    #makepipeline to prevent information leakage
    pipe_lassocv = make_pipeline(MinMaxScaler(), select_lassocv, grid)
    pipe_lassocv = pipe_lassocv.fit(X_train_imp, y_train)
    train_r2 = np.mean(cross_val_score(pipe_lassocv, X_train_imp, y_train, cv=5))
    return contin_cols, enc, pipe_lassocv, train_r2, X_test, y_test
  
  def predict_rent(self):
    contin_cols, enc, pipe_lassocv,train_r2, X_test,y_test = self.score_rent()
    con_test = X_test[contin_cols]
    cat_test = X_test[X_test.columns.difference(contin_cols)]
    
    #impute test data respectively (continous data)
    mice = fancyimpute.MICE(verbose=0)
    con_test = np.asarray(con_test)
    con_test_mice = mice.complete(con_test)
    
    #categorical data
    cat_test = np.asarray(cat_test)
    cat_test_fancyknn = fancyimpute.KNN().complete(cat_test)
    cat_test_fancyknn = np.round(cat_test_fancyknn).astype(int)
    
    #apply boxcox transformation to continuous test data
    con_test_mice_bc=np.empty(con_test_mice.shape)
    from scipy import stats
    for i in range(len(contin_cols)):
      if np.argwhere(con_test_mice[:,i]<0).size==0:
        x=stats.boxcox(con_test_mice[:,i]+ 1e-5)[0]
        x=np.asarray([x])
        con_test_mice_bc[:,i]=x
      else:
        con_test_mice_bc[:,i]=con_test_mice[:,i]
        
    # apply onehot to categorical train data
    oh = enc.transform(cat_test_fancyknn).toarray()
    cat_test_fancyknn_onehot = np.round(oh).astype(int)
    print("Finished onehot")
    #concatenate imputed test data
    X_test_imp=np.concatenate((cat_test_fancyknn_onehot, con_test_mice_bc), axis=1)
    
    # make prediction based on training model
    y_pred = pipe_lassocv.predict(X_test_imp)
    test_r2 = r2_score(y_test,y_pred)
    print(test_r2)
    return X_test, y_test, y_pred
