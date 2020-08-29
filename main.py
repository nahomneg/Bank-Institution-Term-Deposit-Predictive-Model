# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 03:55:18 2020

@author: Nahom Negussie
"""
import pandas as pd

import data
from model import Model
from sklearn import preprocessing

data_frame = pd.read_csv("dataset/bank-additional-full.csv",sep=';'  , engine='python')
processor = data.PreProcessing(data_frame)
model = Model()

#processor.detect_outliers_boxplot(['duration','age','cons.conf.idx'])
#processor.handle_outliers(['duration','age','cons.conf.idx'])
#processor.detect_outliers_boxplot(['duration','age','cons.conf.idx'])

#processor.plot_multiple_categorical_against_target(['poutcome','marital','loan','housing'])
processor.plot_hist_against_target(['age','duration','campaign','emp.var.rate','euribor3m'])
#processor.plot_target_imbalance()
processor.assign_years()
categorical_columns = ['job','marital','education',
                                         'default','loan','housing','contact',
                                         'month','poutcome']
numerical_columns = ['age','pdays','previous','emp.var.rate','cons.price.idx',
                                         'cons.conf.idx','euribor3m','nr.employed','Year']


column_transformer = processor.get_column_transformer(categorical_columns,numerical_columns,['duration','day_of_week'])


X_train,X_test,y_train,y_test=  processor.train_test_split()

model.add_classifier(model.get_random_forest_classifier(),'Random Forest')
#model.add_classifier(model.get_logistic_classifier(),'Logistic Regressor')
model.add_classifier(model.get_xgboost_classifier(),'XGBoost')
model.create_pipes(column_transformer)
kfold = model.get_cross_validation_splitter(is_strattified = False)
strattified_fold = model.get_cross_validation_splitter(is_strattified = False)
models_k_fold = model.compare_models(kfold , X_train, y_train,scoring = 'roc_auc')
models_k_strattified = model.compare_models(kfold , X_train, y_train)

classifiers = models_k_fold['classifiers']
scores_auc = models_k_fold['scores']

models_k_fold_accuracy = model.compare_models(kfold , X_train, y_train,scoring='accuracy')
scores_accuracy = models_k_fold_accuracy['scores']


data = {'Name':classifiers , 'Auc':scores_auc, 'Accuracy': scores_accuracy} 
performance_df = pd.DataFrame(data)



print(performance_df)

best_score = 0
best_classifier = None
i=0
for score in models_k_fold['scores']:
    if(score>best_score):
        best_score = score
        best_classifier = models_k_fold['classifiers'][i]
    i+=1
print(best_classifier)
    
