# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 21:17:29 2020

@author: Nahom Negussie
"""
import pandas as pd

import data
from model import Model
from sklearn import preprocessing


class Main():
    
    def __init__(self):
        self.data_frame = pd.read_csv("dataset/bank-additional-full.csv",sep=';'  , engine='python')
        self.processor = data.PreProcessing(data_frame) #initialize the processor
        self.model = Model() # initialize the model class
    
    def make_plots(self):
        self.processor.detect_outliers_boxplot(['duration','age','cons.conf.idx'])
        self.processor.handle_outliers(['duration','age','cons.conf.idx'])
        self.processor.detect_outliers_boxplot(['duration','age','cons.conf.idx'])
        self.processor.plot_multiple_categorical_against_target(['poutcome','marital','loan','housing'])
        self.processor.plot_hist_against_target(['age','duration','campaign','emp.var.rate','euribor3m'])
        self.processor.plot_target_imbalance()
    
    def run(self):
        #make_plots() # uncomment to make plots
    
        # assign years to every row of the data set
        self.processor.assign_years()
        
        
        # distinguish categorical and numerical columns
        
        categorical_columns = ['job','marital','education',
                                                 'default','loan','housing','contact',
                                                 'month','poutcome']
        numerical_columns = ['age','pdays','previous','emp.var.rate','cons.price.idx',
                                                 'cons.conf.idx','euribor3m','nr.employed','Year']
        
        # get the columns transformer from the model class 
        
        column_transformer = self.processor.get_column_transformer(categorical_columns,numerical_columns,['duration','day_of_week'])
        
        # split the dataframe to training and test sets
        self.X_train,self.X_test,self.y_train,self.y_test=  self.processor.train_test_split()
        
        self.add_classifiers()
        
        
        
        # create pipes
        self.model.create_pipes(column_transformer)
        
        #compare models
        self.compare_models()
      
    # add classifiers to be compared    
    def add_classifiers(self):
        self.model.add_classifier(model.get_random_forest_classifier(),'Random Forest')
        self.model.add_classifier(model.get_logistic_classifier(),'Logistic Regressor')
        self.model.add_classifier(model.get_xgboost_classifier(),'XGBoost')
    
    def compare_models(self):
        # create a K-fold to be used for validating classifiers
        kfold = self.model.get_cross_validation_splitter(is_strattified = False)
        strattified_fold = self.model.get_cross_validation_splitter(is_strattified = False)
        
        print("Comparing the models...")
        models_k_fold = self.model.compare_models(kfold , self.X_train, self.y_train,scoring = 'roc_auc')
        models_k_strattified = self.model.compare_models(kfold , self.X_train, self.y_train)
        
        
        classifiers = models_k_fold['classifiers']
        scores_auc = models_k_fold['scores']
        
        models_k_fold_accuracy = self.model.compare_models(kfold , self.X_train, self.y_train,scoring='accuracy')
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
        print('\n Best classifier is', best_classifier)
        
main = Main()
main.run()      