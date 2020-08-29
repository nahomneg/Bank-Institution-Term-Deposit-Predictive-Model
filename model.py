# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:00:28 2020

@author: Nahom Negussie

This class is concerned with cretaing classifiers, creating pipes, k-fold splitting and comparison of the available clasifiers
"""
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

# Model class concerned with cretaing classifiers, creating pipes, k-fold splitting and comparison of the available classifiers 
class Model():
    
    #initialize the model class with empty pipes and dictionary of classifiers
    def __init__(self):
        self.pipes = []
        self.classifiers = {
    
        }
    
    # get a random forest classifier
    def get_random_forest_classifier(self):
        self.random_forest_classifier = RandomForestClassifier()
        return self.random_forest_classifier
    
    # get a multi_perceptron classifier
    def get_multi_perceptron(self):
        self.multi_perceptron = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=50, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
        return self.multi_perceptron 
    
    # get a logistic classifier
    def get_logistic_classifier(self):
        self.logistic_classifier = LogisticRegression()
        return self.logistic_classifier
    
    # get an xgboost classifier
    def get_xgboost_classifier(self):
        self.xgboost_classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, gamma=0,
                      learning_rate=0.1, max_delta_step=0, max_depth=3)
        return self.xgboost_classifier
    
    
    # adds a classifier to the dictionary of classifiers
    def add_classifier(self,classifier,name):
        self.classifiers[name] = classifier
        
    # create pipes using all classifiers and a column transformer as a preprocessor 
    def create_pipes(self,column_transformer):
        self.pipes = []
        for name , classifier in self.classifiers.items():
            pipe = Pipeline([('preprocessor', column_transformer),
                             ('pca3',PCA(n_components=5)),
                              (name, classifier)])
            self.pipes.append(pipe)
    
    # create a stratified or normal K-Fold depending on is_stratified
    def get_cross_validation_splitter(self,is_strattified=False):
        cv = None
        if(not is_strattified):
            cv = KFold(n_splits=5,shuffle=True, random_state=1)
        else:
            cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
            
        return cv
    
    # compare models by using the pipes and cross validator
    def compare_models(self,cross_validator,X_train,y_train,scoring = 'roc_auc'):
        scores = [] 
        classifiers = []
        for pipe in self.pipes:
            results = cross_val_score(pipe, X_train, y_train, cv=cross_validator, scoring=scoring)
            scores.append(results.mean())
            keys=list(pipe.named_steps.keys())
            classifiers.append(keys[2])
        return {'scores': scores, 'classifiers': classifiers, 'scoring':scoring}
    
    
    
        