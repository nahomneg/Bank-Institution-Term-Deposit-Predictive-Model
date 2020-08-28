# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 03:56:24 2020

@author: Nahom Negussie
"""




import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class PreProcessing:
    def __init__(self,data_frame):
        self.data_frame = data_frame
        self.feature_engineering = self.FeatureEngineering()
    
    def detect_outliers_boxplot(self,columns):
        
        axes = []
        index = 0
        while(index<len(columns)):
            fig, (ax1) = plt.subplots(1)
            fig.set_size_inches(15.5, 7.5)
            fig.suptitle('Identify Numerical Columns with Outliers using boxplots')
            sns.boxplot(data=self.data_frame[[columns[index]]], palette=['moccasin','moccasin'], orient="v",  ax=ax1)
            axes.append(ax1)
            index+=1
        
        
        return axes
    
    def __replace_column_outliers(self,column):
        quartile_one = self.data_frame[column].quantile(0.25)
        quartiile_three = self.data_frame[column].quantile(0.75)
        inter_quartile_range=quartiile_three-quartile_one
        Lower_Whisker = quartile_one-1.5*inter_quartile_range
        Upper_Whisker = quartiile_three+1.5*inter_quartile_range
        self.data_frame.loc[self.data_frame[column] > Upper_Whisker, column] = self.data_frame[column].median()
        self.data_frame.loc[self.data_frame[column] < Lower_Whisker, column] = self.data_frame[column].median()
    
    def handle_outliers(self,columns):
        i = 0
        while(i<len(columns)):
            self.__replace_column_outliers(columns[i])
            i+=1
    
    def assign_years(self):
        self.data_frame['Year'] = self.data_frame.apply(lambda row: self.feature_engineering.get_year(row['month']),axis=1)
    
    def get_data_frame(self):
        return self.data_frame
    
    def get_column_transformer(self,categorical_columns,numerical_columns,drop_columns):
        return ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns),
                                  ('drop_columns' , 'drop', drop_columns),
                                  ('scaler', StandardScaler(),numerical_columns),], 
                                 remainder='passthrough')
    def get_features(self):
        return self.data_frame.loc[:, self.data_frame.columns != 'y']
    
    def get_target(self):
        return self.data_frame['y']
    
    def train_test_split(self):
        return train_test_split(self.get_features(),self.get_target(),test_size=0.1,random_state=0)

        
    class FeatureEngineering():
        def __init__(self):
            self.year=2008
            self.months = dict([('jan', 1),('feb', 2),('mar', 3),('apr', 4),('may', 5),
                    ('jun', 6),('jul', 7),('aug', 8),('sep', 9),('oct', 10),('nov', 11),('dec', 12),('kl', 12)])
            self.current_month = 5
        
        def get_year(self,month):
            current_month = self.months[month]
            if(current_month) >= self.current_month:
                self.current_month = current_month
            else: 
                self.year +=1
                self.current_month = current_month
            return self.year
        
        
