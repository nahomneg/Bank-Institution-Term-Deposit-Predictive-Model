# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 03:56:24 2020

@author: Nahom Negussie


This class is concerned with all the preprocessing, data exploration(plotting) and Feature extraction of the 
Bank of portugal data.
"""



# import the necessary modules
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PreProcessing:
    # initialize class with a constructor that takes the data_frame to be processed
    # set matplot lib settings
    def __init__(self,data_frame):
        self.data_frame = data_frame
        
        font = {'family' : 'bold',
        'weight' : 'bold',
        'size'   : 2}
        plt.rc('xtick',labelsize=23)
        plt.rc('ytick',labelsize=23)
        plt.rc('legend',fontsize=23)
        
        
        plt.rcParams["font.family"] = "cursive"
        plt.rc('font', **font) 
    
    # plot the count of yes and no values of the target column 'y'
    def plot_target_imbalance(self):
        total = len(self.data_frame['y'])*1.
        ax = sns.countplot(x=self.data_frame['y'])
        for p in ax.patches:
            ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))
        plt.xlabel('Subscribed for Term deposit')
        plt.title('Plot showing the class imbalance')
        labels=["Didn't open term deposit","Open term deposit"]
        return ax
    
    
    #multiple plots of the count of yes and no values of target column 'y' per categories of a variable
    def plot_multiple_categorical_against_target(self,columns,target='y'):
        axes = []
        i=0
        while(i<len(columns)):
            # create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            fig.suptitle('Identify the effect of each categorical variable' , fontsize=22)
            
            labels=["Didn't open term deposit","Open term deposit"]
            sns.countplot(x= self.data_frame[columns[i]] , palette=['salmon','gold'],  hue=self.data_frame[target], ax=ax1)
            
            
            
          
            sns.countplot(x= self.data_frame[columns[i+1]] , palette=['salmon','gold'],  hue=self.data_frame[target], ax=ax2)
            ax1.set_title('Count of Yes/No by '+columns[i],fontname='Comic Sans MS', fontsize=18)
            
            
            ax2.set_title('Count of Yes/No by ' + columns[i+1],fontname='Comic Sans MS', fontsize=18)
            ax1.set_xlabel(columns[i], fontsize=20)
            ax2.set_xlabel(columns[i+1], fontsize=20)
            ax1.set_ylabel('Count', fontsize=20)
            ax2.set_ylabel('Count', fontsize=20)
            
            i+=2
            
            axes.append(ax1)
            axes.append(ax2)
        return axes
    
    # plot the count of yes and no values of the target column 'y' per categories of a variable
    def plot_single_categorical_against_target(self,column,target='y'):
        
            plt.figure(figsize=(14,8))
            return sns.countplot(x=column,hue=target, data=self.data_frame,palette=['gold','salmon'])
            
    # plot the correlation heatmap of numerical columns
    def plot_correletion_matrix(self):
        numeric_only = self.data_frame.select_dtypes(exclude='object')
        plt.figure(figsize=(10,7))
        ax = sns.heatmap(numeric_only.corr(),annot=True)
        plt.title('Correlation Matrix')
        return ax
    
    # plot the distribution of numerical columns using histograms
    def plot_distribution(self,columns):
        sns.set()
        axes = []
        i=0
        while i<len(columns):
          fig, (ax1, ax2) = plt.subplots(1, 2)
          fig.set_size_inches(18, 7)
          sns.distplot(self.data_frame[columns[i]],bins=15,color='orange',ax=ax1)
          ax1.set_xlabel(columns[i], fontsize=20)
          ax1.set_ylabel('Value', fontsize=20)
          try:
            sns.distplot(self.data_frame[columns[i+1]],bins=15,color='orange',ax=ax2)
            ax2.set_xlabel(columns[i+1], fontsize=20)
            ax2.set_ylabel('Value', fontsize=20)
          except:
            print('Warning: Odd number of variables provided. One plot will be empty')
          i+=2
          axes.append(ax1)
          axes.append(ax2)
        return axes
                
    # detect outliers of numerical using boxplots
    def detect_outliers_boxplot(self,columns):
        sns.set()
        axes = []
        index = 0
        while(index<len(columns)):
            fig, (ax1,ax2) = plt.subplots(1,2)
            fig.set_size_inches(15.5, 7.5)
            fig.suptitle('Identify Numerical Columns with Outliers using boxplots')
            sns.boxplot(y=columns[index], x='y', data=self.data_frame,palette=['moccasin','moccasin'], orient="v",  ax=ax1)
            ax1.set_xlabel(columns[index], fontsize=20)
            ax1.set_ylabel('Value', fontsize=20)
            try:
                sns.boxplot(y=columns[index+1], x='y', data=self.data_frame, palette=['moccasin','moccasin'], orient="v",  ax=ax2)
                ax2.set_xlabel(columns[index+1], fontsize=20)
            
                ax2.set_ylabel('Value', fontsize=20)
            except:
                print('Odd number of columns provided')
            axes.append(ax1)
            axes.append(ax2)
            index+=2
        
        
        return axes
    
    
    #plot the distribution of the yes and no values of the target variable based on multiple columns
    def plot_hist_against_target(self,columns):
        axes = []
        index = 0
        data1 = self.data_frame[self.data_frame['y'] == 'yes']
        data2 = self.data_frame[self.data_frame['y'] == 'no']
        while(index<len(columns)):
            fig, (ax1,ax2) = plt.subplots(1,2)
            fig.set_size_inches(15.5, 7.5)
            fig.suptitle('Histograms of numerical variables per each target class')
            ax1.hist(data2[columns[index]],color = '#DC4405',alpha=0.7,bins=20, edgecolor='white') 
            ax1.hist(data1[columns[index]],color='#000000',alpha=0.5,bins=20, edgecolor='white')
            
            ax1.set_xlabel(columns[index], fontsize=20)
            ax1.set_ylabel('Count', fontsize=20)
            plt.figlegend(('Yes', 'No'),loc="right",title = "Term deposit")
            try:
                ax2.hist(data2[columns[index+1]],color = '#DC4405',alpha=0.7,bins=20, edgecolor='white') 
                ax2.hist(data1[columns[index+1]],color='#000000',alpha=0.5,bins=20, edgecolor='white')
                ax2.set_xlabel(columns[index+1], fontsize=20)
            except:
                print('Odd number of columns provided')
            axes.append(ax1)
            axes.append(ax2)
            index+=2
        plt.figlegend(('Yes', 'No'),loc="right",title = "Term deposit")
        return axes
    
    # replace outliers of a column with the respective median 
    def __replace_column_outliers(self,column):
        quartile_one = self.data_frame[column].quantile(0.25)
        quartiile_three = self.data_frame[column].quantile(0.75)
        inter_quartile_range=quartiile_three-quartile_one
        Lower_Whisker = quartile_one-1.5*inter_quartile_range
        Upper_Whisker = quartiile_three+1.5*inter_quartile_range
        self.data_frame.loc[self.data_frame[column] > Upper_Whisker, column] = self.data_frame[column].median()
        self.data_frame.loc[self.data_frame[column] < Lower_Whisker, column] = self.data_frame[column].median()
    
    # handle all outliers of the given column by using the above __replace_column_outliers()
    def handle_outliers(self,columns):
        i = 0
        while(i<len(columns)):
            self.__replace_column_outliers(columns[i])
            i+=1
    
    # assign years to each row of the data_frame under consideration
    def assign_years(self):
        feature_engineering = self.FeatureEngineering()
        self.data_frame['Year'] = self.data_frame.apply(lambda row: feature_engineering.get_year(row['month']),axis=1)
    
    # get the current dataframe
    def get_data_frame(self):
        return self.data_frame
    
    
    # get the column transformer that is responsible for one hot encoding and standardization
    def get_column_transformer(self,categorical_columns,numerical_columns,drop_columns):
        return ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns),
                                  ('drop_columns' , 'drop', drop_columns),
                                  ('scaler', StandardScaler(),numerical_columns),], 
                                 remainder='passthrough')
    
    # get all columns except target
    def get_features(self):
        return self.data_frame.loc[:, self.data_frame.columns != 'y']
    
    # get target column
    def get_target(self):
        return self.data_frame['y']
    
    
    # train_test split the dataframe to training and test sets in 80:20 ratio
    def train_test_split(self):
        return train_test_split(self.get_features(),self.get_target(),test_size=0.2,random_state=0)

    # class used to create a new feature year based on 
    class FeatureEngineering():
        # initialize year with 2008 since the data is from 2008 to 2010
        def __init__(self):
            self.year=2008
            self.months = dict([('jan', 1),('feb', 2),('mar', 3),('apr', 4),('may', 5),
                    ('jun', 6),('jul', 7),('aug', 8),('sep', 9),('oct', 10),('nov', 11),('dec', 12),('kl', 12)])
            self.current_month = 5
        
        # return the correct year based on a month by comparing it with the current month
        def get_year(self,month):
            current_month = self.months[month]
            if(current_month) >= self.current_month:
                self.current_month = current_month
            else: 
                self.year +=1
                self.current_month = current_month
            return self.year
        
        
