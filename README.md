# Bank-Institution-Term-Deposit-Predictive-Model

# Business need
The Bank of Portugal wants to find a model that can predict which future clients would subscribe to their term deposit. Having such an effective predictive model can help increase their campaign efficiency as they would be able to identify customers who would subscribe to their term deposit and thereby direct their marketing efforts to them. This would help them better manage their resources.


# Goal
The goal of this project is to come up with such an effective predictive model by using the data collected from customers of Bank of Portugal.


This repository is used for carrying out data exploration, data
cleaning, feature extraction, and developing robust machine learning algorithms that
would aid the Bank Of Portugal in making their marketing campaigns more efficient.

# Data

The Bank of Portugal collected a huge amount of data that includes
customers profiles of those who have to subscribe to term deposits and the ones who
did not subscribe to a term deposit.

## Columns
* Columns:
* age
* job
* marital
* education
* default
* housing
* loan
* contact
* day
* month
* duration
* campaign
* pdays
* previous
* poutcome
are among the most important ones

Explanatory Data Analysis showed the data has a high class imbalance ( 88% : 11 % ).
Duration was excluded from consideration because of its high correlation to the target.
Day of week was also dropped because EDA showed it has little or no impact on the performance of the models.

# New Features
The data contains contact month which was used to come up with a new feature 'Year'.

# Method 

4 classifier algorithms were considered in this project. They were compared againist each using different metrics, but ROC_AUC being the main one. K-Fold and Stratified K-Fold techniques were used to get a validation set from the training data which is then used for cross-validation with 5 folds.
### The classifiers are

* XGBoost
* Logistic Regression
* Multi Layer Perceptron
* Random Forest

Prior to being fed to the clasifiers the dataset's categorical columns were encoded using one hot encoding. The numerical columns were first cleaned from outliers. Then Standardization was applied to them.

# Results

Accuracy's of all 4 models were around 0.9. But since the target has high class imbalance we should not rely on accuracy. 
ROC_AUC score, a more tolerant evaluation metric showed scores centered at 0.76.

XGBoost was the best classifier in both accuracy and ROC_AUC with an accuracy of 0.89 and ROC_AUC of 0.78.
