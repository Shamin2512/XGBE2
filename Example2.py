#Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
#Goal is to predict if protein is SNP or PD

#**import XGBoost and other ML modules**
import pandas as pd #import data for training, encoding and testing
import numpy as np #calc the mean and SD
import xgboost as xgb #XGboost Learning API
import matplotlib.pyplot as plt #graphing/plotting stuff
#import random
#from collections import Counter #imports collections module 
from xgboost import XGBClassifier #SK learn API for XGB model building
from xgboost import XGBRegressor #SK learn API for XGB regression
from sklearn.metrics import matthews_corrcoef #MCC for evaluation
from sklearn.metrics import f1_score #F1 score for evaluation
from sklearn.metrics import accuracy_score #Accuracy for evaluation
from sklearn.model_selection import train_test_split #Splits data frame into the training set and testing set
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer #Scoring metrics 
from sklearn.model_selection import GridSearchCV #Cross validation to improve hyperparameters
from sklearn.metrics import confusion_matrix #creates the confusion matrix - stats on how accurate the test set output is
from sklearn.metrics import ConfusionMatrixDisplay #draws the confusion matrix

xgb.set_config(verbosity=2) #Print all XGB commands
          
#**Create, clean and convert the train.csv dataset to a dataframe**
df = pd.read_csv('E2.csv') #Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'],axis=1, inplace=True) #removes columns unrequired columns, updating the variable df
df.columns = df.columns.str.replace(' ', '_') #Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True) #Replace all blank spaces with underscore (none were present)

#**Encoding the categorical data for dataframe y**
#X_total = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation'])
#X_col = X_total['Mutation_pd'].copy().convert_dtypes(convert_integer=True) #Full dataframe with mutations column. 1 = PD, 0 = SNP
df.reset_index(inplace = True) #resets index count to 0 in dataframe

PD_L = df.loc[df['dataset'] == 'pd']
SNP_L = df.loc[df['dataset'] == 'snp']
PD_L.to_csv('PD.csv')
SNP_L.to_csv("C:\Users\shami\git\XGBE2\")
print(PD_L, SNP_L)
new_df = PD_L.sample(n=2200).update(SNP_L.sample(n=1100))
print(new_df)





#count = X_col['Mutation'].values_counts()

#print(X_col.isin([1]).value_counts())
    
X_full = df.drop('dataset', axis=1).copy() #X is dataframe with data used to train and predict if SNP or PD

#X dataframes all have 1100 points
X1 = X_full.sample(n=2200) 
X2 = X_full.sample(n=2200)
X3 = X_full.sample(n=2200)

y_encoded = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation']) #y is df with mutations changing from object -> unint8 (integer)
y_col = y_encoded['Mutation_pd'].copy().convert_dtypes()

#y dataframes all have 2200 points
y1 = y_col.sample(n=1100) 
y2 = y_col.sample(n=1100)
y3 = y_col.sample(n=1100)

#**Split data into training and test**
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=42, stratify=y1) #Splits data into training and testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=42, stratify=y2) #Splits data into training and testing
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=42, stratify=y3) #Splits data into training and testing

#**XGB Dmatrix training model**
d_train1 = xgb.DMatrix(X_train1, label=y_train1) #all features are floats
d_test1 = xgb.DMatrix(X_test1, label=y_test1)

d_train2 = xgb.DMatrix(X_train2, label=y_train2)
d_test2 = xgb.DMatrix(X_test2, label=y_test2)

d_train3 = xgb.DMatrix(X_train3, label=y_train3)
d_test3 = xgb.DMatrix(X_test3, label=y_test3)

param = { #Dictionary of parameters to initally train the model
    'booster': 'gbtree', #non-linear, tree method (default)
    'verbosity': 1, #outputs the evaluation of each tree
    'eta': 0.1, #Same as learning rate, shrinkage of each step when approaching the optimum value
    'colsample_bytree': 0.8, #How much subsampling for each tree
    'max_depth': 5, #Greater the depth, more prone to overfitting; tune from CV
    'eval_metric': ['auc', 'aucpr'],
    'min_child_weight': 1,
    'objective': 'binary:logistic' #classifies the outcome as either 0 (SNP), or 1 (PD). Non multiclass classification
    }

model1 = xgb.train(param, d_train1, evals= [(d_test1, 'eval'), (d_train1, 'train')], num_boost_round=50, early_stopping_rounds=20)
model2 = xgb.train(param, d_train2, evals= [(d_test2, 'eval'), (d_train2, 'train')], num_boost_round=50, early_stopping_rounds=20)
model3 = xgb.train(param, d_train2, evals= [(d_test2, 'eval'), (d_train2, 'train')], num_boost_round=50, early_stopping_rounds=20)

#Cross validation paramaters
dmatrix_val1 = xgb.DMatrix(X1, y1)
params = {
    'objective': 'binary:hinge',
    'colsample_bytree': 0.3,
    'eta': 0.1,
    'max_depth': 3
}
cross_val1 = xgb.cv(
    params=params,
    dtrain=dmatrix_val1, 
    nfold=5,
    num_boost_round=50, 
    early_stopping_rounds=10, 
    metrics='error', 
    as_pandas=True,
    seed=42
    )
print(cross_val1.head())

dmatrix_val2 = xgb.DMatrix(X2, y2)
params = {
    'objective': 'binary:hinge',
    'colsample_bytree': 0.3,
    'eta': 0.1,
    'max_depth': 3
}
cross_val2 = xgb.cv(
    params=params,
    dtrain=dmatrix_val2, 
    nfold=5,
    num_boost_round=50, 
    early_stopping_rounds=10, 
    metrics='error', 
    as_pandas=True,
    seed=42
    )
print(cross_val2.head())

dmatrix_val3 = xgb.DMatrix(X3, y3)
params = {
    'objective': 'binary:hinge',
    'colsample_bytree': 0.3,
    'eta': 0.1,
    'max_depth': 3
}
cross_val3 = xgb.cv(
    params=params,
    dtrain=dmatrix_val3, 
    nfold=5,
    num_boost_round=50, 
    early_stopping_rounds=10, 
    metrics='error', 
    as_pandas=True,
    seed=42
    )
print(cross_val3.head())

#**Plot confusion matrix using the true and predicted values**
#clf = xgb.XGBClassifier(**param)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

y_pred1 = model1.predict(d_test1) #No longer a pandas DF, is now a numpy array as Dmatrix
y_pred2 = model2.predict(d_test2) #No longer a pandas DF, is now a numpy array as Dmatrix
y_pred3 = model3.predict(d_test3) #No longer a pandas DF, is now a numpy array as Dmatrix
cm1 = confusion_matrix(y_test1, y_pred1 > 0.5)
cm2 = confusion_matrix(y_test2, y_pred2 > 0.5)
cm3 = confusion_matrix(y_test3, y_pred3 > 0.5)



print("Confusion Matrix:\n", cm1)
print(cm2)
print(cm3)
print("MCC:\n", matthews_corrcoef(y_test1, y_pred1), matthews_corrcoef(y_test2, y_pred2>0.5),matthews_corrcoef(y_test2, y_pred3>0.5))
print("F1 Score:\n", f1_score(y_test3, y_pred1>0.5), f1_score(y_test3, y_pred3>0.5), f1_score(y_test3, y_pred3>0.5))

