# Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
# Goal is to predict if protein is SNP or PD

# **import XGBoost and other ML modules**
import pandas as pd  # import data for training, encoding and testing
import numpy as np  # calc the mean and SD
#import xgboost as xgb  # XGboost Learning API
import matplotlib.pyplot as plt  # graphing/plotting stuff
import random as rd

#from xgboost import XGBClassifier  # SK learn API for XGB model building
#from xgboost import XGBRegressor  # SK learn API for XGB regression

from sklearn.metrics import (
    matthews_corrcoef,  # MCC for evaluation
    f1_score,  # F1 score for evaluation
    accuracy_score,  # Accuracy for evaluation
    balanced_accuracy_score, roc_auc_score, make_scorer,  # Scoring metrics
    confusion_matrix,  # creates the confusion matrix - stats on how accurate the test set output is
    ConfusionMatrixDisplay)  # draws the confusion matrix
    
from sklearn.model_selection import (
    train_test_split,  # Splits data frame into the training set and testing set
    GridSearchCV,  # Cross validation to improve hyperparameters
    )
from sklearn.ensemble import RandomForestClassifier #SK learn API

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.utils import shuffle #shuffles rows

# **Create, clean and convert the train.csv dataset to a dataframe**
df = pd.read_csv('E2.csv')  # Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'], axis=1, inplace=True)  # removes columns unrequired columns, updating the variable df
df.columns = df.columns.str.replace(' ', '_')  # Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True)  # Replace all blank spaces with underscore (none were present)
df.reset_index(inplace = True)

#**Data prep**
X = df.drop('dataset', axis =1).fillna('0')
y_encoded = pd.get_dummies(df, columns=['dataset'])
y = y_encoded['dataset_pd'].copy().astype('int32')
print("Number of PD:", len(df.loc[df['dataset'] == 'pd']))
print("Number of SNP:", len(df.loc[df['dataset'] == 'snp']))

#**Model training initialise**
clf = RandomForestClassifier(random_state = 42)
clf.fit(X, y)
StandardScaler().fit(X).transform(X)

#**Parameters pipeline**
pipeline = make_pipeline( #equivilant of fitting to XGB parameters
    StandardScaler(),
    LogisticRegression(solver='saga', max_iter=5000)
    )
# **Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42, stratify=y)

pipeline.fit(X_train, y_train)

print(pipeline.fit)
print(accuracy_score(pipeline.predict(X_test), y_test))
print("MCC:\n", matthews_corrcoef(y_test, y_pred))

#    'booster': 'gbtree',  # non-linear, tree method (default)
#    'verbosity': 1,  # outputs the evaluation of each tree
#    'eta': 0.3,  # Same as learning rate, shrinkage of each step when approaching the optimum value
#    'colsample_bytree': 0.8,  # How much subsampling for each tree
#    'max_depth': 6,  # Greater the depth, more prone to overfitting; tune from CV
#    'eval_metric': ['auc', 'aucpr'],
#    'min_child_weight': 1,
#    'objective': 'binary:logistic'  # classifies the outcome as either 0 (SNP), or 1 (PD). Non multiclass classification
#}


# **Plot confusion matrix using the true and predicted values**
#clf = xgb.XGBClassifier(**param)
#clf.fit(d_train1, y_test1)
#y_pred = clf.predict(X_test)
#ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

y_pred1 = model1.predict(d_test1)  # No longer a pandas DF, is now a numpy array as Dmatrix
y_pred2 = model2.predict(d_test2)  # No longer a pandas DF, is now a numpy array as Dmatrix
y_pred3 = model3.predict(d_test3)  # No longer a pandas DF, is now a numpy array as Dmatrix
cm1 = confusion_matrix(y_test1, y_pred1 > 0.5)
cm2 = confusion_matrix(y_test2, y_pred2 > 0.5)
cm3 = confusion_matrix(y_test3, y_pred3 > 0.5)

print("Confusion Matrix:\n", cm1)
print("Confusion Matrix:\n", cm2)
print("Confusion Matrix:\n", cm3)

print("MCC:\n", matthews_corrcoef(y_test1, y_pred1 >0.5), matthews_corrcoef(y_test2, y_pred2 > 0.5),
      matthews_corrcoef(y_test2, y_pred3 > 0.5))
print("F1 Score:\n", f1_score(y_test3, y_pred1 > 0.5), f1_score(y_test3, y_pred3 > 0.5),
      f1_score(y_test3, y_pred3 > 0.5))
