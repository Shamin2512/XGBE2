# Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
# Goal is to predict if protein is SNP or PD

# **import XGBoost and other ML modules**
import pandas as pd  # import data for training, encoding and testing
import numpy as np  # calc the mean and SD
import xgboost as xgb  # XGboost Learning API
import matplotlib.pyplot as plt  # graphing/plotting stuff
import random as rd
from xgboost import XGBClassifier  # SK learn API for XGB model building
from xgboost import XGBRegressor  # SK learn API for XGB regression
from sklearn.metrics import (
    matthews_corrcoef,  # MCC for evaluation
    f1_score,  # F1 score for evaluation
    accuracy_score,  # Accuracy for evaluation
    balanced_accuracy_score, roc_auc_score, make_scorer,  # Scoring metrics
    confusion_matrix,  # creates the confusion matrix - stats on how accurate the test set output is
    ConfusionMatrixDisplay,  # draws the confusion matrix
)
from sklearn.model_selection import (
    train_test_split,  # Splits data frame into the training set and testing set
    GridSearchCV,  # Cross validation to improve hyperparameters
)
from sklearn.utils import shuffle #shuffles rows

xgb.set_config(verbosity=2)  # Print all XGB commands

# **Create, clean and convert the train.csv dataset to a dataframe**
df = pd.read_csv('E2.csv')  # Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'],
        axis=1,
        inplace=True)  # removes columns unrequired columns, updating the variable df
df.columns = df.columns.str.replace(' ', '_')  # Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True)  # Replace all blank spaces with underscore (none were present)

#**Create dataframe with 1100 of PD and SNP**

PD_L = df.loc[df['dataset'] == 'pd']
SNP_L = df.loc[df['dataset'] == 'snp']

# concat
sample_df = pd.concat((PD_L.sample(n=1100), SNP_L.sample(n=1100))) #df with 1100 PD and 1100 SNP
# shuffle so data points are mixed
sample_df = shuffle(sample_df)
# reset idnex
sample_df.reset_index(drop=True, inplace=True)
# check number of entries from both classes are equal
assert sample_df[sample_df.dataset == "pd"].shape[0] == 1100 #shape[0] checks if there are 1100 rows in mutations column
assert sample_df[sample_df.dataset == "snp"].shape[0] == 1100
X = sample_df.drop('dataset', axis=1) #Training data set created with equal numnber of PD and SNP, but dropped

# X dataframes all have 1100 points without dataset
X1 = shuffle(X)
X2 = shuffle(X)
X3 = shuffle(X)

# Y dataframe one hot encoding
y_encoded = pd.get_dummies(sample_df, columns=['dataset'],
                           prefix=['Mutation'])  # y is df with mutations changing from object -> unint8 (integer)
y_col = y_encoded['Mutation_pd'].copy().astype('int32') #equal number of PD and SNP afte encoding (y_col.value_counts())

#y dataframes all have 2200 points
y1 = shuffle(y_col)
y2 = shuffle(y_col)
y3 = shuffle(y_col)

# **Split data into training and test**
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=42,
                                                        stratify=y1)  # Splits data into training and testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=42,
                                                        stratify=y2)  # Splits data into training and testing
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=42,
                                                        stratify=y3)  # Splits data into training and testing

# **XGB Dmatrix training model**
d_train1 = xgb.DMatrix(X_train1, label=y_train1)  # all features are floats
d_test1 = xgb.DMatrix(X_test1, label=y_test1)

d_train2 = xgb.DMatrix(X_train2, label=y_train2)
d_test2 = xgb.DMatrix(X_test2, label=y_test2)

d_train3 = xgb.DMatrix(X_train3, label=y_train3)
d_test3 = xgb.DMatrix(X_test3, label=y_test3)

param = {  # Dictionary of parameters to initally train the model
    'booster': 'gbtree',  # non-linear, tree method (default)
    'verbosity': 1,  # outputs the evaluation of each tree
    'eta': 0.1,  # Same as learning rate, shrinkage of each step when approaching the optimum value
    'colsample_bytree': 0.8,  # How much subsampling for each tree
    'max_depth': 5,  # Greater the depth, more prone to overfitting; tune from CV
    'eval_metric': ['auc', 'aucpr'],
    'min_child_weight': 1,
    'objective': 'binary:logistic'  # classifies the outcome as either 0 (SNP), or 1 (PD). Non multiclass classification
}

model1 = xgb.train(param, d_train1, evals=[(d_test1, 'eval'), (d_train1, 'train')], num_boost_round=100,
                   early_stopping_rounds=10)
model2 = xgb.train(param, d_train2, evals=[(d_test2, 'eval'), (d_train2, 'train')], num_boost_round=100,
                   early_stopping_rounds=10)
model3 = xgb.train(param, d_train2, evals=[(d_test2, 'eval'), (d_train2, 'train')], num_boost_round=100,
                   early_stopping_rounds=10)

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
