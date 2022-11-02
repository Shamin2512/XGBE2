# Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
# Goal is to predict if protein is SNP or PD

import pandas as pd  # import data for training, encoding and testing
import numpy as np  # calc the mean and SD
import matplotlib.pyplot as plt  # graphing/plotting stuff
import random as rd
import time

from sklearn.metrics import (
    matthews_corrcoef,  # MCC for evaluation
    f1_score,  # F1 score for evaluation
    balanced_accuracy_score, roc_auc_score, make_scorer,  # Scoring metrics
    confusion_matrix,  # creates the confusion matrix - stats on how accurate the test set output is
    )
from sklearn.model_selection import (
    train_test_split,  # Splits data frame into the training set and testing set
    GridSearchCV,  # Cross validation to improve hyperparameters
    StratifiedKFold
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
print("Total samples:", len(df))
print("Number of PD:", len(df.loc[df['dataset'] == 'pd']))
print("Number of SNP:", len(df.loc[df['dataset'] == 'snp']))

#**Model training initialise**
clf = RandomForestClassifier(random_state = 42)
start = time.time()
clf.fit(X, y)
StandardScaler().fit(X).transform(X)

#**Parameters pipeline**
pipeline = make_pipeline( #equivilant of fitting to XGB parameters
    StandardScaler(),
    LogisticRegression(solver='saga', max_iter=2000, class_weight = [{0:8557286432160804, 1:1.0-8557286432160804}]),
    verbose=2
    )
# **Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42, stratify=y)

pipeline.fit(X_train, y_train) #scale training data
stop = time.time()
#weights = np.linspace(0.0,0.99,200) # creates 200 evenly spaced values between 0 and 0.99
#param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]} # applys weights to each value between 0 and 0.99
start2=time.time()

gridsearch = GridSearchCV(
    estimator = LogisticRegression(solver='saga'),
    param_grid = {}, #dictionary of parameters to search through
    cv = StratifiedKFold(),
    n_jobs = 1, #how many processors to run in parallel
    scoring = 'f1',
    verbose = 3 
    ).fit(X_train, y_train)
stop2=time.time()

y_pred = clf.predict(X_test)
print("Training time:", stop-start)
print("Validation tuning time:", stop2-start2)
print("Accuracy:\n", accuracy_score(pipeline.predict(X_test), y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("MCC:\n", matthews_corrcoef(y_test, y_pred))
print("F1:\n", f1_score(y_test, y_pred))
