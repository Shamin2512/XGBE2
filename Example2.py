# Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
# Goal is to predict if protein is SNP or PD

#Imports the required modules and packages
import pandas as pd  #Import data for training, encoding and testing
import numpy as np  #Calc the mean and SD
import matplotlib.pyplot as plt  #Graphing/plotting stuff
import random as rd
import time
import sys

from sklearn.metrics import(
    matthews_corrcoef,  # CC for evaluation
    f1_score,  #F1 score for evaluation
    balanced_accuracy_score, roc_auc_score, make_scorer,  #Scoring metrics
    confusion_matrix,  #Creates the confusion matrix - stats on how accurate the test set output is
        )
from sklearn.model_selection import(
    train_test_split,  # Splits data frame into the training set and testing set
    GridSearchCV,  # Cross validation to improve hyperparameters
    StratifiedKFold
        )
from sklearn.ensemble import RandomForestClassifier #SK learn API for classificastion random forests

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle #shuffles rows
from sklearn.neighbors import KNeighborsClassifier #allows for confidence scores to be predicted for each

np.set_printoptions(threshold=sys.maxsize) #full array printing



def Data_Clean(): #Create, clean and convert the train.csv dataset to a dataframe**
    df = pd.read_csv('E2.csv')  #Pandas creates data frame from the .csv mutation data
    df.drop(['pdbcode:chain:resnum:mutation'], axis=1, inplace=True)  # removes columns unrequired columns, updating the variable df
    df.columns = df.columns.str.replace(' ', '_')  # Removes gaps in column names
    df.replace(' ', '_', regex=True, inplace=True)  # Replace all blank spaces with underscore (none were present)
    df.reset_index(inplace = True)
    X = df.drop('dataset', axis =1).fillna('0')
    y_encoded = pd.get_dummies(df, columns=['dataset'])
    y = y_encoded['dataset_pd'].copy().astype('int32')
    print("Total samples:", len(df))
    print("Number of PD:", len(df.loc[df['dataset'] == 'pd']))
    print("Number of SNP:", len(df.loc[df['dataset'] == 'snp']))
Data_Clean()

def Tree_Building():
    clf = RandomForestClassifier(random_state = 42)
    clf.fit(X, y) #generates forest from training data
    StandardScaler().fit(X).transform(X) #scales data 
    pipeline = make_pipeline(#equivilent of fitting to XGB parameters
        StandardScaler(),
        LogisticRegression(solver='saga', max_iter=2000),
        verbose=2
        )   
    cs = clf.predict_proba(X)
    print(cs)
    with open('ConfidenceScores.txt', 'w+') as f:
        data=f.read()
        f.write(str(cs))
Tree_Building()
    
# **Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42, stratify=y)
pipeline.fit(X, y)

#weights = np.linspace(0.0,0.99,200) # creates 200 evenly spaced values between 0 and 0.99
#param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]} # applys weights to each value between 0 and 0.99

gridsearch = GridSearchCV( #validation
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
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("MCC:\n", matthews_corrcoef(y_test, y_pred))
print("F1:\n", f1_score(y_test, y_pred))


input('Press ENTER to exit')
