# Example 2 is inbalanced data set; ~2200 in PD and ~1100 in SNP. Data must be correctly split into training and testing
# Goal is to predict if protein is SNP or PD

#Imports the required modules and packages
import pandas as pd  #Import data for training, encoding and testing
import numpy as np  #Calc the mean and SD
import matplotlib.pyplot as plt  #Graphing/plotting stuff
import random as rd
import time
import sys

from sklearn import tree

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle #shuffles rows
from sklearn.neighbors import KNeighborsClassifier #allows for confidence scores to be predicted for each



np.set_printoptions(threshold=sys.maxsize) #full array printing



#Create, clean and convert dataset E2.csv to PD dataframe**
df = pd.read_csv('E2.csv')  #Create PD data frame from .csv
df.drop(['pdbcode:chain:resnum:mutation'], axis=1, inplace=True)  #Removes unrequired columns
df.columns = df.columns.str.replace(' ', '_')  # Removes any blank attributes
df.replace(' ', '_', regex=True, inplace=True)  # Replace all blank spaces with underscore (none were present)
df.reset_index(drop=True, inplace = True) #Resets index numbering from 0 and drops column
X_properties = df.drop('dataset', axis =1).fillna('0') #Instances for classification training
y_encoded = pd.get_dummies(df, columns=['dataset']) #Encodes dataset column so "PD" and "SNP" attributes are  0 or 1
y_dataset = y_encoded['dataset_pd'].copy().astype('int32') # Column where 1 = PD, 0 = SNP, intergers

print("Total samples:", len(df))
print("Number of PD:", len(df.loc[df['dataset'] == 'pd']))
print("Number of SNP:", len(df.loc[df['dataset'] == 'snp']))

#training and test split

X_train, X_test, y_train, y_test = train_test_split(X_properties, y_dataset, train_size = 0.8, random_state=42, stratify=y_dataset) #80% training and 20% testing split. Strartify ensures fixed poportion of y is in both sets
start=time.time() #Start timer for model building
clf = RandomForestClassifier(random_state = 42, n_estimators = 100) #Defines the Random Forest
clf.fit(X_train, y_train) #Generates a random forest from the training dataset

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1200)
tree.plot_tree(clf.estimators_[45],
               feature_names = None, 
               class_names= None,
               filled = True)
plt.savefig('clf_individualtree.png', bbox_inches = 'tight', pad_inches=0)

StandardScaler().fit(X_train).transform(X_train) #Scales data 
pipeline = make_pipeline( #Sets the random forest parameters
    StandardScaler(),
    LogisticRegression(solver='saga', max_iter=2000),
    verbose=2
    )





cs = clf.predict_proba(X) #Outputs the predictions on an instance's classification for each tree
stop=time.time()
    
# **Split data into training and test**
print(X, y)
with open('SNPorPD.txt', 'w+') as f:
        data=f.read()
        f.write(str(y.to_string()))

pipeline.fit(X, y) #applies list if transformers to give a fitted model

gridsearch = GridSearchCV( #validation
    estimator = LogisticRegression(solver='saga'),
    param_grid = {}, #dictionary of parameters to search through
    cv = StratifiedKFold(),
    n_jobs = 1, #how many processors to run in parallel
    scoring = 'f1',
    verbose = 3 
    ).fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Training time:", stop-start)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("MCC:\n", matthews_corrcoef(y_test, y_pred))
print("F1:\n", f1_score(y_test, y_pred))
