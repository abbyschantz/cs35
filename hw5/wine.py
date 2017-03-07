#
# GITHUB REPO: https://github.com/abbyschantz/cs35/tree/master/hw5
#

import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
import pandas as pd

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('wine.csv', header=0)    # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

X_data = df.drop('Hue', axis=1).values        # everything except the 'hue' column
y_data = df[ 'Hue' ].values      # also addressable by column name(s)

X_data_orig = X_data[:,:]
y_data_orig = y_data[:]

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
#X_data_orig = df.iloc[:,0:13].values        # iloc == "integer locations" of rows/cols
#y_data_orig = df[ 'Hue' ].values      # individually addressable columns (by name)
feature_names = df.columns.values          # get the names into a list!


X_data_full = X_data_orig[0:,:]  # make the 10 into 0 to keep all of the data
y_data_full = y_data_orig[0:]    # same for this line


targetsList = []
for hue in y_data_full:
    if not hue in targetsList:
        if hue == -1:
            continue
        targetsList.append(str(hue))

target_names = ['targetsList']   # and a list of the labels...
print("target_name", target_names)
#
# cross-validation and scoring to determine parameters...
# 

#
# we can scramble the data - but only if we know the test set's labels!
# 
# indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
# X_data_full = X_data_full[indices]
# y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full[0:30,:]              # the final testing data
X_train = X_data_full[30:,:]              # the training data

y_test = y_data_full[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[30:]                  # the training outputs/labels (known)

#
# cross-validation to determine the Decision Tree's parameter (to find max_depth)
#

max_depth=4
#
best_score = 0
index = 0
for i in range(1,11):
    score = 0
    dtree = tree.DecisionTreeRegressor(max_depth=i)

    for j in range(1, 11):  # run at least 10 times.... take the average cv testing score
        #
        # split into our cross-validation sets...
        #
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        dtree = dtree.fit(cv_data_train, cv_target_train) 
        #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
        #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
        score += dtree.score(cv_data_test,cv_target_test)
    average_score = score/10
    print(i, average_score)
    if average_score > best_score:
        best_score = average_score
        index = i
print("DT best score is ", best_score)



# dtree.feature_importances_  [already computed]


#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

X_test = X_data_orig[0:30,0:]              # the final testing data
X_train = X_data_orig[30:,:]              # the training data

y_test = y_data_orig[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_data_orig[30:]                  # the training outputs/labels (known)

#
# show the creation of three tree files (three max depths)
#
# best = 0
# for max_depth in [1,2,3,4]:
#     #
#     # we'll use max_depth between 1 and 3
#     #
#     dtree = tree.DecisionTreeRegressor(max_depth=max_depth)

#     # this next line is where the full training data is used for the model
#     dtree = dtree.fit(X_data_full, y_data_full) 
#     print("\nCreated and trained a knn classifier")  #, knn

#     #
#     # write out the dtree to tree.dot (or another filename of your choosing...)
#     tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',   # constructed filename!
#                             feature_names=feature_names,  filled=True, rotate=False, # LR vs UD
#                             class_names=target_names, leaves_parallel=True)  
#     # the website to visualize the resulting graph (the tree) is at www.webgraphviz.com
    #
dtree = tree.DecisionTreeRegressor(max_depth=5)

# this next line is where the full training data is used for the model
dtree = dtree.fit(X_train, y_train) 

# here are some examples, printed out:
print("wine_X_test's predicted outputs are")
print(dtree.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)







#
# now, show off Random Forests!
# 

#
# The data is already in good shape -- a couple of things to define again...
#

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full[0:30,:]              # the final testing data
X_train = X_data_full[30:,:] 

y_test = y_data_full[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[30:]                  # the training outputs/labels (known)


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#
#



# adapt for cross-validation (at least 10 runs w/ average test-score)
best_score = 0
index = 0
for i in range(1, 11):
    score = 0
    rforest = ensemble.RandomForestRegressor(max_depth=i, n_estimators=100)
    for j in range(1,11):

        #
        # split into our cross-validation sets...
        #
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        rforest = rforest.fit(cv_data_train, cv_target_train) 
        #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
        #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
        score += rforest.score(cv_data_test,cv_target_test)
    average_score = score/10
    print(i, average_score)
    if average_score > best_score:
        best_score = average_score
        index = i
print("randomForest best score", best_score)

    # rforest.estimators_  [a list of dtrees!]


#
# we'll use max_depth == 2
#
max_depth = 2
rforest = ensemble.RandomForestRegressor(max_depth=max_depth, n_estimators=100)


#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

X_test = X_data_full[0:30,:]              # the final testing data
X_train = X_data_full[30:,:] 

y_test = y_data_orig[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_data_orig[30:]                  # the training outputs/labels (known)

# this next line is where the full training data is used for the model
rforest = rforest.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

#
# feature importances
#
print("feature importances:", rforest.feature_importances_)  


# here are some examples, printed out:
print("wine_X_test's predicted outputs are")
print(rforest.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)



#
# Imputing example with iris data
#
WANT_IMPUTING_EXAMPLE = False
if WANT_IMPUTING_EXAMPLE == True:
    #
    # imputing missing values with a random forest...
    #
    #
    # to run this, you'll need _numeric_ labels (0,1,2) instead of 'setosa,'versicolor', etc.
    #    be sure to update that (way up above!)

    # we _create_ some missing data to show off imputing...
    #
    X_data_missing = X_data_full.copy()
    print("Original:", X_data_missing[-10:,:])
    y_data_missing = y_data_full.copy()  # we don't mess with this!

    # loop through the last 10 rows and set some to np.nan (not a number)
    #
    noise_rate = 0.25
    NUMROWS, NUMCOLS = X_data_missing.shape
    for row in range(NUMROWS-10,NUMROWS):
        for col in range(2,4):
            if np.random.uniform() < noise_rate:  # noise_rate chance of happening...
                X_data_missing[row,col] = np.nan

    # reassemble data together! np.hstack == horizontal stacking
    #
    all_data = np.hstack( [X_data_missing,y_data_missing.reshape(NUMROWS,1)] )
    print("Missing:", all_data[-10:,:])

    from impute import ImputeLearn
    from sklearn import neighbors

    #
    # impute with either RFs or KNN :-)
    #
    all_data_imp = ImputeLearn( all_data ).impute(learner = ensemble.RandomForestRegressor(n_estimators = 100,max_depth=2))
    #all_data_imp = ImputeLearn( all_data ).impute(learner = neighbors.KNeighborsRegressor(n_neighbors=5))

    print("Imputed:", all_data_imp[-10:,:])



####################################### Extra Credit ########################################


import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('wine_missing.csv', header=0)
df.head()
df.info()


# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# extract the underlying data with the values attribute:
X_data = df.drop('Hue', axis=1).values        # everything except the 'survival' column
y_data = df[ 'Hue' ].values      # also addressable by column name(s)

# We'll stick with numpy - here's the conversion to a numpy array
X_data_full = df.iloc[:,0:13].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ 'Hue' ].values      # also addressable by column name(s)


# we can drop the initial (unknown) rows -- if we want to test with known data
X_data_full = X_data_full[30:,:]   # 2d array
y_data_full = y_data_full[30:]     # 1d column


# we can scramble the remaining data if we want - only if we know the test set's labels 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

# The first nine are our test set - the rest are our training
X_test = X_data_full[0:30,0:13]              # the final testing data
X_train = X_data_full[30:,0:13]              # the training data

y_test = y_data_full[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[30:]                  # the training outputs/labels (known)



#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

#
# feature engineering...
#

# here is where you can re-scale/change column values...
# X_data[:,0] *= 100   # maybe the first column is worth 100x more!
# X_data[:,3] *= 100   # maybe the fourth column is worth 100x more!


#
# here, you'll implement the kNN model and cross validation
#
from sklearn.neighbors import KNeighborsRegressor

test_list = []
best_score = 0
index = 0

for i in range(1,11):
    score = 0
    knn = KNeighborsRegressor(n_neighbors=i)   # i is the "k" in kNN

    for j in range(1,11):
        # cross-validate (use part of the training data for training - and part for testing)
        #   first, create cross-validation data (here 3/4 train and 1/4 test)
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
        cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        knn.fit(cv_data_train, cv_target_train)
        score += knn.score(cv_data_test,cv_target_test)
        print("score is ", score)
    average_score = score/10

    print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
    print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))
    print("the average score is ", average_score)

    if average_score > best_score:
        best_score = average_score
        index = i

print("the best number of neighbors is ", index)    
print("the best score is ", best_score)


knn = KNeighborsRegressor(n_neighbors=index)
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 
knn.fit(cv_data_train, cv_target_train)

#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

# this next line is where the full training data is used for the model
knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("digit_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)



for i in range(len(df.index)):
    if df["Hue"].iloc[i] == -1:
        df["Hue"].iloc[i] = knn.predict(X_test[i])[0]

print(df)




"""
Here are our results from our analysis of best neighbors:

the best number of neighbors is  6
the best score is  0.242464594835

Created and trained a knn classifier
digit_X_test's predicted outputs are
[ 1.01166667  1.09166667  0.61666667  1.075       0.65166667  1.195       0.835
  1.04        1.05        0.895       1.12833333  1.01833333  1.09        1.025
  1.09833333  0.94666667  1.16333333  1.13166667  1.01666667  0.81
  1.02166667  1.22333333  1.17333333  0.975       0.75        0.595       0.795
  1.05666667  1.08833333  1.09666667]
and the actual labels are
[ 1.09  1.09  0.57  0.75  0.61  1.19  1.13  1.11  0.96  0.57  1.02  0.9
  0.82  1.05  1.09  0.91  1.25  1.12  1.04  0.81  0.7   1.15  1.22  0.96
  0.89  0.72  0.77  1.16  1.23  1.19]

     Color Intensity       Hue  OD280/OD315 of diluted wines  Proline
0               5.75  1.011667                          3.17     1510
1               7.30  1.091667                          2.88     1310
2               4.50  0.616667                          3.52      770
3               4.80  1.075000                          3.22     1195
4               3.95  0.651667                          2.77     1285
5               3.70  1.195000                          2.69     1020
6               4.90  0.835000                          3.44     1065
7               8.90  1.040000                          3.10     1260
8               7.05  1.050000                          3.26     1190
9               5.75  0.895000                          1.59      450
10              3.05  1.128333                          1.82      870
11              3.35  1.018333                          3.50      985
12              2.65  1.090000                          2.52      500
13              2.57  1.025000                          3.13      463
14              2.60  1.098333                          3.21      562
15              2.15  0.946667                          3.30      290
16              2.45  1.163333                          2.77      562
17              2.70  1.131667                          3.02      312
18              2.90  1.016667                          2.81      562
19              3.94  0.810000                          2.84      352
20              2.12  1.021667                          2.78      342
21              5.40  1.223333                          1.42      530
22              5.00  1.173333                          1.29      600
23              7.65  0.975000                          1.86      625
24             10.80  0.750000                          1.47      480
25              7.60  0.595000                          1.55      640
26              7.90  0.795000                          1.48      725
27             10.68  1.056667                          1.56      695
28              8.50  1.088333                          1.92      630
29              7.30  1.096667                          1.56      750

"""



