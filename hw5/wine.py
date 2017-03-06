#
# read iris data
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

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_orig = df.iloc[:,0:13].values        # iloc == "integer locations" of rows/cols
y_data_orig = df[ 'Hue' ].values      # individually addressable columns (by name)
feature_names = df.columns.values          # get the names into a list!

X_data = df.drop('Hue', axis=1).values        # everything except the 'hue' column
y_data = df[ 'Hue' ].values      # also addressable by column name(s)

X_data_full = X_data_orig[0:,:]  # make the 10 into 0 to keep all of the data
y_data_full = y_data_orig[0:]    # same for this line


targetsList = []
for hue in y_data_full:
    if not hue in targetsList:
        if hue == -1:
            continue
        targetsList.append(str(hue))

target_names = targetsList   # and a list of the labels...
print("target_name", target_names)
#
# cross-validation and scoring to determine parameters...
# 

#
# we can scramble the data - but only if we know the test set's labels!
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full[0:10,0:13]              # the final testing data
X_train = X_data_full[10:,0:13]              # the training data

y_test = y_data_full[0:10]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[10:]                  # the training outputs/labels (known)

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

X_test = X_data_orig[0:9,0:13]              # the final testing data
X_train = X_data_orig[9:,0:13]              # the training data

y_test = y_data_orig[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_data_orig[9:]                  # the training outputs/labels (known)

#
# show the creation of three tree files (three max depths)
#
best = 0
for max_depth in [1,2,3,4]:
    #
    # we'll use max_depth between 1 and 3
    #
    dtree = tree.DecisionTreeRegressor(max_depth=max_depth)

    # this next line is where the full training data is used for the model
    dtree = dtree.fit(X_data_full, y_data_full) 
    print("\nCreated and trained a knn classifier")  #, knn

    #
    # write out the dtree to tree.dot (or another filename of your choosing...)
    tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True)  
    # the website to visualize the resulting graph (the tree) is at www.webgraphviz.com
    #


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
X_test = X_data_full[0:10,0:13]              # the final testing data
X_train = X_data_full[10:,0:13]              # the training data

y_test = y_data_full[0:10]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[10:]                  # the training outputs/labels (known)


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#
#


# adapt for cross-validation (at least 10 runs w/ average test-score)
best_score = 0
index = 0
for i in range(1, 11):
    for j in range(1,11):
        score = 0
        rforest = ensemble.RandomForestRegressor(max_depth=i, n_estimators=10)

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
        score += dtree.score(cv_data_test,cv_target_test)
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

X_test = X_data_orig[0:9,0:13]              # the final testing data
X_train = X_data_orig[9:,0:13]              # the training data

y_test = y_data_orig[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_data_orig[9:]                  # the training outputs/labels (known)

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





