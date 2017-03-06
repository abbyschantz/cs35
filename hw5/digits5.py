#
#names: Eliana Keinan, Liz Harder, Abby Schantz 
# Final Copy
#read digits data
#

from sklearn.datasets import load_digits
digits = load_digits()


import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
import pandas as pd

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('digits5EC.csv', header=0)    # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_orig = df.iloc[:,0:63].values        # iloc == "integer locations" of rows/cols
y_data_orig = df[ '64' ].values      # individually addressable columns (by name)
feature_names = df.columns.values          # get the names into a list!
target_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']   # and a list of the labels...
print ("target names are,", target_names)


X_data_full = X_data_orig[0:,:]  # make the 10 into 0 to keep all of the data
y_data_full = y_data_orig[0:]    # same for this line


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
X_test = X_data_full[0:10,0:63]              # the final testing data
X_train = X_data_full[10:,0:63]              # the training data

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
    dtree = tree.DecisionTreeClassifier(max_depth=i)
    for j in range(1,11):  # run at least 10 times.... take the average cv testing score
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
        #print("score is ", score)
    average_score = score/10
    print(i,average_score)
    if average_score > best_score:
        best_score = average_score
        index = i
print("DT best score is ", best_score)

#import sys
#sys.exit(0)

# dtree.feature_importances_  [already computed]


#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

X_test = X_data_orig[0:9,0:63]              # the final testing data
X_train = X_data_orig[9:,0:63]              # the training data

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
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # this next line is where the full training data is used for the model
    dtree = dtree.fit(X_data_full, y_data_full) 
    print("\nCreated and trained a decision tree classifier")  #, knn


    # write out the dtree to tree.dot (or another filename of your choosing...)
    tree.export_graphviz(dtree, out_file='tree' + str(max_depth) +'.dot',   # constructed filename!
                            feature_names=feature_names,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True)  
    # the website to visualize the resulting graph (the tree) is at www.webgraphviz.com
    #



# here are some examples, printed out:
print("digit_X_test's predicted outputs are")
print(dtree.predict(X_test))

# and here are the actual labels (digit)
print("and the actual labels are")
print(y_test)


X_test = X_data_full[0:10,0:64]              # the final testing data
X_train = X_data_full[10:,0:64]              # the training data

y_test = y_data_full[0:10]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[10:]                  # the training outputs/labels (known)


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#
#
#rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=10)


# adapt for cross-validation (at least 10 runs w/ average test-score)

if True:
    dList2 = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    avgTests = []
    for max_depth in dList2:
        rforest = ensemble.RandomForestClassifier(max_depth=max_depth) 
        avg_test = 0
        avg_train = 0  
        for i in range(10):
            #
            # split into our cross-validation sets...
            #
            cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
                cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

            # fit the model using the cross-validation data
            #   typically cross-validation is used to get a sense of how well it works
            #   and tune any parameters, such as the max_depth and n_estimators here...
            rforest = rforest.fit(cv_data_train, cv_target_train) 
            #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
            #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
            avg_train += dtree.score(cv_data_train,cv_target_train)
            avg_test += dtree.score(cv_data_test,cv_target_test)

        avg_train = avg_train/10
        avg_test = avg_test/10
        avgTests.append(avg_test)
        #print("Value:", max_depth, "Average Train:", avg_train)
        #print("Value:", max_depth, "Average Test:", avg_test)
        #print("avgList:", avgTests)
    bestAvg = avgTests.index(max(avgTests))
    #print("bestAvg:", bestAvg)
    bestd2 = dList2[bestAvg]
    #print("best Depth:", bestd2)
# rforest.estimators_  [a list of dtrees!]
if True:
    dList3 = [25,50,75,100,125,150,175,200]
    avgTests = []
    for n in dList3:
        rforest = ensemble.RandomForestClassifier(max_depth=bestd2, n_estimators=n) 
        avg_test = 0
        avg_train = 0  
        for i in range(10):
            #
            # split into our cross-validation sets...
            #
            cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
                cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

            # fit the model using the cross-validation data
            #   typically cross-validation is used to get a sense of how well it works
            #   and tune any parameters, such as the max_depth and n_estimators here...
            rforest = rforest.fit(cv_data_train, cv_target_train) 
            #print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
            #print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))
            avg_train += dtree.score(cv_data_train,cv_target_train)
            avg_test += dtree.score(cv_data_test,cv_target_test)

        avg_train = avg_train/10
        avg_test = avg_test/10
        avgTests.append(avg_test)
        print("Value:", max_depth, "Average Train:", avg_train)
        print("Value:", max_depth, "Average Test:", avg_test)
        print("avgList:", avgTests)
    bestAvg = avgTests.index(max(avgTests))
    print("bestAvg:", bestAvg)
    bestn = dList3[bestAvg]
    print("best N:", bestn)

#
# we'll use max_depth == 2
#
max_depth = bestd2
n_estimators = bestn
rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=bestn)


#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

X_test = X_data_orig[0:10,0:64]              # the final testing data
X_train = X_data_orig[10:,0:64]              # the training data

y_test = y_data_orig[0:10]                  # the final testing outputs/labels (unknown)
y_train = y_data_orig[10:]                  # the training outputs/labels (known)

# this next line is where the full training data is used for the model
rforest = rforest.fit(X_train, y_train) 
print("\nCreated and trained a randomforest classifier") 

#
# feature importances
#
print("feature importances rf:", rforest.feature_importances_)  
print("feature importances dt:", dtree.feature_importances_)  



# here are some examples, printed out:
print("rforests test's predicted outputs are")
print(rforest.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)
print ("rforest max depth", max_depth)
print ("rforet n_estimators", n_estimators)


"""Extra Credit: 
I drew 10 digits and made them into png files. The using the code from the homework instructions, 
I resized the image into 8x8 pixels and then extract the 64 values (from 0 to 15). This code outputted 
a string of 64 digits which I then placed into the top of the digits5EC.csv file. I ran the digits5.py 
file normally except I replaced the original digits5.csv with digits5EC.csv in order to see the results. 

Decision Tree Model Results: 
digit_X_test's predicted outputs are
[4 3 5 3 3 0 4 3 6]
and the actual labels are
[4 9 5 2 8 0 1 2 6]

Random Forest Model Results:
rforests test's predicted outputs are
[4 9 0 3 2 0 4 2 6 0]
and the actual labels are
[4 9 5 2 8 0 1 2 6 0]

The results from both these tests show that...
5/10 correct in the Decision Tree 
6/10 correct in the Random Forest Results

"""