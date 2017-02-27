#
# Names: Eliana Keinan, Abby Schantz, Liz Harder
# iris.py
# SEE GITHUB
# https://github.com/abbyschantz/cs35/tree/master/hw4
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

print("+++ Start of pandas +++\n")
# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('iris.csv', header=0)    # read the file
df.head()                                 # first five lines
df.info()                                 # column details

# There are many more features to pandas...  Too many to cover here

# One important feature is the conversion from string to numeric datatypes!
# As input features, numpy and scikit-learn need numeric datatypes
# You can define a transformation function, to help out...
def transform(s):
    """ from string to number
          setosa -> 0
          versicolor -> 1
          virginica -> 2
    """
    d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
    return d[s]
    
# 
# this applies the function transform to a whole column
#
#df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_full = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ 'irisname' ].values      # individually addressable columns (by name)


#
# we can drop the initial (unknown) rows -- if we want to test with known data
X_data_full = X_data_full[9:,:]   # 2d array
y_data_full = y_data_full[9:]     # 1d column


#
# we can scramble the remaining data if we want - only if we know the test set's labels
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]



#
# The first nine are our test set - the rest are our training
#
X_test = X_data_full[0:9,0:4]              # the final testing data
X_train = X_data_full[9:,0:4]              # the training data

y_test = y_data_full[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[9:]                  # the training outputs/labels (known)



#
# feature engineering...
#


# here is where you can re-scale/change column values...
# X_data[:,0] *= 100   # maybe the first column is worth 100x more!
# X_data[:,3] *= 100   # maybe the fourth column is worth 100x more!




#
# create a kNN model and tune its parameters (just k!)
#   here's where you'll loop to run 5-fold (or 10-fold cross validation)
#   and loop to see which value of n_neighbors works best (best cv testing-data score)
#
from sklearn.neighbors import KNeighborsClassifier

test_list = []
best_score = 0
index = 0

for i in range(1,11):

    score = 0
    knn = KNeighborsClassifier(n_neighbors=i)   # i is the "k" in kNN

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
    average_score = score/10

    print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
    print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))
    print("the average score is ", average_score)

    if average_score > best_score:
        best_score = average_score
        index = i

print("the best number of neighbors is ", index)    
print("the best score is ", best_score)


knn = KNeighborsClassifier(n_neighbors=index)
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
print("iris_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


# 
# here is where you'll more elegantly format the output - for side-by-side comparison
#     then paste your results for the unknown irises below
#

for i in range(len(df.index)):
    if df["irisname"].iloc[i] == "unknown":
        df["irisname"].iloc[i] = knn.predict(X_test[i])[0]

print(df)



#
# for testing values typed in
#
def test_by_hand(knn):
    """ allows the user to enter values and predict the
        label using the knn model passed in
    """
    print()
    Arr = np.array([[0,0,0,0]]) # correct-shape array
    T = Arr[0]
    T[0] = float(input("sepal length? "))
    T[1] = float(input("sepal width? "))
    T[2] = float(input("petal length? "))
    T[3] = float(input("petal width? "))
    prediction = knn.predict(Arr)[0]
    print("The prediction is", prediction)
    print()


# import sys   # easy to add break points...
# sys.exit(0)


"""
Comments and results:

The best number of neighbors for our code was 5 neighbors,
which produced a score of 0.9696 accuracy.

The predicted values for the first 9 irises are shown below:

     sepallen  sepalwid  petallen  petalwid    irisname
0         5.7       2.8       4.1       1.3  versicolor
1         6.3       3.3       6.0       2.5   virginica
2         6.2       2.9       4.3       1.3  versicolor
3         5.1       2.5       3.0       1.1   virginica
4         5.4       3.4       1.5       0.4   virginica
5         5.2       4.1       1.5       0.1      setosa
6         5.8       2.7       5.1       1.9   virginica
7         5.7       2.9       4.2       1.3  versicolor
8         4.8       3.1       1.6       0.2  versicolor


"""
