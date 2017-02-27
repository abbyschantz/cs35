#
#
# titanic.py
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
# Here's an example of dropping the 'body' column:
df = df.drop('body', axis=1)  # axis = 1 means column
df = df.drop('boat', axis=1)

# let's drop all of the rows with missing data:
df = df.dropna()

# let's see our dataframe again...
# I ended up with 1001 rows (anything over 500-600 seems reasonable)
df.head()
df.info()



# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

# let's see our dataframe again...
df.head()
df.info()


# you will need others!


print("+++ end of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
X_data = df.drop('survived', axis=1).values        # everything except the 'survival' column
y_data = df[ 'survived' ].values      # also addressable by column name(s)

# We'll stick with numpy - here's the conversion to a numpy array
X_data_full = df.iloc[:,0:14].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ 'survived' ].values      # also addressable by column name(s)

# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
# we can drop the initial (unknown) rows -- if we want to test with known data
X_data_full = X_data_full[44:,:]   # 2d array
y_data_full = y_data_full[44:]     # 1d column
#

# we can scramble the remaining data if we want - only if we know the test set's labels 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]


# The first nine are our test set - the rest are our training
X_test = X_data_full[0:44,0:14]              # the final testing data
X_train = X_data_full[44:,0:14]              # the training data

y_test = y_data_full[0:44]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[44:]                  # the training outputs/labels (known)






# feature engineering...
#X_data[:,0] *= 100   # maybe the first column is worth much more!
#X_data[:,3] *= 100   # maybe the fourth column is worth much more!




#
# the rest of this model-building, cross-validation, and prediction will come here:
#     build from the experience and code in the other two examples...
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
print("digit_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)



for i in range(len(df.index)):
    if df["survived"].iloc[i] == -1:
        df["survived"].iloc[i] = knn.predict(X_test[i])[0]

print(df)


"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how high were you able to get the average cross-validation (testing) score?



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:




And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Past those labels (just labels) here:
You'll have 10 lines:



"""