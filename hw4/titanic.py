#
# Abby, Eliana, and Liz
# titanic.py
# SEE GITHUB 
# https://github.com/abbyschantz/cs35/tree/master/hw4
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
df = df.drop('cabin', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)

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


def tr_embark(s):
	d = {'S': 0, 'C': 1, 'Q': 2}
	return d[s]
df['embarked'] = df['embarked'].map(tr_embark)
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
print("COL 1", X_data[:,1])
X_data[:,1] *= 100   # maybe the fourth column is worth much more!
print("COL 2", X_data[:,2])
X_data[:,2] *= 100
print("COL 5", X_data[:,5])
X_data[:,5] *= 100





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


We got a k value of 1 and an average corss-validation score of 0.755648535565.


      pclass  survived  sex   age  sibsp  parch      fare  embarked
0          3         1    0  24.0      0      0    7.0500         0
1          3         1    0  25.0      0      0    7.0500         0
2          3         1    0  35.0      0      0    8.0500         0
3          3         0    0  18.0      0      0    8.3000         0
4          3         0    1  19.0      1      0    7.8542         0
5          3         1    0  32.0      0      0   22.5250         0
6          1         0    0  24.0      0      1  247.5208         1
7          1         1    1  50.0      0      1  247.5208         1
8          2         0    0   1.0      2      1   39.0000         0
9          2         1    1   4.0      2      1   39.0000         0
10         2         1    1  12.0      2      1   39.0000         0
11         2         1    1  36.0      0      3   39.0000         0
12         2         1    0  34.0      0      0   13.0000         0
13         2         0    1  19.0      0      0   13.0000         0
14         2         0    0  23.0      0      0   13.0000         0
15         2         1    0  26.0      0      0   13.0000         0
16         2         1    0  42.0      0      0   13.0000         0
17         2         0    0  27.0      0      0   13.0000         0
18         3         0    0   1.0      1      2   20.5750         0
19         1         0    0  53.0      1      1   81.8583         0
20         1         0    0   4.0      0      2   81.8583         0
21         1         1    1  54.0      1      1   81.8583         0
22         3         1    0  40.5      0      0   15.1000         0
23         1         1    1  24.0      3      2  263.0000         0
24         1         0    1  28.0      3      2  263.0000         0
25         1         1    1  23.0      3      2  263.0000         0
26         1         1    0  19.0      3      2  263.0000         0
27         1         0    0  64.0      1      4  263.0000         0
28         1         1    1  60.0      1      4  263.0000         0
29         2         1    0  24.0      2      0   73.5000         0



"""