#
# Names: Eliana Keinan, Abby Schantz, Liz Harder
# digits.py
#
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
df.head()
df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the column
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data_full = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ 'label' ].values      # also addressable by column name(s)


# we can drop the initial (unknown) rows -- if we want to test with known data
X_data_full = X_data_full[22:,:]   # 2d array
y_data_full = y_data_full[22:]     # 1d column


# we can scramble the remaining data if we want - only if we know the test set's labels 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

# The first nine are our test set - the rest are our training
X_test = X_data_full[0:22,0:63]              # the final testing data
X_train = X_data_full[22:,0:63]              # the training data

y_test = y_data_full[0:22]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[22:]                  # the training outputs/labels (known)



#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show(block=False)
    
# try it!
row = 3
Pixels = X_data_full[row:row+1,:]
show_digit(Pixels)
print("That image has the label:", y_data_full[row])


#
# feature engineering...
#

# here is where you can re-scale/change column values...
# X_data[:,0] *= 100   # maybe the first column is worth 100x more!
# X_data[:,3] *= 100   # maybe the fourth column is worth 100x more!


#
# here, you'll implement the kNN model and cross validation
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
    if df["64"].iloc[i] == -1:
        df["64"].iloc[i] = knn.predict(X_test[i])[0]

print(df)

"""
Comments and results:

Adapting from the iris data set was pretty smooth. The main changes we had to
make were regarding the data set sizing at the beginning of the code.

The results of our function were that the best number of neighbors is
3, which produced the best score of 0.9856.


Predicted labels of missing values with full data:
    7, 7, 2, 8, 8, 6, 8, 0, 7, 1, 0, 3

Predicted labels of partially erased data:
    8, 9, 3, 2, 2, 4, 2, 0, 1, 3


Full output, for reference:
         62  63       64     label
    0      0   0  digit 7  digit -1
    1      0   0  digit 7  digit -1
    2      0   0  digit 2  digit -1
    3      0   0  digit 8  digit -1
    4      0   0  digit 8  digit -1
    5      0   0  digit 6  digit -1
    6      0   0  digit 8  digit -1
    7      0   0  digit 0  digit -1
    8      0   0  digit 7  digit -1
    9      0   0  digit 1  digit -1
    10    11   0  digit 0  digit -1
    11    13   2  digit 3  digit -1
    12     0   0  digit 8  digit -1
    13     0   0  digit 9  digit -1
    14     3   0  digit 3  digit -1
    15     0   0  digit 2  digit -1
    16     0   0  digit 2  digit -1
    17    13   0  digit 4  digit -1
    18     0   0  digit 2  digit -1
    19     1   0  digit 0  digit -1
    20     0   0  digit 1  digit -1
    21     0   0  digit 3  digit -1


"""