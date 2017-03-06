#
#
# titanic5.py
# GITHUB REPO: https://github.com/abbyschantz/cs35/tree/master/hw5
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd
from sklearn import tree
from sklearn import ensemble

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('fare', axis=1)
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)



# let's drop all of the rows with missing data:
df = df.dropna()


# converts string data of sex to numerical data, 0 and 1
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
X_data = df.drop('survived', axis=1).values        # everything except the 'survival' column
y_data = df[ 'survived' ].values      # also addressable by column name(s)

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
X_data_full = X_data[42:,:]
y_data_full = y_data[42:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:42,0:12]             # the final testing data
X_train = X_data_full[42:,0:12]             # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]                  # the training outputs/labels (known)

# feature engineering...
X_data[:,0] *= 550   # pclass
X_data[:,1] *= 450   # sex
X_data[:,2] *= 25  	 # age
X_data[:,3] *= 20    # sibs
X_data[:,4] *= 1     # parch

print('\n')
print("DT ANALYSIS\n")
#
# cross-validation to determine the Decision Tree's parameter (to find max_depth)
#
max_depth=10
#

depth = 0
best_score = 0

for n in range(1,max_depth+1):

    dtree = tree.DecisionTreeClassifier(max_depth=n)
    avg_score = 0
        
    for i in range(10):  # runs 10 times
        #
        # split into our cross-validation sets...
        #
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        dtree = dtree.fit(cv_data_train, cv_target_train) 
        avg_score += dtree.score(cv_data_test,cv_target_test)
    
    avg_score = avg_score / 10

    print(n, ":", avg_score)
    #print("Average DT testing score at depth ++",n,"++ was: ", avg_score)

    if best_score < avg_score:
        best_score = avg_score
        depth = n

print("The best DT score was ", best_score, "with a depth of ", depth)



#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#


#rerun with optimal depth
dtree = tree.DecisionTreeClassifier(max_depth=depth)
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 
dtree = dtree.fit(cv_data_train, cv_target_train) 
score = dtree.score(cv_data_test,cv_target_test)



print("DT score was ", score)
print("Feature importances:", dtree.feature_importances_)
print("Predicted outputs:")
print(dtree.predict(X_test))
print("and the actual outcomes are:")
print(y_test, "\n")

feature_names = df.columns.values
feature_names2 = []
for i in range(len(feature_names)):
    if feature_names[i] != "survived":
        feature_names2.append(feature_names[i])


target_names = ['0','1'] 
tree.export_graphviz(dtree, out_file='titanic_tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names2,  filled=True, rotate=False, # LR vs UD
                            class_names=target_names, leaves_parallel=True) 


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#
#
max_depth=6
#

depth = 0
best_score = 0

for n in range(1,max_depth+1):

    rforest = ensemble.RandomForestClassifier(max_depth=n, n_estimators=1000)
    avg_score = 0
        
    for i in range(10):  # run at least 10 times.... take the average testing score
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
        avg_score += rforest.score(cv_data_test,cv_target_test)
    
    avg_score = avg_score / 10

    print(n, ":", avg_score)

    if best_score < avg_score:
        best_score = avg_score
        depth = n

print("The best RT score was ", best_score, "with a depth of ", depth)



X_test = X_data_full[0:42,0:12]              # the final testing data
X_train = X_data_full[42:,0:12]              # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]

max_depth = depth
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2)
rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=1000)

rforest = rforest.fit(X_train, y_train) 
score = rforest.score(cv_data_test,cv_target_test)
print("RT test score:", score)
print("RT feature importances:", rforest.feature_importances_) 

print("The predicted outputs are")
print(rforest.predict(X_test),"\n")

print("and the actual labels are")
print(y_test)


############################################ AGES ###########################################################

df = pd.read_csv('titanic5.csv', header=0)
df.head()
df.info()

df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('fare', axis=1)
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)
df = df[np.isfinite(df['pclass'])]
df = df[np.isfinite(df['survived'])]

df['sex'] = df['sex'].map(tr_mf)

df = df[np.isfinite(df['sex'])]
df = df[np.isfinite(df['sibsp'])]
df = df[np.isfinite(df['parch'])]

#replace all NaN ages with -1
df['age'].fillna(-1, inplace=True)
df = df.sort('age', ascending=True)
df['age'] = df['age'].astype(int)

count = 0
for i in range(len(df['age'])):
    if df['age'][i] == -1:
        count +=1


# extract the underlying data with the values attribute:
X_data = df.drop('age', axis=1).values        # everything except the 'age' column
y_data = df[ 'age' ].values

X_data_full = X_data[count:,:]
y_data_full = y_data[count:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:count,0:5]              # the final testing data
X_train = X_data_full[count:,0:5]              # the training data
y_test = y_data_full[0:count]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[count:]                  # the training outputs/labels (known)


X_data[:,1] *= 500   #survived

cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
    cross_validation.train_test_split(X_train, y_train, test_size=0.2)
rforest = ensemble.RandomForestClassifier(max_depth=depth, n_estimators=1000)

rforest = rforest.fit(X_train, y_train) 
score = rforest.score(cv_data_test,cv_target_test)
print("Inputed test score:", score)
print("Inputed feature importances:", rforest.feature_importances_) 

predicted_values =  rforest.predict(X_test)
print("The predicted outputs are")
print(predicted_values,"\n")

print("and the actual labels are")
print(y_test)

for i in range(len(df['age'])):
    if df['age'][i] == -1:
        df['age'][i] = predicted_values[0]
        predicted_values = predicted_values[1:]

X_data = df.drop('survived', axis=1).values
y_data = df[ 'survived' ].values


X_data_full = X_data[42:,:]
y_data_full = y_data[42:]

indicies = np.random.permutation(len(X_data_full))
X_data_full = X_data_full[indicies]
y_data_full = y_data_full[indicies]


X_test = X_data_full[0:42,0:12]             # the final testing data
X_train = X_data_full[42:,0:12]             # the training data
y_test = y_data_full[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[42:]                  # the training outputs/labels (known)

# feature engineering...
X_data[:,0] *= 550  #pclass
X_data[:,1] *= 430  #sex
X_data[:,2] *= 25  	#age
X_data[:,3] *= 20   #sibs
X_data[:,4] *= 1   	#parch



dtree = tree.DecisionTreeClassifier(max_depth=depth)
cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 
dtree = dtree.fit(cv_data_train, cv_target_train) 
score = dtree.score(cv_data_test,cv_target_test)



print("Revised DT score with inputed ages and a depth of ",depth, "was ", score)
print("Feature importances:", dtree.feature_importances_)
print("Predicted outputs are")
print(dtree.predict(X_test))
print("and the actual outcomes are")
print(y_test, "\n")


"""
Our DT results:
The best DT score was  0.802139037433 with a depth of  4
DT score was  0.818181818182
Feature importances: [ 0.19004027  0.61164342  0.12895005  0.06936626  0.        ]
Predicted outputs:
[1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0
 0 0 0 0 0]
and the actual outcomes are:
[1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 0 0 0 0 1 0 0 0 1 0
 0 1 0 0 0]


Our RT results:
The best RT score was  0.810695187166 with a depth of  4
RT test score: 0.83422459893
RT feature importances: [ 0.2064859   0.56451728  0.1141947   0.05270007  0.062102
05]
The predicted outputs are
[0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0
 0 1 1 0 0]

and the actual labels are
[0 1 0 0 0 1 0 0 0 0 1 0 1 0 1 0 1 1 1 1 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 0
 0 1 1 0 0]

"""
