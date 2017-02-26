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

#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#



# feature engineering...
#X_data[:,0] *= 100   # maybe the first column is worth much more!
#X_data[:,3] *= 100   # maybe the fourth column is worth much more!




#
# the rest of this model-building, cross-validation, and prediction will come here:
#     build from the experience and code in the other two examples...
#




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