#
# hw3pr2.py 
#
# Person or machine?  The rps-string challenge...
#
# This file should include your code for 
#   + extract_features( rps ),               returning a dictionary of features from an input rps string
#   + score_features( dict_of_features ),    returning a score (or scores) based on that dictionary
#   + read_data( filename="rps.csv" ),       returning the list of datarows in rps.csv
#
# Be sure to include a short description of your algorithm in the triple-quoted string below.
# Also, be sure to include your final scores for each string in the rps.csv file you include,
#   either by writing a new file out or by pasting your results into the existing file
#   And, include your assessment as to whether each string was human-created or machine-created
# 
#

"""
Short description of (1) the features you compute for each rps-string and 
      (2) how you score those features and how those scores relate to "humanness" or "machineness"





"""


# Here's how to machine-generate an rps string.
# You can create your own human-generated ones!

import random

def gen_rps_string( num_characters ):
    """ return a uniformly random rps string with num_characters characters """
    result = ''
    for i in range( num_characters ):
        result += random.choice( 'rps' )
    return result

# Here are two example machine-generated strings:
rps_machine1 = gen_rps_string(200)
rps_machine2 = gen_rps_string(200)
# print those, if you like, to see what they are...




from collections import defaultdict

#
# extract_features( rps ):   extracts features from rps into a defaultdict
#
def extract_features( rps ):
    """ <include a docstring here!>
    """
    d = defaultdict( float )  # other features are reasonable
    number_of_s_es = rps.count('s')  # counts all of the 's's in rps
    d['s'] = 42                      # doesn't use them, however
    return d   # return our features... this is unlikely to be very useful, as-is






#
# score_features( dict_of_features ): returns a score based on those features
#
def score_features( dict_of_features ):
    """ <include a docstring here!>
    """
    d = dict_of_features
    random_value = random.uniform(0,1)
    score = d['s'] * random_value
    return score   # return a humanness or machineness score







#
# read_data( filename="rps.csv" ):   gets all of the data from "rps.csv"
#
def read_data( filename="rps.csv" ):
    """ <include a docstring here!>
    """
    # you'll want to look back at reading a csv file!
    List_of_rows = []   # for now...
    return List_of_rows





#
# you'll use these three functions to score each rps string and then
#    determine if it was human-generated or machine-generated 
#    (they're half and half with one mystery string)
#
# Be sure to include your scores and your human/machine decision in the rps.csv file!
#    And include the file in your hw3.zip archive (with the other rows that are already there)
#
