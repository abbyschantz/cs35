#
# hw3pr2.py 
# Names: Eliana Keinan, Abby Schantz, Liz Harder
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
For our project, we determine if a string is machine-generated or human-generated
based on the probablity of the strings having consecutive letters that are the 
same.  Thus we use the probability of multiple letter strings to say that if there
are long strings of the same letter, it is more likely human. Then we use a cutoff
value of 150 "points" to determine if a string is human or machine





"""


# Here's how to machine-generate an rps string.
# You can create your own human-generated ones!

import random
import csv

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
	
	#d = defaultdict( float )  # other features are reasonable
	#number_of_s_es = rps.count('s')  # counts all of the 's's in rps
	#d['s'] = 42                      # doesn't use them, however
	#return d   # return our features... this is unlikely to be very useful, as-is

	#d = defaultdict( float )  # other features are reasonable
	"""
	d = {}
	
	length = 1
	for i in range(len(rps)):
		if i == len(rps)-1:
			#print(rps[i])
			if length in d:
				d[length] += 1
			else:
				d[length] = 1
		elif rps[i] == rps[i+1]:
			length += 1
		else:
			if length in d:
				d[length] += 1
			else:
				d[length] = 1
			length = 1
	return d





#
# score_features( dict_of_features ): returns a score based on those features
#
def score_features( dict_of_features ):
	""" this function scores the features of the string to determine if it 
		is more or less human based on how long strings are. positive numbers 
		are more human, negative numbers are more machine
	"""
	d = dict_of_features
	human_points = 0
	machine_points = 0
	
	for key in d:
		prob = (1/3)**(key)
		
		human_points += 1/prob * d[key]
		machine_points += 3*d[key]

	
	human_score = human_points - machine_points

	return human_score   # return a humanness score


#
# read_data( filename="rps.csv" ):   gets all of the data from "rps.csv"
#
def read_data( filename="rps.csv" ):
	""" readcsv takes as
		 + input:  csv_file_name, the name of a csv file
		and returns
		 + output: a list of lists, each inner list is one row of the csv
		   all data items are strings; empty cells are empty strings
	"""
	
	try:
		csvfile = open( filename, newline='' )  # open for reading
		csvrows = csv.reader( csvfile )              # creates a csvrows object

		all_rows = []                               # we need to read the csv file
		for row in csvrows:                         # into our own Python data structure
			#all_rows.append( row[3] )                  # adds only the word to our list
			all_rows.append(row)

		del csvrows                                  # acknowledge csvrows is gone!
		csvfile.close()                              # and close the file
		return all_rows                              # return the list of lists

	except FileNotFoundError as e:
		print("File not found: ", e)
		return []


def write_to_csv( list_of_rows, filename ):
    """ readcsv takes as
         + input:  csv_file_name, the name of a csv file
        and returns
         + output: a list of lists, each inner list is one row of the csv
           all data items are strings; empty cells are empty strings
    """
    try:
        csvfile = open( filename, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")


def main():

	""" this function runs the algorithm and outputs an amended csv file
		with the results of if it's a human or a machine
	"""

	L = read_data()
	print(L)
	for i in L:
		score = score_features(extract_features(i[3]))
		print(score)
		i[2] = score
		if score > 200:
			i[1] = "human"
		else:
			i[1] = "machine"
	
	write_to_csv(L,"rps.csv")
	
	return L


def batch_play(rps1,rps2):
	""" takes tow rps-strings and plays them against each other
		by comparing the gestures corresponding in each of the
		two strings and returns which string wins more often (1 or 2)
	"""
	count1 = 0
	count2 = 0

	for i in range(len(rps1)):
		if rps1[i] == 'r' and rps2[i] == 's':
			count1 += 1
		elif rps1[i] == 's' and rps2[i] == 'p':
			count1 += 1
		elif rps1[i] == 'p' and rps2[i] == 'r':
			count1 += 1
		elif rps2[i] == 'r' and rps1[i] == 's':
			count2 += 1
		elif rps2[i] == 's' and rps1[i] == 'p':
			count2 += 1
		elif rps2[i] == 'p' and rps1[i] == 'r':
			count2 += 1


	if count1 > count2:
		return 1
	elif count2 > count1:
		return 2
	else:
		return 0


def winner():
	""" this function returns the index number of the string that wins the most
		number of times against the other strings in the csv file
	"""

	L = read_data()
	wins_list = []
	for i in L:
		wins = 0
		for j in L:
			if batch_play(i[3],j[3]) == 1:
				wins += 1
		wins_list += [wins]
	
	return wins_list.index(max(wins_list))

	

#
# you'll use these three functions to score each rps string and then
#    determine if it was human-generated or machine-generated 
#    (they're half and half with one mystery string)
#
# Be sure to include your scores and your human/machine decision in the rps.csv file!
#    And include the file in your hw3.zip archive (with the other rows that are already there)
#
