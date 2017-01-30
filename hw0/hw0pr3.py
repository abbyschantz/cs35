""" Liz Harder, Eliana Keinan, Abby Schantz"""

from collections import defaultdict
import os
import os.path

""" PROBLEM 3 PART 1 compare_words() gives answer for 2009 v 2013 with country """
def compare_words():
	co09 = country09()
	co13 = country13()
	if co09 > co13:
		return "The 2009 address uses the word country more times" 
	elif co13 > co09:
		return "The 2013 address uses the word country more times" 
	else: 
		return "The 2009 and 2013 address use the word country an equal number of times"


def country09():
	original_dir = os.getcwd()
	os.chdir("addresses")
	words = ['country']
	file = open("2009.txt", "r")
	os.chdir(original_dir)
	text = file.read()
	text = text.replace(',',' ')
	text = text.replace('.',' ')
	text = text.lower()
	for compare_words in words:
		count = 0
		for word in text.split():
			if compare_words.lower() == word:
				count = count + 1
		return count

def country13():
	original_dir = os.getcwd()
	os.chdir("addresses")
	words = ['country']
	file = open("2013.txt", "r")
	os.chdir(original_dir)
	text = file.read()
	text = text.replace(',',' ')
	text = text.replace('.',' ')
	text = text.lower()
	for compare_words in words:
		count = 0
		for word in text.split():
			if compare_words.lower() == word:
				count = count + 1
		return count


def war_count(filename):
	original_dir = os.getcwd()
	os.chdir("addresses")
	words = ['war']
	file = open(filename, "r")
	os.chdir(original_dir)
	text = file.read()
	text = text.replace(',',' ')
	text = text.replace('.',' ')
	text = text.lower()
	for compare_words in words:
		count = 0
		for word in text.split():
			if compare_words.lower() == word:
				count = count + 1
		return [filename, count]


def compare_two(filename1, filename2):
	
	filename1number = filename1[1]
	filename2number = filename2[1]
	if filename1number > filename2number:
		return filename1
	else:
		return filename2



""" PROBLEM 3 PART 2 war() gives answer for which has most war"""
def war():
	most = ["0000.txt", 0]
	for filename in os.listdir("addresses"):
		if filename.endswith(".txt"):
			new = war_count(filename)
			bigger = compare_two(most, new)
			most = bigger
	return most


""" words with four or more letters """ 
def four_words(filename):
	original_dir = os.getcwd()
	os.chdir("addresses")
	file = open(filename, "r")
	os.chdir(original_dir)
	text = file.read()
	text = text.lower()
	text = text.split()
	count = 0
	for word in text:
		word = get_only_letters(word)
		if len(word) == 4:
			count += 1
	return [filename, count]


def get_only_letters(list):
	letters = "abcdefghijklmnopqrstuvwxyz"
	newString = ''
	for let in list:
		if let in letters:
			newString += let
	return newString


""" PROBLEM 3 PART 3 fours() gives answer for which has most 4 letter words"""
def fours():
	most = ["0000.txt", 0]
	for filename in os.listdir("addresses"):
		if filename.endswith(".txt"):
			new = four_words(filename)
			bigger = compare_two(most, new)
			most = bigger
	return most


def countquestions(filename):
	""" counts the number of questions in a specific address
	"""
	original_dir = os.getcwd()
	os.chdir("addresses")

	file = open(filename, "r")
	text = file.read()

	os.chdir(original_dir)

	count = 0
	for word in text:
		if word[-1] == '?':
			count += 1
		
	return count


def mostquestions():
	""" returns how many questions each address has
	"""

	L = os.listdir("addresses") 

	d = {}
	count = 0


	for filename in L:
		if filename.endswith(".txt"):
			if filename not in d:
				d[filename] = countquestions(filename)


	most = max(d.values())

	return most


""" WE V I """ 

def compare_two_most(filename1, filename2, iwe):
	
	filename1number = filename1[iwe]
	filename2number = filename2[iwe]
	if filename1number > filename2number:
		return filename1
	else:
		return filename2

def compare_two_least(filename1, filename2, iwe):
	
	filename1number = filename1[iwe]
	filename2number = filename2[iwe]
	if filename1number < filename2number:
		return filename1
	else:
		return filename2


def i_v_we(filename):
	original_dir = os.getcwd()
	os.chdir("addresses")
	file = open(filename, "r")
	os.chdir(original_dir)
	text = file.read()
	text = text.lower()
	text = text.split()
	wes = 0
	eyes = 0
	for word in text:
		if word == "we":
			wes += 1
		if word == "i":
			eyes += 1
	"""print(filename, "-> I:", eyes, "We:", wes)"""
	return [filename, eyes, wes]

def pronouns():
	L = L = os.listdir("addresses") 
	mostI = ["0000.txt", 0, 0]
	mostWe = ["0000.txt", 0, 0]
	leastI = ["0000.txt", 9000000, 9000000]
	leastWe = ["0000.txt", 9000000, 9000000]
	for filename in L:
		if filename.endswith(".txt"):
			new = i_v_we(filename)
			biggerI = compare_two_most(mostI, new, 1)
			mostI = biggerI
			biggerWe = compare_two_most(mostWe, new, 2)
			mostWe = biggerWe
			smallerI = compare_two_least(leastI, new, 1)
			leastI = smallerI
			smallerWe = compare_two_least(leastWe, new, 2)
			leastWe = smallerWe
	return [mostI, mostWe, leastI, leastWe]

def length(filename):
	file = open(filename, "r")
	text = file.read()

	return len(text)


def longest():
	L = os.listdir("addresses") 
	os.chdir( "addresses" )

	d = {}

	for filename in L:
		if filename.endswith(".txt"):
			if filename not in d:
				d[filename] = length(filename)
	os.chdir("..")

	longest = max(d.values())

	return longest



def main():
	print (compare_words())
	warAnswer = war()
	print ('The year ', warAnswer[0], ' has the most uses of the war war totaling ', warAnswer[1])
	fourAnswer = fours()
	print ('The year ', fourAnswer[0], ' has the most four letter words totaling ', fourAnswer[1])
	print('The year 1861 has the most number of questions totaling ' , mostquestions())

	pros = pronouns()
	print('In, ',pros[0][0], 'of all the addresses, the most I\'s were used, ',pros[0][1])
	print('While in, ',pros[2][0], 'of all the addresses, the fewest I\'s were used, ',pros[2][1])
	print('In contrast, in', pros[1][0], 'there was the greatest use of the word we at', pros[1][2])
	print('Finally, in', pros[3][0], 'the fewest uses of the word we came in at', pros[3][2])
	print('To see full output of uses, uncomment the print statement in i_v_we')
	print('The most words used in a speech was', longest())


""" 
1. 2013 has more uses of the word country than 2009.
2. The word war is used the most in 1821. It is used 16 times.
3. 1841 has the most four letter words with a total of 1000. 
4. 1861 has the most questions with a total of 22. This is interesting because most addresses have less than 5.
"""


