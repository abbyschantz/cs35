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


def main():
	print (compare_words())
	warAnswer = war()
	print ('The year ', warAnswer[0], ' has the most uses of the war war totaling ', warAnswer[1])
	fourAnswer = fours()
	print ('The year ', fourAnswer[0], ' has the most four letter words totaling ', fourAnswer[1])

""" 
1. 2013 has more uses of the word country than 2009.
2. The word war is used the most in 1821. It is used 16 times.
3. 1841 has the most four letter words with a total of 1000. 
"""


