

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

def first_file():
	original_dir = os.getcwd()
	os.chdir("addresses")
	words = ['war']
	file = open("1789.txt", "r")
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
		return ["1789.txt", count]

""" PROBLEM 3 PART 2 filename() gives answer for which has most war"""
def filename():
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
	for let in string:
		if let in letters:
			newString += let
	return newString


""" PROBLEM 3 PART 3 fours() gives answer for which has most war"""
def fours():
	most = ["0000.txt", 0]
	for filename in os.listdir("addresses"):
		if filename.endswith(".txt"):
			new = four_words(filename)
			bigger = compare_two(most, new)
			most = bigger
	return most



