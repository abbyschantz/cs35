from collections import defaultdict
import os
import os.path

def filename():
	original_dir = os.getcwd()
	count = 0
	phone_files = os.listdir("phone_files")
	os.chdir("phone_files")
	for i in phone_files:
		if i == ".DS_Store":
			print "in ds store"
		else:
			current = str(i)
			for files in os.listdir(current):

				count += 1
	return count



def get_ten(filename):
	file = open(filename, "r")
	text = file.read()
	print text
	count = 0
	word = get_only_numbers(text)
	if len(word) == 10:
		return True

def get_only_numbers(list):
	numbers = "0123456789"
	newString = ''
	for i in range(len(list)):
		if list[i] in numbers:
			newString += list[i]
	return newString

def ten_dig():
	original_dir = os.getcwd()
	count = 0
	phone_files = os.listdir("phone_files")
	os.chdir("phone_files")
	phone_dir = os.getcwd()
	for i in phone_files:
		if i == ".DS_Store":
			print "in ds store"
		else:
			current = str(i)
			for files in os.listdir(current):
				os.chdir(current)
				if get_ten(files):
					count += 1
				os.chdir(phone_dir)

				
	return count


def check_909(filename):
	file = open(filename, "r")
	text = file.read()
	word = get_only_numbers(text)
	if word[0:3] == '909':
		return True


def niners():
	original_dir = os.getcwd()
	count = 0
	phone_files = os.listdir("phone_files")
	os.chdir("phone_files")
	phone_dir = os.getcwd()
	for i in phone_files:
		if i == ".DS_Store":
			print "in ds store"
		else:
			current = str(i)
			for files in os.listdir(current):
				os.chdir(current)
				if check_909(files):
					count += 1
				os.chdir(phone_dir)
			
	return count
	

