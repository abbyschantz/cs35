from collections import defaultdict
import os
import os.path
import csv
import sys

def make_csv():
	f = open("sample.txt", 'wt')
	try:
		writer = csv.writer(f)
		writer.writerow( ('Last', 'First', 'phonenumber') )
		L = os.listdir("phone_files")
		os.chdir("phone_files")
		for foldername in L[1:]:
			files = os.listdir(foldername)
			os.chdir(foldername)
			for filename in files:
				if filename == ".DS_Store":
					print ("error: in ds store")
				else:
					print(filename)
					writer.writerow( (get_last_name(filename), get_first_name(filename), get_only_numbers(filename)) )
			os.chdir("..")
		os.chdir("..")
	finally:
		f.close()

		

def words_from_nums(filename):
	original_dir = os.getcwd()
	"""os.chdir("phone_test")"""
	file = open(filename, "r")
	os.chdir(original_dir)
	file = file.read()
	numbers = get_only_numbers(file)
	letters = get_only_letters(file)
	print(numbers)
	print (letters)


def get_only_letters(filename):
	letters = "abcdefghijklmnopqrstuvwxyz ,"
	newString = ''
	original_dir = os.getcwd()
	"""os.chdir("phone_test")"""
	file = open(filename, "r")
	os.chdir(original_dir)
	file = file.read()
	file = file.lower()
	file = file.strip()
	for i in range(len(file)):
		if file[i] in letters:
			newString += file[i]
	return newString

def get_only_numbers(filename):
	original_dir = os.getcwd()
	"""os.chdir("phone_test")"""
	file = open(filename, "r")
	os.chdir(original_dir)
	file = file.read()
	numbers = "0123456789"
	newString = ''
	for i in range(len(file)):
		if file[i] in numbers:
			newString += file[i]
	return newString

def get_first_name(filename):
	file = get_only_letters(filename)
	if ',' in file:
		file = (file.replace(" ", ""))
		file = (file.partition(",")[2])
		return file
	else:
		file = file.partition(" ")[0]
		file = (file.replace(" ", ""))
		return file

def get_last_name(filename):
	file = get_only_letters(filename)
	if ',' in file:
		file = (file.replace(" ", ""))
		file = (file.partition(",")[0])
		return file
	else:

		file = file.partition(" ")[2]
		file = (file.replace(" ", ""))
		return file


