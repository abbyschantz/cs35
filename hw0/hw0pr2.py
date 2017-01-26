# hw0pr2.pr
#
# Eliana Keinan, Liz Harder, Abby Schantz

import os
import os.path

def count_files():
    """ counts the number of .txt files in the folder
    """

    L = os.listdir("phone_files") 
    os.chdir( "phone_files" )
    
    count = 0

    for foldername in L[1:]:
        
        files = os.listdir(foldername)  
        os.chdir(foldername)

        for filename in files:
            count += 1

        os.chdir("..")

    os.chdir("..") 
    return count



def count_ten():
    """ returns the number of phone numbers that contian exactly 10 digits
    """

    L = os.listdir("phone_files") 
    os.chdir( "phone_files" )
    
    count = 0

    for foldername in L[1:]:
        files = os.listdir(foldername)  
        os.chdir(foldername)

        for filename in files:
            if count_digits(filename) == 10:
                count += 1

        os.chdir("..")

    os.chdir("..") 
    return count

def count_digits(filename):
    """ counts the number of digits in a phone number file
    """

    file = open(filename,"r")
    text = file.read()
    count = 0

    for i in range(len(text)):
        if text[i] in '0123456789':
            count += 1
    return count


def is909(filename):
    """ checks if the number is in the area code 909
    """

    file = open(filename,"r")
    text = file.read()
    
    text = clean_digits(text)
    if len(text) == 10 and text[0:3] == '909':
        return True

    else:
         return False


def clean_digits(s):
    """ returns only the digits in the input string s
    """
    new_string = ''
    for i in range(len(s)):
        if s[i] in '1234567890':
            new_string += s[i]
    return new_string


def count_909():
    """ returns the number of phone numbers that are in 909
    """

    L = os.listdir("phone_files") 
    os.chdir( "phone_files" )
    
    count = 0

    for foldername in L[1:]:
        files = os.listdir(foldername)  
        os.chdir(foldername)

        for filename in files:
            if is909(filename):
                count += 1

        os.chdir("..")

    os.chdir("..") 
    return count

def count_garcia():
    """ returns the number of phone numbers that contain garcia
    """

    L = os.listdir("phone_files") 
    os.chdir( "phone_files" )
    
    count = 0

    for foldername in L[1:]:
        files = os.listdir(foldername)  
        os.chdir(foldername)

        for filename in files:
            if isGarcia(filename):
                count += 1

        os.chdir("..")

    os.chdir("..") 
    return count


def isGarcia(filename):
    """ checks if the number is in the area code 909
    """

    file = open(filename,"r")
    text = file.read()
    text = text.lower()
    
    if 'garcia' in text:
        return True



def main():
    print("There are ", count_files(), " .txt files in the whole set.")
    print("There are ", count_ten(), " phone numbers that are exactly 10 digits.")
    print("There are ", count_909(), " phone numbers that are in 909.")
    print("There are ", count_garcia(), " people with the name 'GARCIA' in the whole set.")

"""
ANSWERS:

1. There are 9896 .txt files in the whole set.
2. There are 3988 phone numbers that are exactly 10 digits.
3. There are 9 phone numbers that are in 909.
4. There are 237 people with the name "GARCIA" in the whole set.


"""