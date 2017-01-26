#
# hw0pr1.py
# Eliana Keinan, Liz Harder, Abby Schantz

# An example function

def plus1( N ):
    """ returns a number one larger than its input """
    return N+1


# An example loop (just with a printed countdown)

import time

def countdown( N ):
    """ counts downward from N to 0 printing only """
    for i in range(N,-1,-1):
        print("i ==", i)
        time.sleep(0.01)

    return    # no return value here!


# ++ Challenges:  create and test as many of these five functions as you can.
#
# The final three will be especially helpful!
#
# times42( s ):      which should print the string s 42 times (on separate lines)
# alien( N ):          should return the string "aliii...iiien" with exactly N "i"s
# count_digits( s ):    returns the number of digits in the input string s
# clean_digits( s ):    returns only the digits in the input string s
# clean_word( s ):    returns an all-lowercase, all-letter version of the input string s

def times42(s):
    """ prints the string s 42 times (on separate lines)
    """
    for i in range(42):
        print(s)

def alien(N):
    """ returns the string "aliiiii...iiien" with exactly N "i"s
    """
    print('al' + 'i'*N + 'en')

def count_digits(s):
    """ returns the number of digits in the input 
    """
    count = 0
    for i in range(len(s)):
        if s[i] in '1234567890':
            count += 1
    return count

def clean_digits(s):
    """ returns only the digits in the input string s
    """
    new_string = ''
    for i in range(len(s)):
        if s[i] in '1234567890':
            new_string += s[i]
    return new_string
            

def clean_word(s):
    """ returns all-lowercase, all-letter version of the input string s
    """
    new_string = ''
    for i in range(len(s)):
        if ord(s[i]) > ord('a') and ord(s[i]) < ord('z'):
            new_string += s[i]

    return new_string
