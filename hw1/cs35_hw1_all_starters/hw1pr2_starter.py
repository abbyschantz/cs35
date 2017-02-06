#
# Names: Eliana Keinan, Abby Schantz, Liz Harder
# 

import csv

#
# readcsv is a starting point - it returns the rows from a standard csv file...
#
def readcsv( csv_file_name ):
    """ readcsv takes as
         + input:  csv_file_name, the name of a csv file
        and returns
         + output: a list of lists, each inner list is one row of the csv
           all data items are strings; empty cells are empty strings
    """
    try:
        csvfile = open( csv_file_name, newline='' )  # open for reading
        csvrows = csv.reader( csvfile )              # creates a csvrows object

        all_rows = []                               # we need to read the csv file
        for row in csvrows:                         # into our own Python data structure
            all_rows.append( row )                  # adds only the word to our list

        del csvrows                                  # acknowledge csvrows is gone!
        csvfile.close()                              # and close the file
        return all_rows                              # return the list of lists

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []



#
# write_to_csv shows how to write that format from a list of rows...
#  + try   write_to_csv( [['a', 1 ], ['b', 2]], "smallfile.csv" )
#
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


#
# csv_to_html_table_starter
#
#   Shows off how to create an html-formatted string
#   Some newlines are added for human-readability...
#
def csv_to_html_table_starter( csvfilename ):
    """ csv_to_html_table_starter
           + an example of a function that returns an html-formatted string
        Run with 
           + result = csv_to_html_table_starter( "example_chars.csv" )
        Then run 
           + print(result)
        to see the string in a form easy to copy-and-paste...
    """
    # probably should use the readcsv function, above!
    html_string = '<table>\n'    # start with the table tag
    html_string += '<tr>\n'

    html_string += "place structured data here!\n" # from list_of_rows !

    html_string += '</tr>\n'
    html_string += '</table>\n'
    return html_string


def first_letter_counter_weighted(csv_file_name):
    """ returns the weighted count of words that start with each letter
    """

    L = readcsv(csv_file_name)
    d = {}

    for word in L:
        word[0] = word[0].lower()
        if ord(word[0][0]) >= ord('a') and ord(word[0][0]) <= ord('z'):
            if word[0][0] in d:
                d[word[0][0]] += 1*(float(word[1]))
            else:
                d[word[0][0]] = 1*(float(word[1]))

    return d


def last_letter_counter_weighted(csv_file_name):
    """ returns the weighted count of words that end with each letter
    """

    L = readcsv(csv_file_name)
    d = {}

    for word in L:
        word[0] = word[0].lower()
        if ord(word[0][-1]) >= ord('a') and ord(word[0][-1]) <= ord('z'):
            if word[0][-1] in d:
                d[word[0][-1]] += 1*(float(word[1]))
            else:
                d[word[0][-1]] = 1*(float(word[1]))

    return d


def first_last_equal_counter_weighted(csv_file_name):
    """ returns the weighted count of words (longer than 1 letter) that start and end with the same letter
    """

    L = readcsv(csv_file_name)
    d = {}

    for word in L:
        word[0] = word[0].lower()
        if ord(word[0][0]) >= ord('a') and ord(word[0][0]) <= ord('z'):
            if word[0][0] == word[0][-1] and len(word[0]) > 1:
                if word[0][0] in d:
                    d[word[0][0]] += 1*(float(word[1]))
                else:
                    d[word[0][0]] = 1*(float(word[1]))

    return d


def output_first_letter(csv_file_name):
    """ outputs a csv file for the first letter counter
    """
    D = first_letter_counter_weighted(csv_file_name)
    E = last_letter_counter_weighted(csv_file_name)
    F = first_last_equal_counter_weighted(csv_file_name)
    
    list_of_rows = []
    for key, value in D.items():
        list_of_rows.append((key,value))
    
    list_of_rows.sort()
    
    write_to_csv( list_of_rows, "firstletter.csv")




def output_first_letter(csv_file_name):
    """ outputs a csv file for the first letter counter
    """
    D = first_letter_counter_weighted(csv_file_name)
    
    list_of_rows = []
    for key, value in D.items():
        list_of_rows.append((key,value))
    
    list_of_rows.sort()
    
    write_to_csv( list_of_rows, "firstletter.csv")

def output_last_letter(csv_file_name):
    """ outputs a csv file for the last letter counter
    """
    D = last_letter_counter_weighted(csv_file_name)
    
    list_of_rows = []
    for key, value in D.items():
        list_of_rows.append((key,value))
    
    list_of_rows.sort()
    
    return write_to_csv( list_of_rows, "lastletter.csv")


def output_first_last_letter(csv_file_name):
    """ outputs a csv file for the first last equal counter
    """
    D = first_last_equal_counter_weighted(csv_file_name)
    
    list_of_rows = []
    for key, value in D.items():
        list_of_rows.append((key,value))
    
    list_of_rows.sort()
    
    return write_to_csv( list_of_rows, "firstlastletter.csv")


### FIX THIS STILL ###
def csv_to_html_table( csvfilename ):
    """ 
    """
   
   result = csv_to_html_table_starter("FILE")
   print(result)
   
   
    # probably should use the readcsv function, above!
    html_string = '<table>\n'    # start with the table tag
    html_string += '<tr>\n'

    html_string += "place structured data here!\n" # from list_of_rows !

    html_string += '</tr>\n'
    html_string += '</table>\n'
    return html_string




def main():
    """ creates three files that provide different analysis on letter frequencies
    """
    
    csv_file_name = "wds.csv"
    
    output_first_letter(csv_file_name)
    output_last_letter(csv_file_name)
    output_first_last_letter(csv_file_name)