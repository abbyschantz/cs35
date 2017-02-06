#
# starter file for hw1pr2, cs35 spring 2017...
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


