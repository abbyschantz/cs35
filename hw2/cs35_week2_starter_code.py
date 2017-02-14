# Liz Harder, Eliana Keinan, and Abby Schantz
# starting examples for cs35, week2 "Web as Input"
#

import requests
import string
import json

"""
Examples you might want to run during class:

Web scraping, the basic command (Thanks, Prof. Medero!)

#
# basic use of requests:
#
url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
result = requests.get(url)
text = result.text   # provides the source as a large string...

#
# try it for another site...
#

# 
# let's demo the weather example...
# 
url = 'http://api.wunderground.com/api/49e4f67f30adb299/geoloookup/conditions/q/Us/Ca/Claremont.json' # JSON!
	   # try it + source
result = requests.get(url)
data = result.json()      # this creates a data structure from the json file!
# What type will it be?
# familiarity with dir and .keys() to access json data...

#
# let's try the Open Google Maps API -- also provides JSON-formatted data
#   See the webpage for the details and allowable use
#
# Try this one by hand - what are its parts?
# http://maps.googleapis.com/maps/api/distancematrix/json?origins=%22Claremont,%20CA%22&destinations=%22Seattle,%20WA%22&mode=%22walking%22
#
# Take a look at the result -- imagine the structure of that data... That's JSON! (Sketch?)
#
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Problem 1 starter code
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
# 
#
def google_distances_api_scrape(filename_to_save="distances.json"):
	""" a short function that shows how  
		part of Google Maps' API can be used to 
		obtain and save a json file of distances data...
	"""
	url="http://maps.googleapis.com/maps/api/distancematrix/json"

	city1="Claremont,CA"
	city2="Seattle,WA"
	my_mode="walking"

	inputs={"origins":city1,"destinations":city2,"mode":my_mode}

	result = requests.get(url,params=inputs)
	data = result.json()
	print("data is", data)

	# save this json data to file
	f = open( filename_to_save, "w" )     # opens the file for writing
	string_data = json.dumps( data, indent=2 )  # this writes it to a string
	f.write(string_data)                        # then, writes that string to a file...
	f.close()                                   # and closes the file
	print("\nfile", filename_to_save, "written.")
	# no need to return anything, since we're better off reading it from file!
	return


def google_distances_api_process(filename_to_read = "distances.json"):
	""" a function with examples of how to manipulate json data --
		here the data is from the file scraped and saved by 
		google_distances_api_starter()
	"""
	f = open( filename_to_read, "r" )
	string_data = f.read()
	data = json.loads( string_data )
	print("data (not spiffified!) is\n\n", data, "\n")

	print("Accessing its components:\n")

	row0 = data['rows'][0]
	print("row0 is", row0, "\n")

	cell0 = row0['elements'][0]
	print("cell0 is", cell0, "\n")

	distance_as_string = cell0['distance']['text']
	print("distance_as_string is", distance_as_string, "\n")

	# here, we may want to continue operating on the whole json dictionary
	# so, we return it:
	return data


#
# multicity_distance_scrape
#

origins = ["San Francisco, CA", "New York, NY", "Claremont, CA", "Atlana, GA", "Cincinnati, OH", "Westport, CT"]
destinations = ["Las Vegas, NV", "Boston, MA", "Portland, ME"]
def multicity_distance_scrape( Origins = origins, Dests = destinations, filename_to_save="multicity.json" ):
	""" 
	returns a file with the distances between each origin city and each desination. 
	Defaulted to run with
		origins = ["San Francisco, CA", "New York, NY", "Claremont, CA", "Atlana, GA", "Cincinnati, OH", "Westport, CT"]
		destinations = ["Las Vegas, NV", "Boston, MA", "Portland, ME"]
	"""
	url="http://maps.googleapis.com/maps/api/distancematrix/json"
	with open( filename_to_save, "w" ) as f:
		for city in origins:
			for dest in destinations:
				city1=city
				city2=dest
				my_mode="driving"

				inputs={"origins":city1,"destinations":city2,"mode":my_mode}

				result = requests.get(url,params=inputs)
				data = result.json()
				print("data is", data)
			
				# save this json data to file
				#f = open( filename_to_save, "w" )     # opens the file for writing
				string_data = json.dumps( data, indent=2 )  # this writes it to a string
				f.write(string_data)                        # then, writes that string to a file...
								   # and closes the file
	print("\nfile", filename_to_save, "written.")
	# no need to return anything, since we're better off reading it from file!
	return



#
# multicity_distance_process
#
#def multicity_distance_process(filename_to_read = "multicity.json"):



#
# a main function for problem 1 (the multicity distance problem)
#
def main_pr1():
	""" a top-level function for testing things! """
	# these were the cities from class:
	# Origins = ['Pittsburgh,PA','Boston,MA','Seattle,WA']  # starts
	# Dests = ['Claremont,CA','Atlanta,GA']         # goals
	#
	# Origins are rows...
	# Dests are columns...
	pass







# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Problem 2a starter code
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
#
#

def apple_api_id_scraper(artist_name, filename_to_save="appledata_id.json"):
	""" 
	"""
	### Use the search url to get an artist's itunes ID
	search_url = "https://itunes.apple.com/search"
	parameters = {"term":artist_name, "entity":"musicArtist","media":"music","limit":200}
	result = requests.get(search_url, params=parameters)
	data = result.json()

	# save to a file to examine it...
	f = open( filename_to_save, "w" )     # opens the file for writing
	string_data = json.dumps( data, indent=2 )  # this writes it to a string
	f.write(string_data)                        # then, writes that string to a file...
	f.close()                                   # and closes the file
	print("\nfile", filename_to_save, "written.")

	# we'll return a useful value: the artist id...
	#
	# Note: it's helpful to find the iTunes artistid and return it here
	# (this hasn't been done yet... try it!) 
	artistId = (data["results"][0]["artistId"])

	return artistId   # This is the Beatles...


#
# 
#
def apple_api_full_scraper(artistid, filename_to_save):
	""" 
	Takes an artistid and grabs a full set of that artist's albums.
	"The Beatles"  has an id of 136975
	"""
	lookup_url = "https://itunes.apple.com/lookup"    
	parameters = {"entity":"album","id":artistid}    
	result = requests.get(lookup_url, params=parameters)
	data = result.json()

	# save to a file to examine it...
	f = open( filename_to_save, "w" )     # opens the file for writing
	string_data = json.dumps( data, indent=2 )  # this writes it to a string
	f.write(string_data)                        # then, writes that string to a file...
	f.close()                                   # and closes the file
	print("\nfile", filename_to_save, "written.")

	# we'll leave the processing to another function...
	return



#
#
#
def apple_api_full_process(filename_to_read):
	""" example of extracting one (small) piece of information from 
		the appledata json file...
	"""
	f = open( filename_to_read, "r" )
	string_data = f.read()
	data = json.loads( string_data )
	#print("data (not spiffified!) is\n\n", data, "\n")

	# for live investigation, here's the full data structure
	return data



#
#
#
def most_productive_scrape(artist1, artist2, fname1="artist1.json", fname2="artist2.json"):
	"""
	Scrapes data for two artists from iTunes API and returns for processing
	"""

	#get artist IDs
	artist1Id = apple_api_id_scraper(artist1)
	artist2Id = apple_api_id_scraper(artist2)

	#get artist1 data
	apple_api_full_scraper(artist1Id, fname1)
	artist1Data = apple_api_full_process(fname1)

	#get artist2 data
	apple_api_full_scraper(artist2Id, fname2)
	artist2Data = apple_api_full_process(fname2)

	return



#
#
#
def most_productive_process(fname1="artist1.json", fname2="artist2.json"):
	"""
	Reads json files for two artists, generated by iTunes API.
	Companres results and returns the name of the more productive artist
	"""
	f1 = open(fname1, "r")
	f2 = open(fname2, "r")
	string_data1 = f1.read()
	string_data2 = f2.read()
	data1 = json.loads(string_data1)
	data2 = json.loads(string_data2)

	#gets the number of results for each artist 
	data1resultCount = data1['resultCount']
	data2resultCount = data2['resultCount']

	#gets the artist name from each artist id
	data1ArtistName = data1["results"][0]["artistName"]
	data2ArtistName = data2["results"][0]["artistName"]

	#prints results 
	print(data1ArtistName, "return(s)", data1resultCount, "results")
	print(data2ArtistName, "return(s)", data2resultCount, "results")
	if data1resultCount == data2resultCount:
		return "The artists return an equal number of results"
	elif data1resultCount > data2resultCount:
		return data1ArtistName
	else:
		return data2ArtistName


#
# main_pr2()  for testing problem 2's functions...
#
def main_pr2():
	""" a top-level function for testing things... """
	#TEST 1 - Katy Perry v. Steve Perry 
	most_productive_scrape( "Katy Perry", "Steve Perry" )
	most_productive_process()  # uses default filenames!
	#TEST 2 - Taylor Swift v. Calvin Harris 
	most_productive_scrape( "Taylor Swift", "Calvin Harris" )
	most_productive_process()
	#TEST 3 - Beyonce v. Kanye 
	most_productive_scrape( "Beyonce", "Kanye" )
	most_productive_process()

	return

"""
Overview of progress on this problem - test cases you ran

For example: most_productive_scrape( "Taylor Swift", "Kanye West" ); most_productive_process()
"""



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Problem 2b starter code
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import datetime

#
# earthquake examples...
#
""" unlike the previous problems, this starter code really
	just shows snippets -- you might try at the command-prompt
	or by editing commenting in/out the lines of this function...
"""
## Playground! Use this space to explore and play around with the USGS API
#


def quake_api_count_scraper(starttime, endtime, filename_to_save="quakedata_id.json"):
	""" 
	"""
	### Use the search url to get an artist's itunes ID
	url = "http://earthquake.usgs.gov/fdsnws/event/1/count"
	parameters={"format":"geojson","limit":"20000","starttime":starttime,"endtime":endtime}
	result = requests.get(url,params=parameters)
	data = result.json()
	print(data)


	# save to a file to examine it...
	f = open( filename_to_save, "w" )     # opens the file for writing
	string_data = json.dumps( data, indent=2 )  # this writes it to a string
	f.write(string_data)                        # then, writes that string to a file...
	f.close()                                   # and closes the file
	print("\nfile", filename_to_save, "written.")
	return data["count"]



def quake_api_full_process(filename_to_read):
	""" example of extracting one (small) piece of information from 
		the appledata json file...
	"""
	f = open( filename_to_read, "r" )
	string_data = f.read()
	data = json.loads( string_data )
	#print("data (not spiffified!) is\n\n", data, "\n")

	# for live investigation, here's the full data structure
	return data

def quake_api_get_count(filename_to_read="quakedata_id.json"):
	days = get_seven_dates()
	max_quakes = ["0000-00-00", 0]
	for day in days:
		end = day + datetime.timedelta(days=1)
		count = quake_api_count_scraper(day, end)

		print (day, "had", count, "earthquakes")
		if count > max_quakes[1]:
			max_quakes = [day, count]
	print("In the past seven days,",max_quakes[0], "had the most earthquakes at", max_quakes[1])
	return 


def get_seven_dates():
	days_list = []
	for i in range(7):
		i_days_prior = datetime.date.today() - datetime.timedelta(days=i)
		print (i_days_prior)
		i -= 1
		days_list.append(i_days_prior)
	return days_list



print("\nProblem 2b's starter-code results:\n")
now = datetime.date.today()
print("Now, the date is", now)   # counts from 0001-01-01
print("Now, the ordinal value of the date is", now.toordinal())
print("now.fromordinal(42) is", now.fromordinal(42))

def main_pr2b():
	""" runs the quake_api_get_count function
	"""
	quake_api_get_count()


#
# Or, you might copy individual lines to the Python command prompt...
#
"""
# reference: 
url = "http://earthquake.usgs.gov/fdsnws/event/1/query"
parameters={"format":"geojson","limit":"20000","starttime":"2017-02-05","endtime":"2017-02-06"}
result = requests.get(url,params=parameters)
data = result.json()
print("data")

# save to a file to examine it...
filename_to_save = "quake.json"
f = open( filename_to_save, "w" )     # opens the file for writing
string_data = json.dumps( data, indent=2 )  # this writes it to a string
f.write(string_data)                        # then, writes that string to a file...
f.close()                                   # and closes the file
print("\nfile", filename_to_save, "written.")

#timestamp = ... from the quake data
#print(dir(datetime))
#  The timestamps in the quake data are in one-thousandths of a day!
#dt = datetime.datetime.utcfromtimestamp(timestamp/1000) #, datetime.timezone.utc)
#print(dt)

# dates!
now = datetime.date.today()
print("Now, the date is", now)   # counts from 0001-01-01
print("Now, the ordinal value of the date is", now.toordinal())
print("now.fromordinal(42) is", now.fromordinal(42))
"""




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Problem 3 -- please take a look at problem3_example.py for an example!
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#
# This problem is very scalable -- start with a small prediction task
#   (serious or not...) that involves analyzing at least two sites...
# 
# Feel free to find your own json-based APIs -- or use raw webpages
#   and BeautifulSoup! (This is what the example does...)
#
def movie_api_scraper(movie, filename_to_save="movie.json"):
	""" INCLUDE DOC
	"""
	### Use the search url to get a movies rating
	url = "http://www.omdbapi.com/?"
	parameters={"t":movie}
	result = requests.get(url,params=parameters)
	data = result.json()
	print(data)

	# save to a file to examine it...
	f = open( filename_to_save, "w" )     # opens the file for writing
	string_data = json.dumps( data, indent=2 )  # this writes it to a string
	f.write(string_data)                        # then, writes that string to a file...
	f.close()                                   # and closes the file
	print("\nfile", filename_to_save, "written.")

def movie_api_full_process(filename_to_read="movie.json"):
	""" example of extracting one (small) piece of information from 
		the appledata json file...
	"""
	oscar_noms = ['Hidden figures', 'la la land', 'lion', 'Fences', 'Arrival','Hacksaw ridge','Manchester by the sea', 'Moonlight' ]
	f = open( filename_to_read, "r" )
	string_data = f.read()
	data = json.loads( string_data )
	#print("data (not spiffified!) is\n\n", data, "\n")

	# for live investigation, here's the full data structure
	print(data["imdbRating"])
	print(data["Actors"])
	print(data["Actors"][0])
	return data
	




