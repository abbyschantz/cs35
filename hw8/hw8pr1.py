#
# coding: utf-8
#
# hw8pr1.py - the k-means algorithm -- with pixels...
#
# Names: Abby Schantz, Eliana Keinan, Liz Harder

# import everything we need...
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils
import cv2
import math
import numpy as np
import os

# choose an image...
IMAGE_NAME = "./jp.png"  # Jurassic Park
IMAGE_NAME = "./batman.png"
IMAGE_NAME = "./hmc.png"
IMAGE_NAME = "./thematrix.png"
IMAGE_NAME = "./fox.jpg"
IMAGE_NAME = "./alien.jpg"
#IMAGE_NAME = "./starbucks.png"
image = cv2.imread(IMAGE_NAME, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
new_image = image.copy()
# reshape the image to be a list of pixels
image_pixels = image.reshape((image.shape[0] * image.shape[1], 3))
new_image_pixels = image_pixels

# choose k (the number of means) in  NUM_MEANS
# and cluster the pixel intensities
NUM_MEANS = 10
clusters = KMeans(n_clusters = NUM_MEANS)
clusters.fit(image_pixels)

# After the call to fit, the key information is contained
# in  clusters.cluster_centers_ :
k_values = []
count = 0
for center in clusters.cluster_centers_:
	print("Center #", count, " == ", center)
	# note that the center's values are floats, not ints!
	center_integers = [int(p) for p in center]
	print("   and as ints:", center_integers)
	k_values.append(center_integers)
	count += 1

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clusters)
bar = utils.plot_colors(hist, clusters.cluster_centers_)


# in the first figure window, show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# in the second figure window, show the pixel histograms 
#   this starter code has a single value of k for each
#   your task is to vary k and show the resulting histograms
# this also illustrates one way to display multiple images
# in a 2d layout (fig == figure, ax == axes)
#
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
title = str(NUM_MEANS)+" means"
ax[0,0].imshow(bar);    ax[0,0].set_title(title)
ax[0,1].imshow(bar);    ax[0,1].set_title(title)
ax[1,0].imshow(bar);    ax[1,0].set_title(title)
ax[1,1].imshow(bar);    ax[1,1].set_title(title)
for row in range(2):
	for col in range(2):
		ax[row,col].axis('off')
#plt.show(fig)


num_rows, num_cols, num_chans = new_image.shape
for row in range(num_rows):
	for col in range(num_cols):
		smallest_diff = float('inf')
		smallest_diff_i = 0
		r, g, b = image[row, col]
		curr_pyth = math.sqrt((r**2)+(g**2)+(b**2))
		for i in range(len(k_values)):
			i_pyth = math.sqrt((k_values[i][0]**2) + (k_values[i][1]**2) + (k_values[i][2]**2))
			curr_diff = abs(i_pyth - curr_pyth)
			if curr_diff <= smallest_diff:
				smallest_diff = curr_diff
				smallest_diff_i = i
		new_image[row, col] = k_values[smallest_diff_i]
plt.figure()
plt.axis("off")
plt.imshow(new_image)
plt.show(fig)

	
#
# comments and reflections on hw8pr1, k-means and pixels
"""
 + Which of the paths did you take:  
	+ posterizing 
		COMPLETED (see above)
	+ algorithm-implementation
		COMPLETED (see below)
 + How did it go?  Which file(s) should we look at?
 	It went well. We simply looped through the pixels and reassigned them to 
 	that in the list of choices of colors with the smallest difference
 	between them. 
 	FILES:
 		- alien_changing_k
 		- starbucks_changing_k
 + Which function(s) should we try...
 	The function runs automatically when you run the file in python. 
 	Just change whichever image it is you want to use as was done with 
 	the starter code. 
"""
#
#

# Extra Credit: Part B, Choice 2: the algs paths
# 
""" Notes from HW instructions: 
1. first, initialize k centers at random
2. next, determine which center each data point belongs to (is closest to)
3. then, re-compute the locations of the centers as the means (averages) of those data points belonging to it
4. continue until convergence
"""

def initialize_center(data, k):
	"""This function randomly chooses k number of points from this list to be our initial centers"""
	centers = random.sample(data,k)
	return centers

def cluster_points(data, centers):
	""" this function input the data and the centers randomly generated from 
	initialize_center and outputs a dictionary of keys (made up of the centers) 
	and the following list of points that are closest to that center """
	clusters = {}
	for i in data:
		closestCenter = -1
		dist_closestCenter = float('inf')
		for x in centers:
			dist = math.sqrt( (i[0] - x[0])**2 + (i[1] - x[1])**2 )
			if dist < dist_closestCenter:
				closestCenter = x
				dist_closestCenter = dist
		if closestCenter in clusters:
			clusters[closestCenter].append(i)
		else:
			clusters[closestCenter] = [i]
	return clusters

def recenter(clusters):
	""" recenter takes in the clusters from the cluster_points 
	function and returns new centers that are calculated by taking 
	the means of the lists from the prior dictionary and forming a 
	new center point"""
	newcenters = []
	keys = sorted(clusters.keys())
	for k in keys:
		sumx, sumy = 0, 0
		for p in clusters[k]:
			sumx += p[0]
			sumy += p[1]
		avgx = sumx/len(clusters[k])
		avgy = sumy/len(clusters[k])
		newcenters += [(avgx, avgy)]
	return newcenters

def convergence(oldcenters, centers):
	"""this function tests to see if the centers have changed 
	by seeing if the centers are equal to the old centers"""
	return set(oldcenters) == set(centers)

def kmeans(data,k):
	"""This is the main function that runs the kmeans algoritm. 
	It continues to interate through the helper functions until 
	convergence has been reached. This fucntion outputs a dictionary 
	made up of a the new centers and the points that make up that center's cluster"""
	oldcenters = []
	centers = initialize_center(data,k)
	clusters = {}
	while (convergence(centers, oldcenters) == False):
		oldcenters = centers
		clusters = cluster_points(data,oldcenters)
		#print(clusters)
		centers = recenter(clusters)
	return (clusters)

""" Reflection: 
For extra credit we chose to implement the kmeans function 
through an algorithm. It was helpful to create the helper 
functions for each step of the math formula (given to us in 
the homework instructions) as it really clarified what kmeans did. 
If I were to choose a method after doing both I would use the first 
since there is less room for error, however it was helpful to do the 
two paths for the same homework problem. 
"""