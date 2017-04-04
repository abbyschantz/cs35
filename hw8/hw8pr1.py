#
# coding: utf-8
#
# hw8pr1.py - the k-means algorithm -- with pixels...
#

# import everything we need...
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils
import cv2
import math

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
	+ posterizing or 
	+ algorithm-implementation

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