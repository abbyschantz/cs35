# ## Problem 3:  green-screening!
# 
# Names: Liz Harder, Eliana Keinan, Abby Schantz
#
# This question asks you to write one function that takes in two images:
#  + orig_image  (the green-screened image)
#  + new_bg_image (the new background image)
#  
# It also takes in a 2-tuple (corner = (0,0)) to indicate where to place the upper-left
#   corner of orig_image relative to new_bg_image
#
# The challenge is to overlay the images -- but only the non-green pixels of
#   orig_image...
#

#
# Again, you'll want to borrow from hw7pr1 for
#  + opening the files
#  + reading the pixels
#  + create some helper functions
#    + defining whether a pixel is green is the key helper function to write!
#  + then, creating an output image (start with a copy of new_bg_image!)
#
# Happy green-screening, everyone! Include at least TWO examples of a background!
#

import cv2

import numpy as np
from matplotlib import pyplot as plt

# potential foreground images
raw_eliana = cv2.imread('eliana.jpg',cv2.IMREAD_COLOR)
eliana_fg = cv2.cvtColor(raw_eliana, cv2.COLOR_BGR2RGB)

raw_liz_el = cv2.imread('liz_el.jpg',cv2.IMREAD_COLOR)
liz_el_fg = cv2.cvtColor(raw_liz_el, cv2.COLOR_BGR2RGB)

raw_liz = cv2.imread('liz.jpg',cv2.IMREAD_COLOR)
liz_fg = cv2.cvtColor(raw_liz, cv2.COLOR_BGR2RGB)

raw_bunny = cv2.imread('bunny.jpg',cv2.IMREAD_COLOR)
bunny_fg = cv2.cvtColor(raw_bunny, cv2.COLOR_BGR2RGB)

#potential background images
raw_abby = cv2.imread('abby.jpg',cv2.IMREAD_COLOR)
abby_bg = cv2.cvtColor(raw_abby, cv2.COLOR_BGR2RGB)

raw_alien = cv2.imread('alien.jpg',cv2.IMREAD_COLOR)
alien_bg = cv2.cvtColor(raw_alien, cv2.COLOR_BGR2RGB)


def remove_green(image):
    """
    replaces anything characterized as green with white
    input: an r,g,b image
    output: a transformed r,g,b image
    """
    new_image = image.copy()
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            if g>=r and r<80 and g>=b and g>=120:
                new_image[row,col] = [0, 255 , 0]
            else: 
                new_image[row,col] = [r,g,b]
    return new_image

# this is the signature for green_screening
def green_screen( orig_image, new_bg_image, corner=(0,0) ):

    bg_image = new_bg_image.copy()
    num_rows1, num_cols1, num_chans1 = bg_image.shape
    
    fg_image = remove_green(orig_image.copy())
    small_fg_image = cv2.resize(fg_image, dsize=(num_cols1, num_rows1))
    num_rows2, num_cols2, num_chans2 = small_fg_image.shape  
    
#     plt.imshow(small_image)
#     plt.show()
    
    num_rows = min(num_rows1, num_rows2)
    num_cols = min(num_cols1, num_cols2)
    
    for row in range(num_rows):
        for col in range(num_cols):
            fg_row = row-corner[0]
            fg_col = col-corner[1]
            if 0<=fg_row<num_rows and 0 <=fg_col<num_cols:
                r1, g1, b1 = small_fg_image[row-corner[0],col-corner[1]]
                r2, g2, b2 = bg_image[row][col] # + corner[0],col+corner[1]]
                if r1==0 and g1==255 and b1==0:   #pixels in the foreground
                    bg_image[row,col] = [r2,g2,b2]
                else: bg_image[row,col] = [r1,g1,b1]
    
    final_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR) # convert back!    
    cv2.imwrite( "green_screen1.png", final_image )

    return bg_image


#Example 1 (this program is very slow - it is best to do one example at a time)
""" Green definition: if g>=r and r<50 and g>=b and g>=120: """
bunny_abby = green_screen(bunny_fg, abby_bg, [300,-150])
plt.imshow(bunny_abby)
plt.show()

#Example 2
"""Green definition: if g>=r and r>b and g>=b and b<150 and g>=120:"""
#liz_alien = green_screen(liz_fg, alien_bg,[10,10])
#plt.imshow(liz_alien)
#plt.show()



"""
REFLECTION: 

Comment and reflection    You should include a short triple-quoted comment at the bottom of hw7pr3.py that 
reflects on how you implemented your green_screen function. In addition, you should include at least two 
examples -- using your own images -- of the green-screening in action! Creative backgrounds are especially welcome!


The first step of this process was defining the green in the 
green-screen image. I chose to define green as a pixel where 
the RGB ratios were as follows: (if g>=r and r<50 and g>=b 
and g>=150). This ensured that green was the largest number 
of the three but I also needed to include that r should be 
less that 50 becuase otherwise green areas were not being removed. 
There are still shades of green that are not covered by this ratio 
but I felt as though the amount of green leftover was acceptatble. 
In order to make this even better, a specific definition of green 
was used for each image. This definition is written in quotes under each example. 
Instead of actually removing the green pixel, I converted all the 
"green" pixels into pixels with an RGB of (0,255,0) which is quite a 
distinctive color. 

In the green_screen function I took the background image and overlayed 
the foreground image. For every foreground pixel that matched the RGB (0,255,0) 
the background image pixel was used instead of the foreground pixel. 

"""