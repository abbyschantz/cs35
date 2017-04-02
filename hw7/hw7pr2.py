# Names: Eliana Keinan, Abby Schantz, Liz Harder
#
# ## Problem 2:  steganography
# 
# This question asks you to write two functions, likely with some helper functions, that will enable you
# to embed arbitrary text (string) messages into an image (if there is enough room!)

# For extra credit, the challenge is to be
# able to extract/embed an image into another image...

#
# You'll want to borrow from hw7pr1 for
#  + opening the file
#  + reading the pixels
#  + create some helper functions!
#  + also, check out the slides :-) 
#
# Happy steganographizing, everyone!
#

import numpy as np
from matplotlib import pyplot as plt
import cv2


def opencv(image_name):
    """ function to open and read image file, adapted from hw7pr1 """
    # Reading and color-converting an image to RGB
    raw_image = cv2.imread(image_name,cv2.IMREAD_COLOR) 

    # convert an OpenCV image (BGR) to an "ordinary" image (RGB) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
 
    plt.imshow(image)    # plt.show()
    # plt.xticks([]), plt.yticks([])  # to hide axis labels
    input("Hit enter to continue...")
    # In[ ]:

    print("image.shape is", image.shape)
    input("Hit enter to continue...")

    return image   # for use further downstream...

# Part A: here is a signature for the decoding
# remember - you will want helper functions!

def extract_last_bits(image):
    """ extracts the last bit of each element of the inputted
        image and returns a string of the image's last bits
    """

    last_bits = ''
    for r in range(image.shape[0]-1): #loop through pixels
        for c in range(image.shape[1]-1): # loop through elements
            red = image[r,c,0]  # red
            red = bin(red)[-1]

            green = image[r,c,1]  # gre
            green = bin(green)[-1]

            blue = image[r,c,2]  # blu
            blue = bin(blue)[-1]

            last_bits += str(red)
            last_bits += str(green)
            last_bits += str(blue)

    return last_bits


def desteg_string( image_name ):
    """ this takes in a steganoroahized image named image_name and 
        goes through each pixel and extracts the lowest-order bit 
        from its channels.  It then returns and prints the hidden
        message
    """
    image = opencv(image_name)
    last_bits = extract_last_bits(image)
    message = ''
    length = len(last_bits)

    for i in range(length - 1):
        if len(last_bits) > 7:
            character = last_bits[0:8]
            if character == '00000000':
                break
            character = int(character,2)
            character = chr(character)

            message += character
            last_bits = last_bits[8:]
    
    print(message)
    
    return message



# Part B: here is a signature for the encoding/embedding
# remember - you will want helper functions!


def message_to_binary(message):
    """ takes in a message as a string and returns the message
        as a string of binary numbers
    """
    
    b_message = ''
    for i in range(len(message)):
        character = message[i]
        character = ord(character)
        character = bin(character)[2:]
        if len(character) < 8:
            add = 8 - len(character)
            character = "0"*add + character
        b_message += character
    b_message += "00000000"
    return b_message
    



def steganographize( image_name, message ):
    """ takes in an image and a message (a string) and returns
        a copy of the image with the least significant bt of some/all 
        of its pixels changed to hold the message, one bit at a time
    """
    
    image = opencv(image_name)
    new_image = image.copy()

    b_message = message_to_binary(message)

    for r in range(new_image.shape[0]-1): #loop through pixels
        for c in range(new_image.shape[1]-1): # loop through elements
            if len(b_message) > 2:
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

                green = new_image[r,c,1]  # gre
                green = bin(green)[2:]
                green = green[:-1] + b_message[1]
                new_image[r,c,1] = int(green,2)

                blue = new_image[r,c,2]  # blu
                blue = bin(blue)[2:]
                blue = blue[:-1] + b_message[2]
                new_image[r,c,2] = int(blue,2)
            
            if len(b_message) == 2:                
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

                green = new_image[r,c,1]  # gre
                green = bin(green)[2:]
                green = green[:-1] + b_message[1]
                new_image[r,c,1] = int(green,2)
            
            if len(b_message) == 1:
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

            b_message = b_message[3:]
    
    raw_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR) # convert back!
    cv2.imwrite("image_out.png", raw_image )

    return new_image



def steganographize_image( orig_image_name, image_message_name ):
    """ takes in two images and returns the first image (orig_image)
        with the second image (image_message) hidden in it
    """
    orig_image = opencv(orig_image_name)
    new_image = orig_image.copy()


    b_message = image_to_binary(image_message_name)
    for r in range(new_image.shape[0]-1): #loop through pixels
        for c in range(new_image.shape[1]-1): # loop through elements
            if len(b_message) > 2:
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

                green = new_image[r,c,1]  # gre
                green = bin(green)[2:]
                green = green[:-1] + b_message[1]
                new_image[r,c,1] = int(green,2)

                blue = new_image[r,c,2]  # blu
                blue = bin(blue)[2:]
                blue = blue[:-1] + b_message[2]
                new_image[r,c,2] = int(blue,2)
            
            if len(b_message) == 2:                
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

                green = new_image[r,c,1]  # gre
                green = bin(green)[2:]
                green = green[:-1] + b_message[1]
                new_image[r,c,1] = int(green,2)
            
            if len(b_message) == 1:
                red = new_image[r,c,0]  # red
                red = bin(red)[2:]
                red = red[:-1] + b_message[0]
                new_image[r,c,0] = int(red,2)

            b_message = b_message[3:]

    raw_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR) # convert back!
    cv2.imwrite("image_with_message.png", raw_image )    

    return new_image


def image_to_binary(image_name):
    """ takes in an image and returns the RGB elements
        of the image in a string of 8-bit binary numbers
    """
    image = opencv(image_name)

    b_message = ''
    for r in range(image.shape[0]-1): #loop through pixels
        for c in range(image.shape[1]-1): # loop through elements  
            red = image[r,c,0]
            green = image[r,c,1]
            blue = image[r,c,2]

            red = bin(red)[2:]
            green = bin(green)[2:]
            blue = bin(blue)[2:]

            if len(red) < 8:
                add = 8 - len(red)
                red = "0"*add + red
            
            if len(green) < 8:
                add = 8 - len(green)
                green = "0"*add + green
            
            if len(blue) < 8:
                add = 8 - len(blue)
                blue = "0"*add + blue

            b_message += red
            b_message += green
            b_message += blue

        b_message += "00000000"
    return b_message



def desteg_image( image_with_message ):
    """ takes in an image with another image hidden in it and
        returns the hidden image
    """
    image = opencv(image_with_message)

    last_bits = extract_last_bits(image)
    
    length = len(last_bits)

    for i in range(length - 1):
        if len(last_bits) > 7:
            element = last_bits[0:8]
            if element == '00000000':
                break
            element = int(character,2)
            character = chr(character)

            message += character
            last_bits = last_bits[8:]
    
    print(message)
    
    return message
    



"""
(1) image with a hidden message:
        
    Run the following two commands to get our hidden message:
        desteg_string( "alien_out.png" )
        desteg_string( "starbucks_out.png" )

(2) challenge your grutors to identify a hidden famous text

    To create this, we first ran steganographize on the mouse image
    and the mysterytext.

    Run desteg_string("mouse_out.png") to get the mystery text!
    It takes a while to run... 

    (HINT: the image we picked should help give away the novel title!)

(EC) image hidden in another image

    For the EC, we only created the first half where you are able to
    steganographize_image but didn't created the desteg_image second half.

    "image_with_message.png" in our zipped folder has an image hidden in it!

"""