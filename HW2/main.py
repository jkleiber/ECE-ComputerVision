"""
Face merging using OpenCV
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Main function
if __name__ == "__main__":

    # Read in the pictures
    justin_img = cv2.imread("justin.jpg") # Me
    andrea_img = cv2.imread("andrea.jpg") # My girlfriend, Andrea

    # Resize the images to be the same size
    img_size = (640, 1132)
    justin_img = cv2.resize(justin_img, img_size)
    andrea_img = cv2.resize(andrea_img, img_size)

    # Make the images grayscale
    justin_img = cv2.cvtColor(justin_img, cv2.COLOR_BGR2GRAY)
    andrea_img = cv2.cvtColor(andrea_img, cv2.COLOR_BGR2GRAY)

    # Show the raw images
    cv2.imshow('Justin', justin_img)
    cv2.imshow('Andrea', andrea_img)
    cv2.waitKey(1)

    gauss_kernel = (11, 11)#(55, 55)
    justin_low_pass = cv2.GaussianBlur(justin_img, gauss_kernel, cv2.BORDER_DEFAULT)
    justin_high_pass = justin_img - justin_low_pass
    andrea_low_pass = cv2.GaussianBlur(andrea_img, gauss_kernel, cv2.BORDER_DEFAULT)
    andrea_high_pass = andrea_img - andrea_low_pass

    combined_img = (justin_low_pass + andrea_high_pass)
    combined_img = (combined_img - combined_img.min())/(combined_img.max() - combined_img.min()) * 255
    combined_img = np.uint8(combined_img)
    #plt.imshow(combined_img, cmap=plt.get_cmap('gray'))
    #plt.show()

    cv2.imshow('Far away', cv2.resize(combined_img, (320, 580)))
    cv2.imshow('Up close', combined_img)

    # Delay 1ms to show the image, and press Q to exit while screen has focus
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()