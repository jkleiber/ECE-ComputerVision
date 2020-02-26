"""
Harris Corner Detector
"""

import cv2
import numpy as np

# K value used in Harris Corner Detector
K = 0.05

def window_function(img, size):
    return cv2.filter2D(img, -1, np.ones((size,size), dtype=np.float32))

def harris_corner_detector(img, kernel_size):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # Compute image derivatives
    Ix = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    Iy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=kernel_size)

    # Compute the M matrix values
    a = window_function(Ix * Ix, 3)
    b = window_function(Ix * Iy, 3)
    d = window_function(Iy * Iy, 3)

    # Find the determinant and the trace of M
    det = (a*d) - (b**2)
    trace = a + d

    # Make a mask for the corners
    R = det - K*trace**2
    mask = (R > 0.01*R.max())

    # Make the mask suppress anything that isn't the local maximum
    mask = ( mask
          * (cv2.filter2D(R, -1, np.array([-1, 1, 0])) > 0)
          * (cv2.filter2D(R, -1, np.array([0, 1, -1])) > 0)
          * (cv2.filter2D(R, -1, np.array([0, 1, -1]).transpose()) > 0)
          * (cv2.filter2D(R, -1, np.array([-1, 1, 0]).transpose()) > 0)
            )
    mask = cv2.filter2D(np.float32(mask), -1, np.ones((3,3))) > 0.5

    # Modify the picture using the mask to show red dots on the corners
    img[mask] = [0,0,255]

    return img

def shi_tomasi_corner_detector(img, kernel_size):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # Compute image derivatives
    Ix = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    Iy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=kernel_size)

    # Compute the M matrix values
    # M = [a b; c d] with b = c
    a = window_function(Ix * Ix, 3)
    b = window_function(Ix * Iy, 3)
    d = window_function(Iy * Iy, 3)

    # Find the eigenvalues using the quadratic formula
    # x^2 - (a + d)x + (ad - b^2) = 0, where x is an eigenvalue
    B = a + d
    C = a*d - b**2

    # Minimum value always uses subtraction
    R = 0.5 * (B - np.sqrt(B**2 - 4*C))

    # Make a mask for the corners
    mask = (R > 0.01 * R.max())

    # Make the mask suppress anything that isn't the local maximum
    mask = ( mask
          * (cv2.filter2D(R, -1, np.array([-1, 1, 0])) > 0)
          * (cv2.filter2D(R, -1, np.array([0, 1, -1])) > 0)
          * (cv2.filter2D(R, -1, np.array([0, 1, -1]).transpose()) > 0)
          * (cv2.filter2D(R, -1, np.array([-1, 1, 0]).transpose()) > 0)
            )
    mask = cv2.filter2D(np.float32(mask), -1, np.ones((3,3))) > 0.5

    # Modify the picture using the mask to show red dots on the corners
    img[mask] = [0,255,0]

    return img

if __name__ == "__main__":
    # Load the test image
    test_img = cv2.imread("test_image.jpg")
    shi_img = np.copy(test_img)

    # Compute the results
    harris_result_img = harris_corner_detector(test_img, kernel_size=5)
    shi_tomasi_result_img = shi_tomasi_corner_detector(shi_img, kernel_size=5)

    # Show the harris result and save it to a file
    cv2.imwrite("harris_result_img.jpg", harris_result_img)
    cv2.imshow('Harris Corner Detector', harris_result_img)

    # Show the Shi-Tomasi result and save it to a file
    cv2.imwrite("shi_tomasi_result_img.jpg", shi_tomasi_result_img)
    cv2.imshow('Shi-Tomasi Corner Detector', shi_tomasi_result_img)

    # Wait for the user to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

