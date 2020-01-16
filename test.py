"""
Simple video capture using OpenCV
"""

import numpy as np
import cv2



# Create the video capture
cap = cv2.VideoCapture(0)

# Set up the window
cv2.namedWindow("Camera")


# Run the program
while True:
    # Capture an image
    ret, frame = cap.read()

    # If an image was captured, show it.
    if ret == True:
        # Write the image to a file
        cv2.imwrite('test.png', frame)
        # Show the image
        cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()