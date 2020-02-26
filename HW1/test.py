"""
Simple video capture using OpenCV
"""

import numpy as np
import cv2

# Main function
if __name__ == "__main__":
    # Create the video capture
    cap = cv2.VideoCapture(0)

    # Set up the video writer to capture 640x480 video at 30 FPS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('HW1.avi',fourcc, 30, (640,480))

    # Run the program
    while True:
        # Capture an image
        ret, frame = cap.read()

        # If an image was captured, show it.
        if ret == True:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert to black and white based on a threshold
            thresh, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            # Write the image to a file after converting to BGR
            bw_video = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
            video_out.write(bw_video)

            # Show the images
            cv2.imshow('Camera', frame)
            cv2.imshow('Black and White', bw_image)

        # Delay 1ms to show the image, and press Q to exit while screen has focus
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()