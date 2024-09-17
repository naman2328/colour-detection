import cv2
import numpy as np

def simulate_red_color_detection(cap):
    while True:
        ret, im = cap.read()  # Read frames from the webcam

        if not ret:
            print("Failed to grab frame")
            break

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # Convert frames from BGR to HSV

        # Define refined color ranges in HSV format for red
        low_red1 = np.array([0, 50, 50])
        high_red1 = np.array([10, 255, 255])
        low_red2 = np.array([170, 50, 50])
        high_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, low_red1, high_red1)
        mask_red2 = cv2.inRange(hsv, low_red2, high_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
       #hsv for black
        low_black = np.array([0, 0, 00])
        high_black = np.array([185, 255, 50])
        mask_black = cv2.inRange(hsv,low_black,high_black)
        #hsv for blue
        low_blue = np.array([100, 50, 50])
        high_blue = np.array([140, 255, 250])
        mask_blue = cv2.inRange(hsv,low_blue,high_blue)
        #hsv for green
        low_green = np.array([35,50,50])
        high_green = np.array([85,255,255])
        mask_green =cv2.inRange(hsv,low_green,high_green)
        
        # Visualize red mask
        cv2.imshow('Red Mask', mask_red)
        cv2.imshow('black Mask', mask_black)
        cv2.imshow('blue Mask', mask_blue)
        cv2.imshow('green mask',mask_green)
        # Find contours for red
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_green,_ = cv2.findContours(mask_green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours and identify red color
        for contour in contours_red:
            if cv2.contourArea(contour) > 10000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(im, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        for contour in contours_black:
            if cv2.contourArea(contour) > 10000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(im, 'black', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        for contour in contours_blue:
            if cv2.contourArea(contour) > 10000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(im, 'blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        for contour in contour_green:
            if cv2.contourArea(contour) > 10000:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(im, 'green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Frame', im)  # Show the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
         break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
    else:
        simulate_red_color_detection(cap)
