import cv2
import numpy as np
import matplotlib.pyplot as pyplot

image = cv2.imread("./test/test_images/straight_lines1.jpg")

lane_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts from BGR to RGB


#Since all lane lines are either yellow or white we are going to crop all yellow and white colors from image/video
def filter_by_color(img):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    #Hue Lightness Saturation
    white_lower = np.array([0,190,0])
    white_upper = np.array([255,255,255])
    yellow_lower = np.array([10,0,90])
    yellow_upper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellow_lower, yellow_upper)
    whitemask = cv2.inRange(hls, white_lower, white_upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
    
    
def region_of_interest(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    #4 rows = 4 points 2 
    #columns =(x,y)
    #shape is array of four points
    #Triangle Shape
    #Base is the entire x-axis while the vertex is at 50% width and 60% height
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.5*x), int(0.6*y)]])
    mask = np.zeros_like(img) #Creates an array of same size as image but all zeros
    #fillPoly takes an array same size as the image, polygon that you want to draw and a color to make same size polygon image
    cv2.fillPoly(mask,pts=[shape],color=255)
    #bitwise_anding to get the desired region
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

mask_by_color = filter_by_color(lane_image)
roi = region_of_interest(mask_by_color)


cv2.imshow("masked_image",roi)
cv2.waitKey(0)