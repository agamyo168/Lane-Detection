import numpy as np
import cv2
import glob
# HOG & Extraction
from skimage.feature import hog
import time
import matplotlib.image as mpimg
#SVC
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
import warnings
from scipy.ndimage import label
import random
from moviepy.editor import VideoFileClip

#Global variables for bash
input_path = None
output_path = None
option = None  #uses -- or - notation 

n_args = len(sys.argv) #minimum is two
input_path = sys.argv[1]
output_path = sys.argv[2]
if(n_args>3):
    option = sys.argv[3]

## HOG

#### Feature Extraction 

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
#     print(features.shape)
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#     Seems like a useless step each channel for all images return same result
#     plt.hist(channel1_hist,bins=32,range=(0,256))
#     plt.hist(channel2_hist,bins=32,range=(0,256))
#     plt.hist(channel3_hist,bins=32,range=(0,256))
#     print(hist_features.shape)
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Return HOG features and visualization
# Hog takes only gray images
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features
def convert_rgb_color(img, conv='YCrCb'):
    if conv == 'RGB':
        return np.copy(img)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

#### Car Detection Function

def detect_cars(img,params, h_shift=0, visualization=False):
     #Parameters:
    svc=params['svc']
    X_scaler=params['scaler']
    orient=params['orient']
    cells_per_step = params['cells_per_step']
    pix_per_cell=params['pix_per_cell']
    cell_per_block=params['cell_per_block']
    spatial_size=params['spatial_size']
    hist_bins=params['hist_bins']
    ystart_ystop_scale = params['ystart_ystop_scale']
    
    #Boxes
    detected_boxes_list = [] # All detected cars in an image
    window_visited_list = [] # All windows in an image
    
    #to make computation efficient by reducing values between 0 to 1
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    #Searching at different window sizes to get as much detection
    for (ystart, ystop, scale) in ystart_ystop_scale:
        
        #Crop region of interest:
        #Far of the camera should have smaller window scale, closer should have bigger window scale.
        
        search_img = img[ystart:ystop, :, :]
        conv_color_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2YCrCb)
        
        
        if scale != 1:
            imshape = conv_color_img.shape
            conv_color_img = cv2.resize(conv_color_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        ch1 = conv_color_img[:,:,0]
        ch2 = conv_color_img[:,:,1]
        ch3 = conv_color_img[:,:,2]
        
        # Define number of blocks:
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 3  # All possible x blocks
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 # All possible y blocks
        nfeat_per_block = orient*cell_per_block**2 # Just a measurement of number of features
        
        # Window size is 64x64
        window = 64
        #Number of blocks per window
        nblocks_per_window = (window // pix_per_cell) - 1# can also mean no of cells / window
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        #nxsteps * nysteps = number of windows I think
        #print("nxsteps:",nxsteps,"nysteps:",nysteps, "number of windows:",nxsteps*(nysteps))
        
        # Compute individual channel HOG features for the entire image
        # This is an optimization much better than getting hog for each single window
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        # Sliding Window
        box_vis=[]
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                # Hog cells in x , y
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # Position of window in pixels in the scaled cropped image
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                
                # Extract the image patch covered by current window
                subimg = cv2.resize(conv_color_img[ytop:ytop+window, xleft:xleft+window], (window,window))
                
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
                
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                
                # Make prediction based on trained model 
                test_prediction = svc.predict(test_features) # Returns 1 if car is detected 0 if not
                
                if(visualization): # Draw a box over the real image
                    # To view scale and compare it to car sizes
                    xbox_left = np.int(xleft*scale) 
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    # Append Detection Position to list 
                    box_vis.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                # Detected cars go here
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    # Append Detection Position to list 
                    detected_boxes_list.append(((xbox_left+h_shift, ytop_draw+ystart),(xbox_left+win_draw+h_shift,ytop_draw+win_draw+ystart)))

        window_visited_list += [box_vis]
    if(visualization):
        return detected_boxes_list, window_visited_list
    return detected_boxes_list
def draw_boxes(img, bboxes, thickness=2):
    imcopy = [np.copy(img),np.copy(img),np.copy(img)]
    for i in range(len(bboxes)):
        for bbox in bboxes[i]:
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            color = (r,g,b)
            cv2.rectangle(img=imcopy[i], pt1=bbox[0], pt2=bbox[1],
                          color=color, thickness=thickness)
    return imcopy

#### Draw Detected Cars

def detected_boxes(img, dboxes, thickness=2):
    imcopy = np.copy(img)
    for dbox in dboxes:
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        color = (r,g,b)
        cv2.rectangle(img=imcopy, pt1=dbox[0], pt2=dbox[1],
                      color=color, thickness=thickness)
    return imcopy
#### Heatmap 

def add_heat(heatmap, boxes):
    # Iterate through list of bboxes
    for box in boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

# Removes false-postivies
def apply_threshold(heatmap, threshold):
    # create a copy to exclude modification of input heatmap
    heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    heatmap = np.clip(heatmap, 0, 255)
    # Return thresholded map
    return heatmap
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    imgcopy = np.copy(img)
#     r = random.randint(0,255)
#     g = random.randint(0,255)
#     b = random.randint(0,255)
    color = (255,20,147)
    for car_number in range(1, labels[1]+1): #car_number gets the value of pixel crossponding to the object
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero() #returns rows and columns of nonzero pixels
        # Identify x and y values of those pixels  
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))) # Diagonal of a rectangle
        # Draw the box on the image

        cv2.rectangle(imgcopy, bbox[0], (bbox[1][0]+10,bbox[1][1]-10), color, 2)
        cv2.putText(imgcopy,"Car",(bbox[0][0],bbox[0][1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    # Return the image
    return imgcopy

#### Video pipeline

dist_pickle = {}
dist_pickle = pickle.load( open("HOG/classifier.p", "rb" ) )
## Redefining parameters
params = {}
params['svc'] = dist_pickle['svc']
params['scaler'] = dist_pickle['scaler']
params['color_space'] ='YCrCb'
params['hog_channel'] ='ALL'
params['orient'] = 9
params['pix_per_cell'] = 8
params['cell_per_block'] = 2
params['cells_per_step'] = 2
params['spatial_size'] = (32,32)
params['hist_bins'] = 32
params['heat_threshold'] = 1
params['hist_range'] = (0,256)
params['ystart_ystop_scale'] = [(405, 510, 1),(400, 600, 1.5), (500, 710, 2) ] 
# Smoother detection
class PrevDetection():
    def __init__ (self):
        # Number labels to store
        self.stored_frames = []
        self.number_of_frames = 7 # Max number of frames to store

    # Put new frame
    def add_frame(self, detection):
        if (len(self.stored_frames) > self.number_of_frames): # pop oldest frame from the array of previous detections if array is full
            tmp = self.stored_frames.pop(0)
        self.stored_frames.append(detection) # add new frame to the array
    
    # Get last N frames
    def get_detections(self):
        detections = []
        for detection in self.stored_frames:
            detections.extend(detection) # add all previously detected frames into one array
        return detections

prev_detections = PrevDetection()

def process_image(img): 
    heat_threshold = params['heat_threshold'] 
    # Using Subsampled HOG windows to get possible detections 
    detection_list = detect_cars(img, params)
    
    #Smoothing part
    prev_detections.add_frame(detection_list)
    detection_list = prev_detections.get_detections()


    # Add heat to detections
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, detection_list)
    # Apply Threshold
    heatmap = apply_threshold(heatmap, heat_threshold)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Draw bounding box 
    result = draw_labeled_bboxes(np.copy(img), labels)
    
    return result

def video_output():
    global input_path,output_path
    myclip = VideoFileClip(input_path)
    clip = myclip.fl_image(process_image)
    clip.write_videofile(output_path, fps =25,audio=False)

video_output()


