from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import sys
#Global variables for bash
input_path = None
output_path = None
option = None  #uses -- or - notation 

n_args = len(sys.argv) #minimum is two
input_path = sys.argv[1]
output_path = sys.argv[2]
if(n_args>3):
    option = sys.argv[3]
# Histogram Equalizer

def hist_equalizer(img):
    equ = cv2.equalizeHist(img)
    return equ

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def gaussian_blur(img,kernel_size=5):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def draw_roi(img,isClosed=True,color=(255,0,0),thickness=5):
    x,y = (img.shape[1],img.shape[0])
    pts = np.array([[0.15*x,0.95*y], [0.43*x,int(0.65*y)],
                [0.58*x,0.65*y], [1*x,0.95*y]],
               np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], 
                      isClosed, color, 
                      thickness)
    return img

def perspective_transform(img,dst_size=(1280,720),inv=0):
    img_size=np.float32([(img.shape[1],img.shape[0])])
    #Region of Interest
    #Order is top left, top right, bottom left, bottom right
    src=np.float32([(0.43,0.65),(0.58,0.65),(0.15,0.95),(0.95,0.95)])
    dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
    srcPoints = src*img_size
    dstPoints = dst*np.float32(dst_size)
    if(inv):
        M = cv2.getPerspectiveTransform(dstPoints,srcPoints) #inverse
    else:
        M = cv2.getPerspectiveTransform(srcPoints,dstPoints) #Returns a matrix that transforms an Image
    warped_image = cv2.warpPerspective(img,M,dst_size)
    return warped_image 

def top_hat_filter(img):
    #to enhance bright objects of interest in a dark background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,3))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    mask = np.zeros_like(tophat)
    mask[((tophat >= 10)&(tophat<=150))] = 1
    return mask

def lane_filter(img):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = gaussian_blur(hls,5)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = gaussian_blur(lab,5)
    
    l_channel = lab[:,:,0]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]    
    
    #l_channel works relatively good under bridge and detects white lines
    tophat = top_hat_filter(l_channel)
    
    #Works well with to differentiate colors @sun/dirt
    #to detect the colored Lane
    s_thresh=(100, 255)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(tophat)
    combined_binary[(s_binary == 1) | (tophat == 1)] = 1
    # to eliminate noise around lane 
    #     kernel = np.ones((10,1),np.uint8)
#     s_binary = cv2.erode(s_binary,kernel,iterations = 2)
    return combined_binary

#Globally defined to store the parameters of past images
left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, 
                   nwindows=9, #Number of windows 
                   margin=100, #half window width 100?
                   minpix = 1 #minimum number of pixels to recenter the window 50? 
                  ):
    global left_a, left_b, left_c,right_a, right_b, right_c  # hn3raf ba3den
    left_fit_= np.empty(3) #parameters of 2nd order polynomial
    right_fit_ = np.empty(3) 
    out_img = np.dstack((img, img, img))*255 #Converts binary to 3 dimensional channel normal RGB
    
    #USING HISTOGRAM METHOD
    histogram = get_hist(img) 
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int32(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    # Indices in image is the coordinates the row is the height is the y
    # the column is the width is the x
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 3) 
        
        #Get non-zero pixels within each window by getting the indices of nonzerox
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0] 
        #[0] get indices of nonzerox only but you can access nonzeroy with it too
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        ''' 
        minpix is used to recenter the window and depends on how good the filter is
        if minpix is low it might be affected by the noise(?)
        therefore we keep minpix at 50 pixels for now to only recenter in the direction of any solid line
        '''
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        

    # Concatenate the arrays of indices
    #All pixels within all 9 windows
    left_lane_inds = np.concatenate(left_lane_inds) 
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    #All x and y coordinates for pixels residing inside the 9 windows
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    #Returns three parameters 
    left_fit = np.polyfit(lefty, leftx,
                          2 #Order of the equation
                         )
    right_fit = np.polyfit(righty, rightx, 2)
    
    #ay^2+by+c
    #store these three parameters
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    #find the mean for the last 10 frames
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting the two curves
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] ) #the points that will be on the y-axis
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2] 
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

#Overlay on frames
def draw_lanes(img, left_fit, right_fit,ploty):
#     ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    #Horizontal stack #dstack = depth stack #vstack = vertical stack
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))]) #flip up to down
    points = np.hstack((left, right)) 
    #Draw two curved lines
    cv2.polylines(color_img, np.int_(left) , False, (255,0,0),50) #Red curved line
    cv2.polylines(color_img, np.int_(right), False, (0,0,255),50) #Blue curved line
    #Draw a polygon of the curve shape
    cv2.fillPoly(color_img, np.int_(points), (0,255,0)) #Green curve
    inv_perspective = perspective_transform(color_img,inv=1)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective,color_img

def get_curve(img, leftx, rightx):
    # get image y-axis
    # linspace takes start, stop, number of steps
    # so here we will start from 0, reach to image last y point
    # and takes steps equal to all image y pixels
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    
    # these values are related to camera
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/730 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # center of x-axis
    car_pos = img.shape[1]/2
    
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10

    return (left_curverad, right_curverad, center)

def debugging_video_frame_by_frame(path):
    ##Live Video Capture Option for debugging mode
    #Note: previous image function not implemented yet
    cap = cv2.VideoCapture(path)
    _,frame = cap.read()
    while(cap.isOpened()):
        action = cv2.waitKey(1) & 0xFF
        if(action == ord('q')):
            break
        #press l to get next image
        if((action == ord('l'))):
            _,frame = cap.read()
#        frames.put(frame)
#        convert frame to RGB first
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        combo_image = final_output(frame)
#        convert combo_image to BGR before display
 
# the image outputs image size more than 1920x1080 and imshow doesn't allow resizing(?)
        img_size = (frame.shape[1],frame.shape[0])
        combo_image = cv2.resize(combo_image,img_size)
#         combo_image = cv2.cvtColor(combo_image, cv2.COLOR_RGB2BGR)
        #######################
        cv2.imshow("result",combo_image)
        
    cap.release()
    cv2.destroyAllWindows()

def final_output(img):
    # define font style
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (52, 119, 235) #RGB
    fontSize=2
    textPosition = (50,50)
    
    img_roi = np.copy(img)
    img_roi = draw_roi(img_roi) #Region of interest 1
    
    img_pip = lane_filter(img) #Returns binary image s_condition + l_channel tophat
    img_pip2 = np.dstack((img_pip,img_pip,img_pip))*255
    cv2.putText(img_pip2, 'Filtered Image', textPosition, font, fontSize, fontColor, 2)
    
   # Birdview
    img_pt = perspective_transform(img_pip, dst_size=(1280,720),inv=0) #birdview
    img_pt[int(img_pt.shape[0]//2):img_pt.shape[0],400:900] = 0
#     img_pt[int(img_pt.shape[0]/2):img_pt.shape[0],250:1000] = 0
    img_pt2 = np.dstack((img_pt,img_pt,img_pt))*255
    cv2.putText(img_pt2, 'Bird eye view', textPosition, font, fontSize, fontColor, 2)

    img_sw, curves, _, ploty = sliding_window(img_pt) #Sliding window image + curve points and parameters 
    cv2.putText(img_sw, 'Sliding window result', textPosition, font, fontSize, fontColor, 2)


    # get curves
    curverad =get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])

    img_final,birdview_curve = draw_lanes(img, curves[0], curves[1],ploty) #draws the overlay
    cv2.putText(birdview_curve, 'Polynomial fit', textPosition, font, fontSize, fontColor, 2)
    #Debugging mode?
    if(option =='-d' or option == '--debug'):

        # resize images
        #Bottom
        img_bot = np.hstack((img_sw,birdview_curve)) #Same width but half height
        img_bot = cv2.resize(img_bot, (img_final.shape[1],int(img_final.shape[0]/2)))
        img = np.vstack((img_final,img_bot))
        
        #Side
        img_stack = np.vstack((img_pip2,img_pt2))
        # img_stack = np.dstack((img_stack,img_stack,img_stack))*255
        img_side = np.vstack((img_roi,img_stack))
        img_side = cv2.resize(img_side, (int(img.shape[1]/2),img.shape[0]))
        img_final = np.hstack((img,img_side))

    cv2.putText(img_final, 'Lane Curvature: {:.0f} m'.format(lane_curve), (50, 50), font, 1, fontColor, 2)
    cv2.putText(img_final, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (50, 80), font, 1, fontColor, 2)
    return img_final

def video_output():
    global input_path,output_path
    myclip = VideoFileClip(input_path)#.subclip(40,43)
    clip = myclip.fl_image(final_output)
    clip.write_videofile(output_path, fps =25,audio=False)
    
video_output()