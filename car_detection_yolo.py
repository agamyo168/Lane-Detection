from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import sys
import os

#Global variables for bash
input_path = None
output_path = None
option = None  #uses -- or - notation 

n_args = len(sys.argv) #minimum is two
input_path = sys.argv[1]
output_path = sys.argv[2]
if(n_args>3):
    option = sys.argv[3]


## Neural Network Setup ##

# Load yolo config and weights file
weights_path =os.path.join("YOLO","yolov3.weights")
config_path = os.path.join("YOLO","yolov3.cfg")
# Load Neural Net
net = cv2.dnn.readNetFromDarknet(config_path,weights_path) # Create neural network Yolo V3
# Get layer names 
names = net.getLayerNames() # Gets names for all Yolo layers
# Predication happen at layers 82 94 106
layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()] # Gets those unconnected layers i.e 82,94,106

## Read Coco file for labels ##

labels_path = os.path.join("YOLO","coco.names")
labels = open(labels_path).read().strip().split("\n")


## Output pipeline ##

def detect_cars_image(img):
    #Input for neural network is called blob object
    #cv.dnn.blobFromImage(img, scale, size, mean)
    (H,W,_) = img.shape
    img = np.copy(img)
    blob = cv2.dnn.blobFromImage(img,
                             1/255.0, #Normalize pixels to 0 and 1
                             (416,416), 
                             crop=False, 
                             swapRB=False 
                            )
    #A blob is a 4D numpy array object (images, channels, width, height)
    #Blob object is sent as an input to network
    net.setInput(blob)
    
    #start_t = time.time()
    
    #forward path to prediction layers
    layers_output = net.forward(layers_names) #The outputs object are vectors of lenght 85
    
    '''
    4x the bounding box (centerx, centery, width, height)
    1x box confidence
    80x class confidence
    '''
    
    #print("A forward path through yolov3 took {} Seconds".format(time.time()-start_t))
    
    boxes = [] # Boxes to draw points of a rectangle diagonal
    confidences = [] # returns percentage of confidence for each box
    classIDs = [] # returns the id of each class
    for output in layers_output: #loop over all detected objects
        for detection in output: #loop over 
            # Storing info about detected object
            scores= detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.85:
                #detection[:4] returns centerx,y and w,h that are percentage of real H,W
                box = detection[:4] * np.array([W,H,W,H]) # Scalar product
                bx, by, bw, bh = box.astype("int")


                x = int(bx-(bw/2)) #Since bx and by are not actually the center of the image
                y = int(by-(bh/2))

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.8,0.4) #returns array of idxs of good boxes
    # IOU = 0.4 to remove overlapping boxes with 40% with the highest score box
#     print(idxs)
  
    #Box drawing
    if(len(idxs)> 0):
        for i in idxs: # Why idxs.flatten()?
            (x,y) = [boxes[i][0], boxes[i][1]]
            (w,h) = [boxes[i][2], boxes[i][3]]
#             r = random.randint(0,255)
#             g = random.randint(0,255)
#             b = random.randint(0,255)
            color = (255,20,147)
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,"{}: {:.4f}".format(labels[classIDs[i]],
                                                confidences[i]) ,(x, y - 5),
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 0.5,
                                                 color,
                                                 2)
        
            
    return img
def video_output():
    global input_path,output_path
    myclip = VideoFileClip(input_path)
    clip = myclip.fl_image(detect_cars_image)
    clip.write_videofile(output_path,audio=False)

video_output()
