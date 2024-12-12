'''ssd - features of objects

frozen  -pre defined model

rstrip - remove extra spaces 

pip install opencv-python 
'''
import cv2  #computer vision

thres=0.5 # Threshold to detect object

cap = cv2.VideoCapture(0)   # Capture video from camera
cap.set(3, 648)    # width
cap.set(4, 448)    #height
cap.set(10, 70)    # adjust brightness

# class Names for Object Detection
className = []  # Empty List
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #Features (Input)
weightsPath = 'frozen_inference_graph.pb' # Model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  #sets input image size
net.setInputScale(1.0 / 127.5)  #input image pixel to normalize
net.setInputMean((127.5, 127.5, 127.5))  #normalizes the image by subtracting
net.setInputSwapRB(True)  #swap red blue channel

#Processing the video feed
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)
	#Displaying the detected objects
    if len(classIds) != 0:
    	for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        	cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        	cv2.putText(img, className[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        	cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    # display the result
    cv2.imshow("Output", img)
	
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break





