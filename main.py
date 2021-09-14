import numpy as np
import cv2, os, imutils, smtplib, winsound, os.path

from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



freq = 2500
duration = 100

def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)  
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=4)
        
        return (locs,preds)

prototxtPath=os.path.sep.join([r'C:/Users/hoya9/Desktop/Mask_detector/data','deploy.prototxt'])
weightsPath=os.path.sep.join([r'C:/Users/hoya9/Desktop/Mask_detector/data','res10_300x300_ssd_iter_140000.caffemodel'])

faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)

maskNet=load_model(r'C:/Users/hoya9/Desktop/Mask_detector/mobilenet_v2.model')

vs=VideoStream(src=0).start()
while True:
    #grab the frame from the threaded video stream and resize it
    #to have a maximum width of 400 pixels
    frame=vs.read()
    frame=imutils.resize(frame,width=800)
    
    #detect faces in the frame and preict if they are waring masks or not
    (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
    
    #loop over the detected face locations and their corrosponding loactions
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (fail_1, fail_2, p) = pred
        
        #determine the class label and color we will use to draw the bounding box and text
        if (fail_1 > fail_2) and (fail_1 > p):
            label = 'please wearing another mask'
        elif(fail_2 > fail_1) and (fail_2 > p):
            label = 'please wearing a mask'
        elif(p > fail_1) and (p > fail_2):
            label = 'good job'
            
        if label == 'good job':
            color = (0, 255, 0)

        elif label == 'please wearing another mask':
            color = (0, 255, 255)
        
        else:
            color = (0, 0, 255)
            winsound.Beep(freq, duration)
            # file_name = 'warning_person.jpg'
            # cv2.imwrite(file_name, frame)
            
            # file_location = 'warning_person.jpg'
            
            # filename = os.path.basename(file_location)
            # attachment = open(file_location, 'rb').read()


        # include the probability in the label
        label = '{}: {:.2f}%'.format(label, max(pred) * 100)
        
        #display the label and bounding boxes
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
    #show the output frame
    cv2.imshow("Mask_Dectecting",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if (key==ord('q') or key == ord('Q')):
        break
        
cv2.destroyAllWindows()
vs.stop()